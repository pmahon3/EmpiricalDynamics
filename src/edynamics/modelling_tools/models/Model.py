import pandas as pd
import numpy as np

from scipy.spatial import distance_matrix
from scipy.sparse import coo_matrix
from scipy.linalg import pinv
from scipy.sparse.csgraph import shortest_path
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from copy import deepcopy
from tqdm import tqdm

from edynamics.modelling_tools.blocks.Block import Block
from edynamics.modelling_tools.data_types.Lag import Lag

np.seterr(divide='ignore', invalid='ignore')


class Model:
    """
    Class Model provides  interface between the basic data structure of a delay embedding, a Block object, and defines
    the methods applied to the delay embedding including SMap projection, Simplex projection, and others.
    """

    def __init__(self, block: Block, target: str, theta: float, n_neighbors: int = None):
        #: Block: a delay embedding used for modelling the time series data
        self.block = block
        #: str: the name of the target column in the delay embedding to predict
        self.target = target
        #: pd.Series: time series data to model
        self.series: pd.Series = block.series
        #: float: locality parameter
        self.theta: float = theta
        #: scipy.sparse.coo_graph: An knn graph of the embedded data
        self.knn_graph: coo_matrix = None
        #: np.array: distance matrix for points/nodes in the knn graph
        self.knn_graph_distance_matrix = None
        # int: the number of nearest neighbours used to construct the knn graph
        self.n_neighbors: int = n_neighbors

    # PUBLIC
    def compile(self):
        """
        Compiles the Block assigned to this model and other ancillary structures used for delay embedding methods.
        """
        # Compile the block
        self.block.compile()
        if self.n_neighbors is not None:
            self._build_knn_graph(knn=self.n_neighbors)

    def predict(self, points: pd.DataFrame, method: str = 'knn', **method_kwargs) -> pd.DataFrame:
        if method == 'knn':
            return self._knn_projection(points=points)
        elif method == 'simplex':
            return self._simplex_projection(points=points)
        elif method == 'smap':
            return self._smap_projection(points)
        else:
            raise ValueError('Invalid method. Specified methods are:\n\tknn\n\tsimplex\n\tsmap')

    def dimensionality(self,
                       start: pd.Timestamp,
                       end: pd.Timestamp,
                       dimensions=10,
                       ) -> [float]:

        times = self.series.loc[start:end].index
        rhos = [None for i in range(dimensions)]

        temporary_model = deepcopy(self)
        lags = [Lag(self.target, tau=-i) for i in range(0, dimensions)]
        for i in tqdm(range(dimensions)):
            temporary_model.block.lags = lags[:i + 2]
            temporary_model.block.compile()

            x = temporary_model.block.get_points(times=self.series.loc[start:end].index)
            y = self.series.loc[start:end]

            y_hat = temporary_model._knn_projection(points=x, knn=None)

            rhos[i] = y_hat[self.target].corr(y[self.target])
        return rhos

    # Nonlinearity and Dimensionality estimators
    def nonlinearity(self,
                     start: pd.Timestamp,
                     end: pd.Timestamp,
                     thetas: [float] = np.linspace(0, 10, 11),
                     p: float = 2.0
                     ) -> [float]:
        temporary_block = [Lag]
        """
        nonlinearity estimates the optimal nonlinearity parameter, theta, for smap projections for a given range of
        observations
        @param start:
        @param end:
        @param thetas: the theta values to test. By default they are 1.0, 2.0, ... , 10.0
        @param p: which p to use when using minkowski norm for the metric, 2 by default.
        @return: a list of floats where the i-th entry is the correlation coefficient of smap-projections and observed
        values for the model target variable, for i-th theta input from thetas.
        """
        times = self.series.loc[start:end].index
        x = self.block.get_points(times)
        y = self.block.series.loc[times + self.block.frequency]
        rhos = [_ for _ in range(len(thetas))]

        for i, theta in enumerate(tqdm(thetas)):
            y_hat = self._smap_projection(x, theta=theta, p=p)
            y_hat = y_hat.droplevel(level=0)
            rhos[i] = y_hat[self.target].corr(y[self.target])
        return pd.DataFrame(rhos, index=thetas)

    # PROTECTED
    # Prediction Functions
    def _simplex_projection(self, points: pd.DataFrame) -> pd.DataFrame:
        pass

    def _knn_projection(self, points: pd.DataFrame, knn: int = None) -> pd.DataFrame:
        """
        Perform a simplex projection forecast from a given point using points in the embedding.
        @points: the points to be projected
        @knn: the number of nearest neighbours to use for each projection, one more than the dimensionality of the
        embedding by default
        @return: the forecasted points
        """

        indices = points.index + points.index.freq
        projections = np.empty(shape=(len(points), self.block.dimension))

        # Get barycentric coordinates of n+1 knn where n is the embeddign dimension
        weights, knn_idxs = self.block._get_weighted_knns(points.values, knn=knn)

        for i, (weight, knn_idx) in enumerate(zip(weights, knn_idxs)):
            projections[i, :] = np.matmul(weight, self.block.frame.iloc[knn_idx + 1].values) / weight.sum()

        return pd.DataFrame(data=projections, index=indices, columns=self.block.frame.columns)

    def _smap_projection(self, points: pd.DataFrame, theta: float = None, steps: int = 1, step_size: int = 1,
                         p: int = 2) -> pd.DataFrame:
        """
        Perform a S-Map projection from the given point. S-Map stands for Sequential locally weighted global linear
        maps. For a given predictor point in the embedding's state space, a weighted linear regression from all vectors
        in the embedding onto the point in question is made. The weights of each component of the regression is
        determined by the distance from the predictor. Each component of the regression is then iterated by a single
        time step and the linear map is used to determine the prediction from the predictor.
        @param points: an n-by-m pandas dataframe of m-dimensional lagged coordinate vectors, stored row-wise, to be
        projected according to the library of points
        @param theta: the nonlinearity parameter used for weighting library points
        @param steps: the number of prediction steps to make out from for each point. By default 1.
        period.
        @param step_size: the number to steps, of length given by the frequency of the block, to prediction.
        @return pd.DataFrame: the smap projected points
        @param p: which minkowski p-norm to use if using the minkowski metric.
        """
        # If theta isn't given use model theta value
        if theta is None:
            theta = self.theta

        indices = self._build_prediction_index(index=points.index, steps=steps, step_size=step_size)
        projections = pd.DataFrame(index=indices, columns=self.block.frame.columns, dtype=float)

        for i in range(len(points)):
            current_time = indices[i * steps][0]
            # Set up the regression
            # X is the library of inputs, the embedded points up to the starting point of the prediction period
            X = self.block.frame.loc[self.block.library.start:current_time][:-step_size]
            # y is the library of outputs, the embedding points at time t+1
            y = self.block.frame.loc[self.block.library.start:current_time][step_size:]

            point = points.values[i]
            for j in range(steps):
                prediction_time = indices[i * steps + j][-1]
                # Get the current state, i.e. point we are predicting from

                # Compute the weights
                distance_matrix_ = self._minkowski(points=point[np.newaxis, :], max_time=current_time, p=p)
                weights = self._exponential(distance_matrix_, theta=theta)
                weights[np.isnan(weights)] = 0.0

                # A is the product of the weights and the library X points, A = w * X
                A = weights * X.values
                # B is the product of the weights and the library y points, A = w * y
                B = weights * y.values
                # Solve for C in B=AC via SVD
                C = np.matmul(pinv(A), B)
                projections.loc[(current_time, prediction_time)] = np.matmul(point, C)

                # todo: replace predictions for lagged variables for either actual values or previous predicted values
                for lag in self.block.lags:
                    lag_time = prediction_time + self.block.frequency * lag.tau
                    if lag.tau == 0:
                        pass
                    elif lag_time <= current_time:
                        projections.loc[(current_time, prediction_time)][lag.lagged_name] = \
                            self.block.get_points([lag_time])[lag.variable_name]
                    elif lag_time > current_time:
                        projections.loc[(current_time, prediction_time)][lag.lagged_name] = \
                            projections.loc[projections.index.get_level_values(level=1) == lag_time].iloc[-1][
                                lag.variable_name]

                point = projections.loc[(current_time, prediction_time)]
        return projections

    # Weighting Functions
    def _exponential(self, distance_matrix_, theta: float = None):
        """
        An exponentially normalized weighting with locality parametrized by theta. For vectors a,b in R^n the
        weighting is: weight = e^{(-theta * |a-b|)/d_bar} where |a-b| are given by the distance matrix. @param:
        distance_matrix_ is the distance matrix from a set of input points to the library points where
        distance_matrix_[i,j] is the distance from the ith input point to the jth library point in the embedding.
        """
        if theta is None:
            theta = self.theta
        return np.exp(-theta * (distance_matrix_ / np.average(distance_matrix_, axis=0)))

    # TODO: experimental, these require normalization?
    def _tricubic(self, distance_matrix_, theta: float, scaler: str = None):
        # Set default theta to model theta
        if theta is None:
            theta = self.theta

        # Scale/standardize distance matrix
        # TODO: I still don't really get theta parameterization and scaling/standardizing here...
        if scaler == 'minmax':
            distance_matrix_ = MinMaxScaler().fit_transform(distance_matrix_.flatten().reshape(-1, 1))
        else:
            pass

        return (1 - (distance_matrix_ / theta)) ** 3 * np.heaviside(1 - (distance_matrix_ / theta), 1.0)

    # todo: experimental
    def _epanechnikov(self, distance_matrix_, theta: float, scaler: str = None):
        # Set default theta to model theta
        if theta is None:
            theta = self.theta

        # Scale/standardize distance matrix
        # TODO: I still don't really get theta parameterization and scaling/standardizing here...
        if scaler == 'minmax':
            distance_matrix_ = MinMaxScaler().fit_transform(distance_matrix_.flatten().reshape(-1, 1))
        elif scaler == 'standard':
            distance_matrix_ = StandardScaler().fit_transform(distance_matrix_.flatten().reshape(-1, 1))

        return (3 / 4) * (1 - (distance_matrix_ / theta) ** 2) * np.heaviside(1 - (distance_matrix_ / theta), 1.0)

    # metrics
    # These are the metrics defining distances of input points to library points to be used in the weighting kernels
    # (...listed above)
    def _minkowski(self, points: np.ndarray, max_time: pd.Timestamp, p: int = 2) -> np.ndarray:
        """
        The minkowski p norm for the latent phase space, R^n.
        @points: the points for which the pairwise distances to the library points are computed
        @max_time: the current time of the prediction. Only points embedded in block up to this time will be used
        to build the distance matrix.
        @return: np.ndarray distance matrix from points to the library embedded points
        """
        return distance_matrix(self.block.frame.loc[self.block.library.start:max_time][:-1].values,
                               points,
                               p=p)

    # other helpers
    def _build_prediction_index(self, index: pd.Index, steps: int, step_size: int) -> pd.MultiIndex:
        """
        @param index: the index of times from which to make predictions
        @param steps: the number of prediction steps to make out from for each time. By default 1.
        @param step_size: the number to steps, of length given by the frequency of the block, to prediction.
        @return pd.MultiIndex: multi index of points where the first index is the starting point for each multi step
        prediction which are given in the second index. E.g. index (t_4, t_10) is the prediction of t_10 made on a
        multistep prediction starting at t_4.
        """
        tuples = list(
            zip(
                index.repeat(repeats=steps),
                # todo: this doesn't work in the degenerative case where step_size = 0
                sum(zip(*[index + self.block.frequency * (step_size + i) for i in range(steps)]), ())
            )
        )
        return pd.MultiIndex.from_tuples(tuples=tuples, names=['Current_Time', 'Prediction_Time'])

    # todo: experimental
    def _build_knn_graph(self, knn: int) -> [coo_matrix, np.array]:
        """
        Build the k nearest neighbours graph, and corresponding distance matrix, of the embedding data.
        @param knn: the number of nearest neighbours used to build the graph.
        """
        # Only include up to the last point in the library set, the last point is reserved for the output of the linear
        # regression
        M = len(self.block.frame) - 1
        # Get the k nearest neighbours and distances of each point in the embedding set
        knn_dists, knn_idxs = self.block.distance_tree.query(self.block.frame[:-1],
                                                             k=[i for i in range(2, knn + 2)])
        idxs = [idx for idx in range(M) for dist in knn_dists[idx]]
        knn_idxs = [nn for nn_list in knn_idxs for nn in nn_list]
        knn_dists = [dist for dist_list in knn_dists for dist in dist_list]
        self.knn_graph = coo_matrix(arg1=(knn_dists, (idxs, knn_idxs)),
                                    shape=(M, M))
        self.knn_graph_distance_matrix = shortest_path(self.knn_graph, directed=False)
