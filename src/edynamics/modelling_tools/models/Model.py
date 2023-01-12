import pandas as pd
import numpy as np

from scipy.spatial import distance_matrix
from scipy.sparse import coo_matrix
from scipy.linalg import pinv
from scipy.sparse.csgraph import shortest_path
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from edynamics.modelling_tools.blocks.Block import Block

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
            self.build_knn_graph(knn=self.n_neighbors)

    # todo: experimental
    def build_knn_graph(self, knn: int) -> [coo_matrix, np.array]:
        """
        Build the k nearest neighbours graph, and corresponding distance matrix, of the embedding data.
        @param knn: the number of nearest neighbours used to build the graph.
        """
        # Only include up to the last point in the library set, the last point is reserved for the output of the linear
        # regression
        M = len(self.block.frame) - 1
        # Get the k nearest neighbours and distances of each point in the embedding set
        knn_dists, knn_idxs = self.block.distance_tree.query(self.block.frame[:-1], k=[i for i in range(2, knn + 2)])
        idxs = [idx for idx in range(M) for dist in knn_dists[idx]]
        knn_idxs = [nn for nn_list in knn_idxs for nn in nn_list]
        knn_dists = [dist for dist_list in knn_dists for dist in dist_list]
        self.knn_graph = coo_matrix(arg1=(knn_dists, (idxs, knn_idxs)),
                                    shape=(M, M))
        self.knn_graph_distance_matrix = shortest_path(self.knn_graph, directed=False)

    def naive_projection(self, points: pd.DataFrame) -> pd.DataFrame:
        """
        The naive projection takes the nearest library time point and projects it forward one time step to make provide
        a prediction.
        """
        points = points.values
        projections = np.empty(shape=points.shape, dtype=float)
        for point in range(len(points)):
            nn = self.block.distance_tree.query(point, k=1)[1]
            projections[point] = point * (point / self.block.frame.iloc[nn].values)

    # todo: experimental
    def simplex_projection(self, points: pd.DataFrame) -> pd.DataFrame:
        """
        Perform a simplex projection forecast from a given point using points in the embedding.
        @points: the points to be projected
        @return: the forecasted embedding vector
        """
        columns = points.columns
        index = points.index + points.index.freq
        points = points.values
        projections = np.empty(shape=points.shape, dtype=float)
        for point in range(len(points)):
            simplex = self.block._get_simplex(point=points[point])
            projections[point] = simplex[1].T @ self.block.frame.iloc[simplex[0] + 1].values
        return pd.DataFrame(data=projections, columns=columns, index=index)

    # S-Map
    def smap_projection(self, points: pd.DataFrame, theta: float = None, steps: int = 1,  step_size: int = 1, p: int = 2) -> np.array:
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
            current_time = indices[i*steps][0]
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
                            projections.loc[projections.index.get_level_values(level=1) == lag_time].iloc[-1][lag.variable_name]

                point = projections.loc[(current_time, prediction_time)]
        return projections

    # Nonlinearity and Dimensionality estimators
    def nonlinearity(self,
                     start: pd.Timestamp,
                     end: pd.Timestamp,
                     thetas: [float] = np.linspace(0, 10, 11),
                     p: float = 2.0
                     ) -> [float]:
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
        percent_correct_direction = [_ for _ in range(len(thetas))]
        for i, theta in enumerate(thetas):
            y_hat = self.smap_projection(x, theta=theta, p=p)
            y_hat = y_hat.droplevel(level=0)
            rhos[i] = y_hat[self.target].corr(x[self.target])
            percent_correct_direction[i] = y_hat[self.target] - y / y.diff()
            percent_correct_direction[i].replace([np.inf, -np.inf], np.nan, inplace=True)
            percent_correct_direction[i].dropna(inplace=True)
            percent_correct_direction[i] = percent_correct_direction[i].mean()
        return pd.DataFrame({'rhos': rhos, '%_correct_direction': percent_correct_direction}, index=thetas)

    def connectedness(self,
                      points: pd.DataFrame,
                      n_neighbors: [int],
                      theta: float = None
                      ) -> [float]:
        """
        nonlinearity estimates the optimal nonlinearity parameter, theta, for smap projections for a given range of
        observations
        @param points: an n-by-m pandas dataframe of m-dimensional lagged coordinate vectors, stored row-wise, to be
        used in the nonlinearity estimation.
        @param n_neighbors: the number of nearest neighbors to build corresponding isometric embeddings.
        @param theta: the nonlinearity parameter for the weighting used in the S-Map projections.
        @return: a list of floats where the i-th entry is the correlation coefficient of smap-projections and observed
        values for the model target variable, for i-th theta input from thetas.
        """
        if theta is None:
            theta = self.theta
        rhos = [_ for _ in range(len(n_neighbors))]
        percent_maes = [_ for _ in range(len(n_neighbors))]
        for i, n_neighbor in enumerate(n_neighbors):
            # Update knn graph
            self.build_knn_graph(knn=n_neighbor)
            projections = self.smap_projection(points, theta=theta)
            rhos[i] = projections[self.target].corr(points[self.target])
            percent_maes[i] = ((projections[self.target] - points[self.target]).abs()
                               / points[self.target].abs()).mean() * 100
        # Return knn graph to original state
        self.build_knn_graph(knn=self.n_neighbors)
        return pd.DataFrame({'rhos': rhos, '%_maes': percent_maes}, index=n_neighbors)

    # PROTECTED
    def _get_jacobian(self, points: pd.DataFrame, theta: float):
        """
        Computes the jacobian of for a given point in the state space of the delay embedding.
        """
        if theta is None:
            theta = self.theta

        points = points.values
        jacobians = [_ for _ in points]
        # TODO: Would it be better to approximate the manifold M, based on the data, D, and define a metric on M which
        #   could be used to compute distances for each x_i, x_j D?

        # X is the library of inputs, the embedding points at time t
        X = self.block.frame.loc[self.block.library.start:self.block.library.end][:-1]
        # y is the library of outputs, the embedding points at time t+1
        y = self.block.frame.loc[self.block.library.start:self.block.library.end][1:]
        # todo: needs to be fixed for minkowski function signature including max_time
        distance_matrix_ = self._minkowski(points, p=2)

        weights = self._exponential(distance_matrix_, theta=theta)
        for i in range(len(points)):
            # A is the product of the weights and the library X points, A = w * X
            A = weights[i][:, np.newaxis][:-1] * X.values
            # B is the product of the weights and the library y points, A = w * y
            B = weights[i][:, np.newaxis][:-1] * y.values
            # Solve for C in B=AC via SVD
            jacobians[i] = np.matmul(pinv(A), B)

        return jacobians

    # todo: experimental
    def _get_hessian(self, points: pd.DataFrame, theta: float):
        """
        Computes an approximation of the hessian matrix for a given point in the state space of the delay embedding.
        """
        if theta is None:
            theta = self.theta

        points = points.values
        hessians = [_ for _ in points]

        # X is the library of inputs, the embedding points at time t
        X = self.block.frame.loc[self.block.library.start:self.block.library.end][:-1]
        # y is the library of outputs, the embedding points at time t+1
        y = self.block.frame.loc[self.block.library.start:self.block.library.end][1:]
        # todo: needs to be fixed for minkowski function signature including max_time
        distance_matrix_ = self._minkowski(points, p=2)

        weights = self._exponential(distance_matrix_, theta=theta)
        for i in range(len(points)):
            # A is the product of the weights and the library X points, A = w * X
            A = weights[i][:, np.newaxis][:-1] * X.values
            # B is the product of the weights and the library y points, A = w * y
            B = weights[i][:, np.newaxis][:-1] * y.values
            # Solve for C in B=AC via SVD
            hessians[i] = np.matmul(pinv(A), B)

        return hessians

    # Weighting Kernels
    # These are the functions for introducing state-space dependence in the local linear regression performed by
    # smap_projection. The distance_matrices are given by a given metric function (...listed below these functions)
    def _exponential(self, distance_matrix_, theta: float):
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
