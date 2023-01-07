import datetime

import pandas as pd
import numpy as np
import networkx as nx

from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

from collections import namedtuple

from Code.src.main.python.EmpiricalDynamics.edynamics.modelling_tools.data_types.Lag import Lag


class Block:
    def __init__(self,
                 library_start: np.datetime64,
                 library_end: np.datetime64,
                 series: pd.Series,
                 frequency: pd.DateOffset,
                 lags: [Lag],
                 ):

        """
        Block defines a coordinate delay embedding of arbitrary lags from a given time series. It also defines basic
        operations on delay embeddings required for prediction schemes.
        """
        #: pd.Series: series to be embedded
        self.series = series
        #: the frequency spacing of the time series
        self.frequency = frequency
        #: List[lag]: list of lags used to create the embedding
        self.lags = lags
        #: pd.DataFrame: pandas dataframe of the delay embedding
        self.frame: pd.DataFrame = None
        #: int: dimension of the embedding, equal to the length of the list of lags. By default the embedding includes
        # the present value, (i.e. a lag of 0)
        self.dimension: int = None
        #: [pd.Timestamp, pd.Timestamp]: time range of points in the embedding used to make predictions
        library = namedtuple('library', 'start end')
        self.library = library(library_start, library_end)
        #: scipy.spatial.cKDTree: a KDTree storing the distances between all pairs of library points for the the delay
        # embedding using the l2 norm in R^n where n is the embedding dimension (i.e. number of lags, len(self.lags))
        self.distance_tree: cKDTree = None
        #: np.array: the l2 distances between all pairs of embedded points.  The distance between the ith and jth point
        # is at location X[m * i + j - ((i + 2) * (i + 1)) // 2], where m is the number of points.
        self.pairwise_distances: np.ndarray = None

    # PUBLIC
    def compile(self, mask_function=None) -> None:
        """
        compile builds the block according to the lags and library times, and constructs the KDTree and isometric
        mapping of embedded points. By convention these structures, with the exception of the block itself, do not
        include the last data point in the embedding, we cannot forecast/project from this point and so use it only as
        a point to forecast/project onto.
        @param mask_function:
        """
        # Build the embedding block
        self._build_block(mask_function=mask_function)
        # Build the KDTree
        self.distance_tree = cKDTree(self.frame.iloc[:-1])
        # Compute pairwise distances
        self.pairwise_distances = pdist(self.frame.values[:-1], metric='euclidean')

    def get_points(self, times: [np.datetime64]):
        """
        get_points retrieves the delay embedded points for any measurement in the series between the start and end
        dates
        @param times: the index of times for the desired points
        """
        if self.lags is None:
            return self.series.loc[times]

        points = pd.DataFrame(index=times, columns=[lag.lagged_name for lag in self.lags],
                              dtype=float)
        for time in points.index:
            points.loc[time] = [self.series.loc[time + self.frequency * lag.tau] for lag in self.lags]
        return points

    # Setters
    def set_series(self, new_series: pd.Series, frequency: pd.DateOffset):
        self.series = new_series
        self.frequency = frequency

    def set_lags(self, lags: [Lag]):
        self.lags = lags

    def set_library(self, library_start: np.datetime64, library_end: np.datetime64):
        library = namedtuple('library', 'start end')
        self.library = library(start=library_start, end=library_end)

    # PROTECTED
    def _build_block(self, mask_function=None) -> None:
        """
        Build the delay embedding block using lags specified for this block object
        @param mask_function:
        """
        self.frame = pd.concat([self.series[self.library.start:self.library.end].shift(periods=-lag.tau)
                                for lag in self.lags],
                               axis=1)
        self.frame.columns = [lag.lagged_name for lag in self.lags]
        if mask_function:
            self.frame = self.frame.loc[self.frame.apply(lambda x: mask_function(x.name), axis=1)]
        self.dimension = len(self.lags)
        self.frame.dropna(inplace=True)

    def _compute_pairwise_distances(self):
        self._pairwise_distances = pdist(self.frame.values)

    def _get_homogeneous_barycentric_coordinates(self, point: np.array) -> [[int], [float]]:
        """
        @param point: the point for which we want to find the homogenous barycentric coordinates using it's k nearest
        neighbours.
        @return: a list of two lists where the first are the integer indices of the minimal simplex and the second are
        the corresponding barycentric weights
        """
        # This works, for dimension n, by finding the n+1 nearest neighbours of the indexed point, p, using a KDTree,
        # and seeing whether the point is contained in the simplex formed by those neighbours. To determine this, the
        # barycentric coordinates of p with respect to the nearest neighbours is computed and if all weights are
        # positive, the point is in the simplex. If not the process is repeated with each nearest neighbour replaced
        # by the next closest neighbour until a bounding simplex for p is found. See the following for more details:
        #   https://math.stackexchange.com/questions/1226707/how-to-check-if-point-x-in-mathbbrn-is-in-a-n-simplex
        #   https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.KDTree.html

        # get the n+1 nearest neighbours of p, not including p
        # k_nn tracks which nearest neighbours we are querying for a valid simplex
        # len(k_nn) is always one greater than the dimension of the embedding
        # i.e. if nn = [2,3,4], simplex_candidates will be populated with the 2nd, 3rd, and 4th closest nearest
        # neighbours
        nearest_neighbours = [i for i in range(1, self.dimension + 2)]
        # starting with (k = self.dimension + 1) nearest neighbours, check if these neighbours from a simplex that
        # encloses the given point. If not, then replace the kth nearest neighbour with the k+1 nearest neighbour and
        # try again
        while True:
            simplex_candidates = self.distance_tree.query(point, k=nearest_neighbours)[1]
            T = self.frame.iloc[simplex_candidates[:-1]].values
            r_n = self.frame.iloc[simplex_candidates[-1]].values
            for i in range(len(T)):
                T[i] = T[i] - r_n
            T = T.T
            # todo: what are the cases when T is singular?
            #  (1) There are i points near p, q_1...q_i, where q_1=q_2...=q_i
            #       (ii) ?
            try:
                _lambda = np.matmul(np.linalg.inv(T), point - r_n)
                _lambda = np.append(_lambda, 1 - _lambda.sum())
            except np.linalg.LinAlgError:
                nearest_neighbours[-1] = nearest_neighbours[-1] + 1
                continue
            return [simplex_candidates, _lambda]
