import datetime

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist, pdist
from scipy.spatial import ConvexHull

from itertools import combinations
from collections import namedtuple

from edynamics.modelling_tools.data_types.Lag import Lag


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

        self._convex_hull: ConvexHull = None

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
            points.loc[time] = [self.series.loc[time + self.frequency * lag.tau].values[0] for lag in self.lags]
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

    def _get_simplex_or_nn_idxs(self, points: [np.array]) -> [[int]]:
        """
        Returns the minimally bounding simplex, from the library points, for each of the given points. If a given point
        lies outside the cloud of library points, then the k nearest neighbours of that point are returned where
        k = n + 1 and n is the dimension of the embedding.
        @param points: the points for which we want to find minimal bounding simplex for.
        @return: a list of the library indices of the minimal simplices or knn's.
        """
        simplices = [None for _ in points]

        for i, point in enumerate(points):
            # If the point lies within the convex hull of library points find the minimally bounding simplex...
            if self._in_hull(point):
                simplices[i] = self._get_minimal_bounding_simplex(point)
            # ...otherwise return the n+1 nearest neighbours where n is the dimension of the embedding
            else:
                simplices[i] = self.distance_tree.query(points, k=[i for i in range(1, self.dimension + 2)])
        return simplices

    def _in_hull(self, point, tolerance=np.finfo(float).eps*2) -> bool:
        return all(np.dot(equation[:-1], point) + equation[-1] <= tolerance for equation in self._convex_hull.equations)

    def _get_minimal_bounding_simplex(self, point) -> list[int]:
        # todo: extremely inefficient at high dimensions (?)
        if self._in_hull(point):
            k = self.dimension + 2
            _, knn_idxs = self.distance_tree.query(point, k=range(1, k))
            candidates = list(map(list, combinations(knn_idxs, self.dimension + 1)))

            while True:
                _lambdas = self._get_barycentric_coordinates(point, candidates)
                result = [not (_lambda < 0).any() for _lambda in _lambdas]

                if any(result):
                    break

                k += 1
                _, knn_idx = self.distance_tree.query(point, k=[k])
                knn_idxs = np.append(knn_idxs, knn_idx)
                new_candidates = list(map(list, list(combinations(knn_idxs, self.dimension + 1))))

                candidates = list(map(list, set(map(tuple, new_candidates)).difference(set(map(tuple, candidates)))))

            return
        else:
            pass

    def _get_weighted_knns(self, point, max_time: np.datetime64, knn: int):
        # Set defualt knn to one greater than the embedding dimension
        if knn is None:
            knn = self.dimension + 1

        knn_idxs = np.empty(shape=(knn), dtype=int)
        count = 0
        k = 1
        while count < knn:
            _, knn_idx = self.distance_tree.query(point, [k])
            if self.frame.index[knn_idx[0]] <= max_time:
                knn_idxs[count] = knn_idx[0]
                count += 1
            k += 1

        _min = np.abs(cdist(point[:, np.newaxis].T, self.frame.iloc[knn_idxs])).min()
        weights = np.exp(-np.abs(cdist(point[:, np.newaxis].T, self.frame.iloc[knn_idxs])) / _min)

        return weights, knn_idxs

    def _get_barycentric_coordinates(self, point, simplex_idxs) -> np.array:
        """
        Computes the barycentric coordinates of a set of points with respet to the simplices denoted by the indices
        within the embedded block.
        :param point: the point for which the barycentric coordinates are to be returned
        :param simplex_idxs: a list containing the indices for the simplex used to compute the coordinates
        for the given points.
        :return: an n x m numpy array of the barycentric coordinates where row n provides the barycentric coordinates
        for the nth point in points relative the the nth simplex given by simplex_idxs.
        """
        coords = np.empty(shape=(1, self.dimension + 1))
        simplex = self.frame.iloc[simplex_idxs].values
        T = (simplex[:-1, :] - simplex[-1, :])
        coords[:, :-1] = np.linalg.pinv(T.T) @ (point - simplex[-1, :]).T
        coords[:, -1] = 1 - coords[:, :-1].sum()

        return coords
