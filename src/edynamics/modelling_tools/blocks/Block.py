import datetime

import pandas as pd
import numpy as np

from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay

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
        #: np.array: the l2 distances between all pairs of embedded points.  The distance between the ith and jth point
        # is at location X[m * i + j - ((i + 2) * (i + 1)) // 2], where m is the number of points.
        self.pairwise_distances: np.ndarray = None
        #: convex_hull: the convex hull of the library points, useful when trying to find the minimal bounding simplex
        # in the library points for a given point
        self._convex_hull: ConvexHull = None
        # delaunay_triangulation: the delaunay triangulation for the library points, useful when trying to find the
        # minimal bounding simplex in the library points for a given point
        self._delaunay_triangulation = None
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
        self._convex_hull = ConvexHull(self.frame)
        self._delaunay_triangulation = Delaunay(self.frame)

    def _compute_pairwise_distances(self):
        self._pairwise_distances = pdist(self.frame.values)

    def _get_simplex(self, points: [np.array]) -> [[int], [float]]:
        """
        @param points: the points for which we want to find minimal bounding simplex for. If a bounding simplex is not
        available then use the k nearest neighbours where k = n + 1 and n is the dimension of the embedding.
        @return: a list of the indices of the minimal simplex
        """
        simplices = [None for _ in points]
        for i, point in enumerate(points):
            # If the point lies within the convex hull of library points find the minimally bounding simplex...
            if self._in_hull(point):
                nearest_distance = np.inf

                for simplex in self._delaunay_triangulation.simplices:
                    point_simplex = [self.frame.iloc[j] for j in simplex]
                    distance = np.linalg.norm(np.array(point_simplex) - np.array(point))
                    if distance < nearest_distance:
                        simplices[i] = point_simplex
                        nearest_distance = distance
            # ...otherwise return the n+1 nearest neighbours where n is the dimension of the embedding
            else:
                simplices[i] = self.distance_tree.query(points, k=[i for i in range(1, self.dimension+2)])

    def _in_hull(self, point) -> bool:
        in_hull = True
        for equation in self._convex_hull.equations:
            if np.dot(point, equation[:-1]) > equation[-1]:
                in_hull = False
                break
        return in_hull
