import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix

from edynamics.modelling_tools.embeddings import Embedding
from .norm import Norm


class Minkowski(Norm):
    def __init__(self, p: int = 2):
        """
        The minkowski p Norm. (I.e. for p=2, the euclidean Norm)

        :param p: which power p to to use for the Norm
        """
        self.p = p

    def distance_matrix(
        self, embedding: Embedding, points: np.array, times: pd.DatetimeIndex
    ) -> np.ndarray:
        """
        Returns the distance matrix from given points to the embedded points.

        :param Embedding embedding: the state space Embedding.
        :param np.array points: the points for which the pairwise distances to the library points are computed
        :param pd.Timestamp times: the times in the library indexing the points we want the distances to.
        :return: the distance matrix from points to each Embedding points. The ij-th entry gives the distances from the
            i-th point in points to the j-th point in the Embedding.
        """
        return distance_matrix(
            embedding.block.loc[times].values,
            points,
            p=self.p,
        )

    def __eq__(self, other):
        if isinstance(other, Minkowski):
            return self.p == other.p
        return False
    