import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix

from .norm import norm

from edynamics.modelling_tools.embeddings import Embedding


class minkowski(norm):
    def __init__(self, p: int = 2):
        """
        The minkowski p norm. (I.e. for p=2, the euclidean norm)

        :param p: which power p to to use for the norm
        """
        self.p = p

    def distance_matrix(self,
                        embedding: Embedding,
                        points: np.array,
                        max_time: pd.Timestamp) -> np.ndarray:
        """
        Returns the distance matrix from given points to the embedded points.

        :param Embedding embedding: the state space Embedding.
        :param np.array points: the points for which the pairwise distances to the library points are computed
        :param pd.Timestamp max_time: the current time of the prediction. Only points embedded in block up to this time will be used
            to build the distance matrix.
        :return: the distance matrix from points to each Embedding points. The ij-th entry gives the distances from the
            i-th point in points to the j-th point in the Embedding.
        """
        return distance_matrix(embedding.block.loc[min(embedding.library_times):max_time].values, points, p=self.p)
