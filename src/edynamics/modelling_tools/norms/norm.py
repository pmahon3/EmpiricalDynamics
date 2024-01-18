import abc

import numpy as np
import pandas as pd

from edynamics.modelling_tools.embeddings import Embedding


class Norm(abc.ABC):
    @abc.abstractmethod
    def distance_matrix(
        self, embedding: Embedding, points: np.ndarray, times: pd.DatetimeIndex
    ) -> np.ndarray:
        """
        Abstract method defining computation of distance matrices from a given set of points to other points in a
        delay Embedding.
        :param Embedding embedding: the state space Embedding.
        :param points: which points to compute distance matrices from.
        :param pd.Timestamp times: the times in the library indexing the points we want the distances to.
        to build the distance matrix.
        :return: np.ndarray distance matrix from points to the library embedded points
        """
        raise NotImplementedError
