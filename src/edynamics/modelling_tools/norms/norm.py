import abc

import numpy as np
import pandas as pd

from edynamics.modelling_tools.embeddings import Embedding


class norm(abc.ABC):
    @abc.abstractmethod
    def distance_matrix(self,
                        block: Embedding,
                        points: np.ndarray,
                        max_time: pd.Timestamp) -> np.ndarray:
        """
        Abstract method defining computation of distance matrices from a given set of points to other points in a
        delay embedding.
        :points: the points for which the pairwise distances to the library points are computed
        :block_frame: the library points to compute the pairwise distances to.
        :max_time: the current time of the prediction. Only points embedded in block up to this time will be used
        to build the distance matrix.
        :return: np.ndarray distance matrix from points to the library embedded points
        """
        raise NotImplementedError
