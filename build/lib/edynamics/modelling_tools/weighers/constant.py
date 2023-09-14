from .weigher import weigher
import numpy as np


class constant(weigher):
    def __init__(self):
        super().__init__(theta=None)

    def weigh(self,
              distance_matrix: np.array):
        return np.ones(shape=distance_matrix.shape)
