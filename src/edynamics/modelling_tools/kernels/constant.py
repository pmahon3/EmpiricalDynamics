from .kernel import Kernel
import numpy as np


class constant(Kernel):
    def __init__(self):
        super().__init__(theta=None)

    def weigh(self,
              distance_matrix: np.array):
        return np.ones(shape=distance_matrix.shape)
