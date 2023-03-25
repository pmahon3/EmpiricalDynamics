from .weigher import weigher
import numpy as np


class exponential(weigher):
    def __init__(self, theta: float):
        super().__init__(theta=theta)

    def weigh(self,
              distance_matrix: np.array) -> np.array:
        """
        An exponentially normalized weighting with locality parametrized by theta. For vectors a,b in R^n the
        weighting is: weight = e^{(-theta * |a-b|)/d_bar} where |a-b| are given by the distance matrix. @param:
        distance_matrix_ is the distance matrix from a set of input points to the library points where
        distance_matrix_[i,j] is the distance from the ith input point to the jth library point in the embedding.
        """
        # Let division by zero result in zero for a weight of 1.0 for identical points
        return np.exp(-self.theta * distance_matrix / np.average(distance_matrix, axis=0))
