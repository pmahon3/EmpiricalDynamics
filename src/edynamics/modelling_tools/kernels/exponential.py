from .kernel import Kernel
import numpy as np
import logging

from copy import copy

logger = logging.getLogger(__name__)


class Exponential(Kernel):
    def __init__(self, theta: float):
        super().__init__(theta=theta)

    def weigh(self, distance_matrix: np.array) -> np.array:
        """
        An exponentially normalized weighting with locality parametrized by theta. For vectors a,b in R^n the
        weighting is: weight = e^{(-theta * |a-b|)/d_bar} where |a-b| are given by the distance matrix.

        :param np.array distance_matrix: is the distance matrix from a set of input points to the library points where
            distance_matrix_[i,j] is the distance from the ith input point to the jth library point in the Embedding.

        """
        # Let division by zero result in zero for a weight of 1.0 for identical points
        average_distances = np.average(distance_matrix[0], axis=0)
        average_distances = np.where(average_distances == 0, np.inf, average_distances)
        return np.exp(
            -self.theta * distance_matrix / average_distances
        )

    def __repr__(self):
        return f"Exponential Kernel:\n" \
               f"\tTheta:{self.theta}"

    def __str__(self):
        return f"Exponential Kernel Object:\n" \
               f"\tTheta:{self.theta}"
