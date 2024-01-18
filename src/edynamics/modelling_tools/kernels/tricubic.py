from .kernel import Kernel
import numpy as np


class Tricubic(Kernel):
    def __init__(self, theta: float):
        super().__init__(theta=theta)

    def weigh(self, distance_matrix: np.array) -> np.array:
        # todo: distance matrix needs to be scaled
        return (1 - (distance_matrix / self.theta)) ** 3 * np.heaviside(
            1 - (distance_matrix / self.theta), 1.0
        )

    def __repr__(self):
        return f"Tricubic Kernel:\n" \
               f"\tTheta:{self.theta}"

    def __str__(self):
        return f"Tricubic Kernel:\n" \
               f"\tTheta:{self.theta}"
