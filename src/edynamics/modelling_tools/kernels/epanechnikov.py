from .kernel import Kernel
import numpy as np


class epanechnikov(Kernel):
    def __init__(self, theta: float):
        super().__init__(theta=theta)

    def weigh(self,
              distance_matrix: np.array) -> np.array:
        # todo: distance matrix requires scaling
        return (3 / 4) * (1 - (distance_matrix / self.theta) ** 2) * np.heaviside(1 - (distance_matrix / self.theta),
                                                                                  1.0)
