import abc
import numpy as np


class Kernel(abc.ABC):
    def __init__(self, theta: float):
        self.theta = theta

    @abc.abstractmethod
    def weigh(self, distance_matrix: np.array):
        raise NotImplementedError

    def __repr__(self):
        return f"Kernel:\n" \
               f"\tTheta:{self.theta}"

    def __str__(self):
        return f"Kernel:\n" \
               f"\tTheta:{self.theta}"
