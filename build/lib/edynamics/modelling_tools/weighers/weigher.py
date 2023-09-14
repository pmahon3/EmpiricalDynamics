import abc
import numpy as np


class weigher(abc.ABC):
    def __init__(self,
                 theta: float):
        self.theta = theta

    @abc.abstractmethod
    def weigh(self,
              distance_matrix: np.array):
        raise NotImplementedError
