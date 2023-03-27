import pandas as pd
import numpy as np
import abc


class observer(abc.ABC):
    def __init__(self, variable_name: str):
        self.variable_name = variable_name
        self.observation_name: str = None

    @abc.abstractmethod
    def observe(self,
                data: pd.DataFrame,
                time: pd.Timestamp) -> np.array:
        pass

    @abc.abstractmethod
    def observation_times(self,
                          frequency: pd.DatetimeIndex.freq,
                          time: pd.Timestamp):
        pass