import abc

import pandas as pd


class Observer(abc.ABC):
    def __init__(self, variable_name: str):
        self.variable_name = variable_name
        self.observation_name: str

    @abc.abstractmethod
    def observe(self, data: pd.DataFrame, times: pd.DatetimeIndex) -> pd.Series:
        pass

    @abc.abstractmethod
    def observation_times(self, frequency: pd.DatetimeIndex.freq, time: pd.DatetimeIndex):
        pass

    @abc.abstractmethod
    def __eq__(self, other):
        pass

    @abc.abstractmethod
    def __hash__(self):
        pass
