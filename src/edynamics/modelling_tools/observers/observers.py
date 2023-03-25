from .observer import observer
import pandas as pd
import numpy as np


class lag(observer):
    def __init__(self, variable_name: str, tau: int):
        """
        The lagged observation function. Observes the given variable at some non-positive specified multiple of the
        data frequency.

        :param str variable_name: the variable to be observed
        :param int tau: the lag multiple.
        """
        super().__init__(variable_name=variable_name)

        self.tau = tau
        if tau == 0:
            self.observation_name = variable_name
        else:
            self.observation_name = variable_name + '_(t' + str(tau) + ')'

    def observe(self, data: pd.DataFrame, time: pd.Timestamp) -> pd.Series:
        """
        Applies the observation function to the data at the given time.

        :param pd.DataFrame data: the data the observation function is applied to.
        :param pd.Timestamp time: the time of the observation.
        :return: pd.Series
        """
        return data.loc[time + data.index.freq * self.tau][self.variable_name]

    def observation_times(self,
                          frequency: pd.DatetimeIndex.freq,
                          time: pd.Timestamp) -> [pd.Timestamp]:
        """
        Determines the lagged time of the observation function given an observation time

        :param pd.DatetimeIndex.freq frequency: the data the observation function is applied to.
        :param pd.Timestamp time: the time of the observation.
        :return: the times required to compute the moving average
        """
        return [time + self.tau * frequency]


class lag_moving_average(observer):
    def __init__(self, variable_name: str, q: int, tau: int = -1):
        super().__init__(variable_name)

        self.q = q
        self.tau = tau
        if tau == 0 and q == 0:
            self.observation_name = variable_name
        else:
            self.observation_name = variable_name + '_(MA_q=' + str(q) + '_\u03C4=' + str(tau) + ')'

    def observe(self, data: pd.DataFrame, time: pd.Timestamp) -> np.array:
        return data.loc[time + data.index.freq * self.q * self.tau:time][::self.tau].mean().values[0]

    def observation_times(self, frequency: pd.DatetimeIndex.freq, time: pd.Timestamp):
        return [time + frequency * self.tau * i for i in range(self.q+1)]


