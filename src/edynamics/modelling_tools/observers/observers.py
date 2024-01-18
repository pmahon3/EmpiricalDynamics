import numpy as np
import pandas as pd

from .observer import Observer


# todo: these need to be "vectorized"
class Lag(Observer):
    def __init__(self, variable_name: str, tau: int):
        """
        The lagged observation function. Observes the given variable at some non-positive specified multiple of the
        DATA frequency.

        :param str variable_name: the variable to be observed
        :param int tau: the lag multiple.
        """
        super().__init__(variable_name=variable_name)

        self.tau = tau
        if tau == 0:
            self.observation_name = variable_name
        else:
            self.observation_name = variable_name + "_(t" + str(tau) + ")"

    def __str__(self):
        return f"Lag(var_name={self.variable_name},tau={self.tau})"

    def observe(self,
                data: pd.DataFrame,
                times: pd.Timestamp) -> pd.Series:
        """
        Applies the lag observation function to the data at the given times.

        :param data:    the data the observation function is applied to.
        :param times:   the time of the observation.
        :return:        the lagged data point

        :type times:    pd.Timestamp or pd.DatetimeIndex
        :type data:     pd.DataFrame
        :rtype:         pd.Series
        """
        # Get the lagged time indices
        index = times + data.index.freq * self.tau

        if not all(index.isin(data.index)):
            raise KeyError(f"Application of the lagged observer {self.__str__()} to the data results in invalid indices"
                           f" at {index.difference(data.index)}")

        return data.loc[index][self.variable_name]

    def observation_times(
            self, frequency: pd.DatetimeIndex.freq, time: pd.Timestamp
    ) -> [pd.Timestamp]:
        """
        Determines the lagged time of the observation function given an observation time

        :param pd.DatetimeIndex.freq frequency: the DATA the observation function is applied to.
        :param pd.Timestamp time: the time of the observation.
        :return: the times required to compute the moving average
        """
        return [time + self.tau * frequency]

    def __eq__(self, other):
        return self.variable_name == other.variable_name and self.tau == other.tau

    def __hash__(self):
        return hash((self.variable_name, self.tau))


class LagMovingAverage(Observer):
    def __init__(self, variable_name: str, q: int, tau: int = -1):
        super().__init__(variable_name)

        self.q = q
        self.tau = tau

        if tau == 0 and q == 0:
            self.observation_name = variable_name
        else:
            self.observation_name = (
                    variable_name + "_(MA_q=" + str(q) + "_\u03C4=" + str(tau) + ")"
            )

    def observe(self, data: pd.DataFrame, times: pd.Timestamp) -> np.array:
        # Calculate the start time
        start_time = times + data.index.freq * self.q * self.tau

        # Resample the data at intervals of self.tau starting from start_time up to 'time'
        # Compute the mean of the resampled data
        resampled_data = data[start_time:times].resample(f'{self.tau}T').mean()

        # Extract the series for the specified variable_name
        # This will return a pandas Series
        return resampled_data[self.variable_name]

    def observation_times(self, frequency: pd.DatetimeIndex.freq, time: pd.Timestamp):
        return [time + frequency * self.tau * i for i in range(self.q + 1)]

    def __eq__(self, other):
        return (
                self.variable_name == other.variable_name
                and self.q == other.q
                and self.tau == other.tau
        )

    def __hash__(self):
        hash((self.variable_name, self.tau, self.q))
