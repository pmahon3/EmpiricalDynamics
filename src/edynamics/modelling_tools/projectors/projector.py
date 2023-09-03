import abc

import numpy as np
import pandas as pd
import ray

from edynamics.modelling_tools.embeddings import embedding
from edynamics.modelling_tools.norms import norm
from edynamics.modelling_tools.weighers import weigher


class projector(abc.ABC):
    def __init__(self,
                 norm_: norm,
                 weigher_: weigher):
        """
        Abstract class defining state space based prediction strategies.

        :param norm_: which vector norm to use.
        :param weigher_: which weighting kernel to use.
        """
        self.norm = norm_
        self.weigher = weigher_

    @abc.abstractmethod
    def predict(self,
                embedding_: embedding,
                points: pd.DataFrame,
                steps: int,
                step_size: int) -> pd.DataFrame:
        """
        Abstract method defining state spaced based prediction methods for predictions of delay Embedding points
        :param embedding_: the state space Embedding.
        :param points: the points to be projected.
        :param steps: the number of prediction steps to make out from for each point. By default 1.
        period.
        :param step_size: the number to steps, of length given by the frequency of the block, to prediction.
        :return: the projected points.
        """
        raise NotImplementedError

    @staticmethod
    def update_values(embedding_: embedding,
                      predictions: pd.DataFrame,
                      current_time: np.datetime64,
                      prediction_time: np.datetime64) -> pd.DataFrame:
        """
        Updates a given predicted point in a predicted data block by replacing the variables with either the
        actual variables, if available, or the projected variables from previous predictions.
        :return: the updated dataframe of projections.
        """
        for obs in embedding_.observers:
            # Get the times needed to make an observation for this observer at this prediction time
            obs_times = obs.observation_times(frequency=embedding_.frequency, time=prediction_time)

            unobserved = [time for time in obs_times if time > current_time]
            observed = [time for time in obs_times if time <= current_time]

            data = pd.concat([predictions.loc[current_time].loc[unobserved][obs.variable_name],
                              embedding_.data.loc[observed][obs.variable_name],
                              ]).to_frame()

            data.sort_index(inplace=True)
            hit = False

            if data.index.inferred_freq is not None:
                data.index.freq = data.index.inferred_freq
            else:
                data.index.freq = embedding_.frequency

            predictions.loc[(current_time, prediction_time)][obs.observation_name] = obs.observe(data=data,
                                                                                                 time=prediction_time)

        return predictions

    @staticmethod
    def build_prediction_index(frequency: pd.DatetimeIndex.freq,
                               index: pd.Index,
                               steps: int,
                               step_size: int) -> pd.MultiIndex:
        """
        :param frequency: the frequency denoting the time span between predictions.
        :param index: the index of times from which to make predictions
        :param steps: the number of prediction steps to make out from for each time. By default 1.
        :param step_size: the number to steps, of length given by the frequency of the block, to prediction.
        :return pd.MultiIndex: multi index of points where the first index is the starting point for each multi step
        prediction which are given in the second index. E.g. index (t_4, t_10) is the prediction of t_10 made on a
        multistep prediction starting at t_4.
        """
        tuples = list(
            zip(
                index.repeat(repeats=steps),
                # todo: this doesn't work in the degenerative case where step_size = 0
                sum(zip(*[index + frequency * (step_size + i) for i in range(steps)]), ())
            )
        )

        return pd.MultiIndex.from_tuples(tuples=tuples, names=['Current_Time', 'Prediction_Time'])
