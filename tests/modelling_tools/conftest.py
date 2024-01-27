import pandas as pd
import numpy as np
import logging
import pytest
import os

from typing import Callable, List

from edynamics.modelling_tools.embeddings import Embedding
from edynamics.modelling_tools.observers import Lag


def build_embedding(data_set: pd.DataFrame, n_lags: int, variable: str, split: float) -> Embedding:
    """
    Defines a delay embedding consisting of lagged observers, from 0, -1, ..., -n_lags, for the given variable in the
    data_set.

    :param pd.DataFrame data_set: which data set to embed
    :param str variable: the name of the variable to apply the lags to
    :param float split: the library/dataset split (0.5 ~ half of all data used in the library)
    :param List[Lag] n_lags: the number of lagged coordinates to apply to the variable
    :return Embedding: a state space embedding
    """
    observers = [Lag(variable_name=variable, tau=-1 * i) for i in range(n_lags)]
    library_times = data_set.index[n_lags:int(len(data_set.index) * split)]

    return Embedding(
        data=data_set,
        observers=observers,
        library_times=library_times,
        compile_block=True)


# test data
def generate_test_index(index: pd.DatetimeIndex, split: float) -> Callable:
    """
    Returns a function for randomly generating a sample of times that can be used to index a sample of
    data points. Each time the returned function is called a new random sample is drawn.

    :param pd.DatetimeIndex index: The index of a specific data set used in an embedding
    :param float split: the library/unobserved split applied on the index (0.5 ~ half of all data used in the library)
    :return: a function that can be used to randomly sample and generate a pandas datetime index based on a data set.
    """

    def generate_index(n_test_points: int) -> pd.DatetimeIndex:
        random_idx = np.random.choice(
            np.arange(int(len(index) * split) + 1, int(len(index)) - 1),
            size=n_test_points,
            replace=False
        )
        random_idx.sort()
        return index[random_idx]

    return generate_index


def train_validate_start_times(steps: int,
                               step_size: int,
                               times: pd.DatetimeIndex,
                               split: float):
    training = times[0:int(split * len(times))]
    test = times[len(training) + 1:]

    return training[::steps * step_size], test[::steps * step_size]


def sample_points(embedding: Embedding, n_test_points: int, split: float) -> pd.DataFrame:
    """
    Defines a sample of points used to make an evaluate predictions for each projector type. The points
    returned are the points from which predictions are made. For split values less than 1.0, an out of sample projection
    is performed based on points sampled from the last x=1.0-split portion of the dataset. For a split value of 1.0 a
    leave

    :param Embedding embedding: the generated embedding whose data we are drawing a sample of points from
    :param float split: the library/unobserved split applied on the index (0.5 ~ half of all data used in the library).
        If the split is 1.0 then it
    :param int n_test_points: the number of points to sample

    :return pd.DataFrame: the points sampled from the embedding data
    """
    index = generate_test_index(embedding.block.index, split)(n_test_points)
    return embedding.get_points(index)


# -FIXTURES------------------------------------------------------------------------------------------------------------#

@pytest.fixture(scope="class")
def embedding(data_set: pd.DataFrame, n_lags: int, split: float):
    variable = data_set.columns[0]

    embedding = build_embedding(data_set=data_set,
                                n_lags=n_lags,
                                variable=variable,
                                split=split
                                )
    return embedding


@pytest.fixture(scope="class")
def points(embedding: Embedding, n_test_points: int, split: float):
    return sample_points(embedding=embedding,
                         n_test_points=n_test_points,
                         split=split)
