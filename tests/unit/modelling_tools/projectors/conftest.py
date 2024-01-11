import pandas as pd
import numpy as np
import logging
import pytest
import os

from typing import Callable

from edynamics.modelling_tools.embeddings import Embedding
from edynamics.modelling_tools.observers import Lag
from edynamics.modelling_tools.projectors import Projector

from tests.utils import plot_error_histogram

logger = logging.getLogger(__name__)


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


def sample_points(embedding: Embedding, n_test_points: int, split: float) -> pd.DataFrame:
    """
    Defines a sample of points used to make an evaluate predictions for each projector type. The points
    returned are the points from which predictions are made.

    :param Embedding embedding: the generated embedding whose data we are drawing a sample of points from
    :param float split: the library/unobserved split applied on the index (0.5 ~ half of all data used in the library)
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


# -COMMON-TESTS--------------------------------------------------------------------------------------------------------#
def projection_performance(projector: Projector,
                           embedding: Embedding,
                           split: float,
                           n_test_points: int,
                           steps: int,
                           step_size: int,
                           n_samples: int,
                           pae_sup: float,
                           pcd_inf: float,
                           pcd_sup: float,
                           request):
    # set up frames for recording sample data
    sample_index = pd.Index(data=np.arange(0, n_samples))
    stats_index = pd.Index(data=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])
    multi_index = pd.MultiIndex.from_product([sample_index, stats_index], names=['sample_id', 'statistic'])

    percent_correct_direction = pd.DataFrame(index=multi_index,
                                             columns=[obs.observation_name for obs in embedding.observers])
    percent_absolute_error = pd.DataFrame(index=multi_index,
                                          columns=[obs.observation_name for obs in embedding.observers])

    # perform predictions
    for i in range(n_samples):
        # get a sample of non-library points for predictions
        points = sample_points(embedding=embedding,
                               n_test_points=n_test_points,
                               split=split)
        actuals = embedding.get_points(times=points.index + embedding.frequency)

        # perform the predictions
        projections = projector.project(
            embedding=embedding,
            points=points,
            steps=steps,
            step_size=step_size)

        # compute percent absolute error statistics
        error = actuals - projections.droplevel(0)

        percent_absolute_error.loc[i] = (error / actuals).abs().describe().values

        # compute percent correct direction statistics
        actual_change = actuals - points.values
        predicted_change = projections.droplevel(0) - points.values

        percent_correct_direction.loc[i] = (predicted_change / actual_change).describe().values

    # compute sample estimates
    mean_percent_absolute_error_hat = percent_absolute_error.xs(
        key='mean',
        level='statistic'
    ).mean()

    mean_percent_correct_direction_hat = percent_correct_direction.xs(
        key='mean',
        level='statistic'
    ).mean()

    # plot error histograms and attach to report
    image_paths = [[
        plot_error_histogram(
            embedding=embedding,
            data=percent_absolute_error,
            statistic='mean',
            title="percent_absolute_error",
            figure_path=os.getenv('PYTEST_REPORT_IMAGES') + '/weighted_least_squares/percent_absolute_error_' +
                        str(projector.__class__) + '.png',
            n_samples=n_samples),

        plot_error_histogram(
            embedding=embedding,
            data=percent_correct_direction,
            statistic='mean',
            title="percent_correct_direction",
            figure_path=os.getenv('PYTEST_REPORT_IMAGES') + '/weighted_least_squares/percent_correct_direction.png' +
                        str(projector.__class__) + '.png',
            n_samples=n_samples)
    ]]
    request.node._image_paths = image_paths

    # log distribution info
    logger.info('mean_percent_absolute_error_statistics:')
    logger.info('\n' + percent_absolute_error.xs(
        key='mean',
        level='statistic'
    ).astype('float').describe().to_string())

    logger.info('mean_percent_correct_direction_statistics:')
    logger.info('\n' + percent_correct_direction.xs(
        key='mean',
        level='statistic'
    ).astype('float').describe().to_string())

    assert (mean_percent_absolute_error_hat < pae_sup).all()
    assert (pcd_inf <= mean_percent_correct_direction_hat).all() and \
           (mean_percent_correct_direction_hat <= pcd_sup).all()


# build prediction index for given points
def build_prediction_index(projector: Projector,
                           embedding: Embedding,
                           points: pd.DataFrame,
                           steps: int,
                           step_size: int):
    index = projector.build_prediction_index(embedding.frequency, points.index, steps, step_size)

    assert isinstance(index, pd.MultiIndex)
    assert len(index) == len(points) * steps
    assert index.names == ['Current_Time', 'Prediction_Time']


# predict k projections for given point
def projection(projector: Projector,
               embedding: Embedding,
               points: pd.DataFrame,
               steps: int,
               step_size: int):
    projections = projector.project(embedding, points, steps, step_size)

    assert isinstance(projections, pd.DataFrame)
    assert len(projections) == len(points) * steps
    assert set(projections.columns) == set(embedding.block.columns)
