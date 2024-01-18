import pandas as pd
import numpy as np
import logging
import os

from edynamics.modelling_tools.embeddings import Embedding
from edynamics.modelling_tools.projectors import Projector

from tests.utils import plot_error_histogram
from ..conftest import sample_points

logger = logging.getLogger(__name__)


# -COMMON-TESTS--------------------------------------------------------------------------------------------------------#
def projection_performance(projector: Projector,
                           embedding: Embedding,
                           split: float,
                           n_test_points: int,
                           steps: int,
                           step_size: int,
                           n_samples: int,
                           image_paths) -> None:
    # set up frames for recording sample data
    sample_index = pd.Index(data=np.arange(0, n_samples))
    stats_index = pd.Index(data=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])
    multi_index = pd.MultiIndex.from_product([sample_index, stats_index], names=['sample_id', 'statistic'])

    percent_correct_direction = pd.DataFrame(index=multi_index,
                                             columns=[obs.observation_name for obs in embedding.observers])
    percent_absolute_error = pd.DataFrame(index=multi_index,
                                          columns=[obs.observation_name for obs in embedding.observers])
    correlations = pd.DataFrame(index=sample_index,
                                columns=[obs.observation_name for obs in embedding.observers])

    # perform predictions
    for i in range(n_samples):
        # get a sample of non-library points for predictions
        points = sample_points(embedding=embedding,
                               n_test_points=n_test_points,
                               split=split)

        # perform the predictions
        projections = projector.project(
            embedding=embedding,
            points=points,
            steps=steps,
            step_size=step_size,
            leave_out=True
        )

        # Compute the sample errors. The case where steps > 1 needs to be handled separately.
        if steps == 1:
            # actual/observed and predictions
            actuals = embedding.get_points(times=points.index + embedding.frequency)
            error = actuals - projections.droplevel(0)

            # Pearson's correlation
            correlations.loc[i] = projections.droplevel(0).corrwith(actuals)

            # percent absolute error
            percent_absolute_error.loc[i] = (error / actuals).abs().describe().values

            # percent correct direction
            actual_change = actuals.diff()
            predicted_change = projections.droplevel(0).diff()
            percent_correct_direction.loc[i] = (predicted_change / actual_change).describe().values

        # If a multistep prediction is performed, the errors need to be computed for each multistep leg. These then
        #   determine the overall sample error.
        # fixme: currently the prediction periods for the multistep predictions may overlap, given that the starting
        #   points are randomly sampled and can be as close as sequential. Does this mess with the independence of the
        #   within sample errors?
        else:

            # data structures to store prediction error stats
            pearsons_total = 0
            # percent absolute error and percent correct direction are aggregated and their statistics described for
            #   the sample
            percent_absolute_errors = pd.DataFrame(columns=embedding.block.columns, index=projections.index,
                                                   dtype=float)
            percent_correct_directions = pd.DataFrame(columns=embedding.block.columns, index=projections.index,
                                                      dtype=float)

            # loop through each of the multistep predictions
            for current_time in points.index:
                # actual/observed and predictions
                actuals = embedding.get_points(projections.loc[current_time].index)
                predictions = projections.xs(key=current_time, level="Current_Time")

                # actual v. predicted change for percent correct direction error
                actual_change = actuals.diff()
                predicted_change = predictions.diff()

                # raw error for percent absolute error
                error = actuals - predictions

                # Pearson's correlation
                pearsons_total += predictions.corrwith(actuals)
                # percent absolute error
                percent_absolute_errors.loc[current_time, :] = (error / actuals).abs().values
                # percent correct direction
                percent_correct_directions.loc[current_time] = (predicted_change / actual_change).values

            # compute the overall pearson correlation and statistical descriptions of percent absolute error and percent
            #   correct direction
            correlations.loc[i] = pearsons_total / n_test_points
            percent_absolute_error.loc[i] = percent_absolute_errors.droplevel(0).describe().values
            percent_correct_direction.loc[i] = percent_correct_directions.droplevel(0).describe().values

    # compute sample estimates
    mean_percent_absolute_error_hat = percent_absolute_error.xs(
        key='mean',
        level='statistic'
    ).mean()

    mean_percent_correct_direction_hat = percent_correct_direction.xs(
        key='mean',
        level='statistic'
    ).mean()

    correlation_hat = correlations.mean()

    # error histograms and attach to report
    new_image_paths = [
        plot_error_histogram(embedding=embedding, data=percent_absolute_error, statistic='mean',
                             title="percent_absolute_error",
                             figure_directory=os.getenv('PYTEST_REPORT_IMAGES'),
                             n_samples=n_samples),
        plot_error_histogram(embedding=embedding, data=percent_correct_direction, statistic='mean',
                             title="percent_correct_direction",
                             figure_directory=os.getenv('PYTEST_REPORT_IMAGES'),
                             n_samples=n_samples),
        plot_error_histogram(embedding=embedding, data=correlations, statistic=None,
                             title="correlations",
                             figure_directory=os.getenv('PYTEST_REPORT_IMAGES'),
                             n_samples=n_samples)
    ]

    # log distribution info
    logger.info('\nmean_percent_absolute_error_statistics:\n' + percent_absolute_error.xs(
        key='mean',
        level='statistic'
    ).astype('float').describe().to_string())

    logger.info('\nmean_percent_correct_direction_statistics:\n' + percent_correct_direction.xs(
        key='mean',
        level='statistic'
    ).astype('float').describe().to_string())

    logger.info('\ncorrelation_statistics:\n' + correlations.astype('float').describe().to_string())

    for path in new_image_paths:
        image_paths.append(path)

    assert 1


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
    projections = projector.project(embedding=embedding, points=points, steps=steps, step_size=step_size,
                                    leave_out=True)

    assert isinstance(projections, pd.DataFrame)
    assert len(projections) == len(points) * steps
    assert set(projections.columns) == set(embedding.block.columns)
