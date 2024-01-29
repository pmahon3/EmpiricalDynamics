import os
import uuid
import pytest
import logging
import random

import matplotlib.pyplot as plt

from src.edynamics.modelling_tools.estimators import dimensionality
from src.edynamics.modelling_tools import Embedding

logger = logging.getLogger(__name__)


@pytest.mark.parametrize('data_set, n_lags, split, n_test_points',
                         [({'type': 'lorenz', 'n_points': 1000}, 3, 1.0, 100)
                          ],
                         indirect=['data_set'],
                         scope="class")
class TestPerformance:
    logging.info("<dimensionality> performance testing...")

    @pytest.mark.parametrize('steps, step_size',
                             [(1, 1),
                              (5, 1),
                              (10, 1),
                              (1, 5),
                              (1, 10)])
    def test_dimensionality(self,
                            embedding: Embedding,
                            steps: int,
                            step_size: int,
                            n_test_points: float,
                            image_paths
                            ):
        # for step_sizes greater than 1 increase the lag of all lagged observers to at least the step size to avoid
        #   multistep predictions that place observation functions on unobserved or unpredicted data.
        #   as it stands the embedding should be a regular delay embedding of n_lags at a spacing of 1.
        if step_size > 1:
            for obs in embedding.observers[1:]:
                obs.tau = obs.tau - step_size
        # also need to increase the start of the library times just in case the library starts where the data starts and
        #   a lagged observer at the first library time/start of the data will be unobservable
        embedding.library_times = embedding.library_times[10:]

        target = embedding.block.columns[0]
        times = embedding.library_times[10:-(steps * step_size)]
        idx_sample = random.sample(range(len(times)), n_test_points)
        times = times[idx_sample]

        dimensionality_ = dimensionality(
            embedding=embedding,
            target=target,
            steps=steps,
            step_size=step_size,
            times=times
        )

        unique_id = uuid.uuid4()
        figure_path = os.path.join(os.getenv('PYTEST_REPORT_IMAGES'), 'dimensionality_' + str(unique_id) + '.png')
        plt.plot(dimensionality_)
        plt.xlabel("E")
        plt.ylabel('rho')
        plt.savefig(figure_path)
        plt.close()
        image_paths.append(figure_path)

        logger.info('\nDimensionality:\n' + dimensionality_.to_string())

        assert 1
