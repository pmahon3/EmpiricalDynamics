import os
import uuid
import pytest
import logging
import random

import matplotlib.pyplot as plt

from edynamics.modelling_tools.estimators import nonlinearity
from edynamics.modelling_tools import Embedding

from ..conftest import generate_test_index

logger = logging.getLogger(__name__)


@pytest.mark.parametrize('data_set, n_lags, split, n_test_points',
                         [({'type': 'lorenz', 'n_points': 1000}, 3, 1.0, 100)
                          ],
                         indirect=['data_set'],
                         scope="class")
class TestPerformance:

    @pytest.mark.parametrize('steps, step_size',
                             [(1, 1),
                              (5, 1),
                              (10, 1),
                              (1, 5),
                              (1, 10)])
    def test_nonlinearity(self,
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
            embedding.set_library(embedding.library_times[-embedding.observers[-1].tau:])
            embedding.compile()

        target = embedding.block.columns[0]
        times = embedding.library_times[-embedding.observers[-1].tau:-(steps*step_size)]
        idx_sample = random.sample(range(len(times)), n_test_points)
        times = times[idx_sample]
        nonlinearity_ = nonlinearity(
            embedding=embedding,
            target=target,
            steps=steps,
            step_size=step_size,
            times=times
        )

        unique_id = uuid.uuid4()
        figure_path = os.path.join(os.getenv('PYTEST_REPORT_IMAGES'), 'nonlinearity_' + str(unique_id) + '.png')
        plt.plot(nonlinearity_)
        plt.xlabel("theta")
        plt.ylabel('rho')
        plt.savefig(figure_path)
        plt.close()
        image_paths.append(figure_path)

        logger.info('\nNonlinearity:\n' + nonlinearity_.to_string())

        assert 1
