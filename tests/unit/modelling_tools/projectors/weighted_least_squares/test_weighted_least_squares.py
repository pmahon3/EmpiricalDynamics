from edynamics.modelling_tools.projectors import WeightedLeastSquares

from ..conftest import *

logger = logging.getLogger(__name__)


@pytest.fixture(scope="class")
def projector(embedding):
    return WeightedLeastSquares()


@pytest.mark.parametrize('data_set, n_lags, split, n_test_points',
                         [({'type': 'lorenz', 'n_points': 100}, 3, 0.5, 10)],
                         indirect=['data_set'],
                         scope="class")
class TestSmoke:
    @pytest.mark.parametrize('steps,    step_size',
                             [(1, 1),
                              (1, 3),
                              (3, 1),
                              (3, 3)]
                             )
    def test_build_prediction_index(self,
                                    projector: Projector,
                                    embedding: Embedding,
                                    points: pd.DataFrame,
                                    steps: int,
                                    step_size: int):
        build_prediction_index(projector=projector, embedding=embedding, points=points, steps=steps,
                               step_size=step_size)

    # predict k projections for given point
    @pytest.mark.parametrize('steps,    step_size',
                             [(1, 1)]
                             )
    def test_projection(self,
                        embedding: Embedding,
                        projector: Projector,
                        points: pd.DataFrame,
                        steps: int,
                        step_size: int):
        projection(embedding=embedding, projector=projector, points=points, steps=steps, step_size=step_size)


@pytest.mark.parametrize('data_set, n_lags, split, n_test_points, n_samples, pae_sup, pcd_inf, pcd_sup, theta',
                         [({'type': 'lorenz', 'n_points': 1000}, 3, 0.5, 100, 100, 0.1, 0.95, 1.05, 1.0),
                          ({'type': 'duffing', 'n_points': 1000}, 3, 0.5, 100, 100, 0.1, 0.95, 1.05, 1.9)],
                         indirect=['data_set'],
                         scope="class")
class TestPerformance:
    """
    Forecasting performance tests for the KNearestNeighbours projector object.
    """

    @pytest.mark.parametrize('steps, step_size',
                             [(1, 1)])
    def test_projection_performance(self,
                                    projector: WeightedLeastSquares,
                                    embedding: Embedding,
                                    theta: float,
                                    split: float,
                                    n_test_points: int,
                                    steps: int,
                                    step_size: int,
                                    n_samples: int,
                                    pae_sup: float,
                                    pcd_inf: float,
                                    pcd_sup: float,
                                    request):
        projector.kernel.theta = theta
        projection_performance(projector=projector, embedding=embedding, split=split, n_test_points=n_test_points,
                               steps=steps, step_size=step_size, n_samples=n_samples, pae_sup=pae_sup,
                               pcd_inf=pcd_inf, pcd_sup=pcd_sup, request=request)
