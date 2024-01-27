from copy import deepcopy

import pandas as pd
import ray
from ray.util.multiprocessing import Pool
from tqdm import tqdm

from edynamics.modelling_tools.embeddings import Embedding
from edynamics.modelling_tools.observers import Lag
from edynamics.modelling_tools.projectors import Projector, KNearestNeighbours
from edynamics.modelling_tools.norms import Minkowski
from edynamics.modelling_tools.kernels import Exponential


def dimensionality(
        embedding: Embedding,
        target: str,
        times: pd.DatetimeIndex,
        max_dimensions: int = 10,
        steps: int = 1,
        step_size: int = 1,
        compute_pool: Pool = None,
        verbose: bool = True
) -> pd.DataFrame:
    """
    Successively adds lags from l=0,-1,-2, up to l=-n where n is given by 'dimensions', evaluating the prediction
    skill using a simplex projection.
    :param Embedding embedding: the delay embedded block.
    :param projector projector: which prediction method to use.
    :param str target: which embedding variable to predict.
    :param pd.DatetimeIndex times: the times to predict from.
    :param int max_dimensions: the maximum number of lags to add to the model.
    :param int steps: the number of steps in a multistep prediction to make where successive predictions are made
        using previous predictions.
    :param int step_size: the step size of each prediction as a multiple of the sampling frequency of the DATA.
    :param compute_pool: a ray computing pool if parallel computing.
    :param verbose: if true a progress bar will be printed. By default, this is false.
    :return: a dataframe indexed by the number of dimensions and corresponding prediction skill, as measured by
        Pearson's correlation coefficient.
    """

    projector: KNearestNeighbours = None


    # Save original observation functions
    original_observers = embedding.observers

    # initialize dataframe to return
    rhos = pd.DataFrame(
        data=[None for _ in range(max_dimensions)],
        index=[i + 1 for i in range(max_dimensions)],
        columns=["rho"],
    )

    # create pool of lags to draw on, one for each dimension, lag multiples of -1
    lags = [Lag(variable_name=target, tau=-i) for i in range(0, max_dimensions)]

    # Run predictions for each dimension
    futures = []
    if compute_pool is not None:
        args = []
        for i in range(max_dimensions):
            embedding_copy = deepcopy(embedding)
            embedding_copy.observers = lags[: i + 1]
            embedding_copy.compile()
            points_ = embedding_copy.get_points(times=times)

            projector = KNearestNeighbours(
                k=embedding_copy.dimension+1,
                norm=Minkowski(p=2),
                kernel=Exponential(theta=0.0)
            )

            args.append([embedding_copy, projector, target, points_, steps, step_size])

        futures = compute_pool.starmap(dimensionality_parallel_step.remote, args)

    else:
        pbar = tqdm(range(max_dimensions), leave=True, disable=verbose)
        for i in pbar:
            pbar.set_description("E = " + str(i + 1))
            embedding.observers = lags[: i + 1]
            embedding.compile()
            points_ = embedding.get_points(times=times)

            projector = KNearestNeighbours(
                k=embedding.dimension+1,
                norm=Minkowski(p=2),
                kernel=Exponential(theta=0.0)
            )

            futures.append(
                dimensionality_step(
                    embedding=embedding,
                    projector=projector,
                    target=target,
                    points=points_,
                    steps=steps,
                    step_size=step_size,
                )
            )

    if compute_pool is not None:
        results = []
        for result in tqdm(futures, disable=verbose):
            results.append(ray.get(result))

    else:
        results = futures

    for i, result in enumerate(results):
        rhos.loc[i + 1] = result

    # restore original observers
    embedding.observers = original_observers

    return rhos


def dimensionality_step(
    embedding: Embedding,
    projector: Projector,
    target: str,
    points: pd.DataFrame,
    steps: int,
    step_size: int,
) -> float:
    y_hat = projector.project(embedding=embedding, points=points, steps=steps, step_size=step_size)

    y = embedding.data.loc[y_hat.index.droplevel(level=0)][target]

    return y_hat.droplevel(level=0)[target].corr(y)


@ray.remote
def dimensionality_parallel_step(
    embedding: Embedding,
    projector: Projector,
    target: str,
    points: pd.DataFrame,
    steps: int,
    step_size: int,
) -> float:
    return dimensionality_step(
        embedding=embedding,
        projector=projector,
        target=target,
        points=points,
        steps=steps,
        step_size=step_size,
    )
