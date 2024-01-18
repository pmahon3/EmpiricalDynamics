from typing import List

import numpy as np
import pandas as pd
import ray

from ray.util.multiprocessing import Pool
from tqdm import tqdm

from edynamics.modelling_tools.embeddings import Embedding
from edynamics.modelling_tools.projectors import Projector, WeightedLeastSquares


def nonlinearity(
        embedding: Embedding,
        target: str,
        times: pd.DatetimeIndex,
        thetas: List[float] = np.linspace(0, 10, 11),
        steps: int = 1,
        step_size: int = 1,
        compute_pool: Pool = None,
        verbose: bool = True
) -> pd.DataFrame:
    """
    Estimates the optimal nonlinearity parameter, theta, for weighted_least_squares projections for a given set of
    observations.

    :param Embedding embedding: the state space Embedding.
    :param str target: the column in the block to predict.
    :param pd.DatetimeIndex times: the times to predict from.
    :param [float] thetas: the theta values to test. By default they are 1.0, 2.0, ... , 10.0.
    :param int steps: the number of steps in a multistep prediction to make where successive predictions are made
        using previous predictions.
    :param int step_size: the step size of each prediction as a multiple of the sampling frequency of the DATA.
    :param compute_pool: a ray computing pool if parallel computing.
    :param verbose: if true a progress bar will be printed. By default, this is false.
    :return: a dataframe of prediction skill, as measured by Pearson's correlation coefficient, indexed by dimension.
    """
    projector = WeightedLeastSquares()

    rhos = pd.DataFrame(
        data=[None for _ in range(len(thetas))], index=thetas, columns=["rho"]
    )

    # Run predictions for each dimension
    futures = []
    if compute_pool is not None:
        args = []
        for i, theta in enumerate(thetas):
            args.append([embedding, theta, projector, target, times, steps, step_size])

        futures = compute_pool.starmap(nonlinearity_parallel_step.remote, args)

    else:
        pbar = tqdm(thetas, leave=True, disable=verbose)
        for i, theta in enumerate(pbar):
            pbar.set_description("\u03b8 = " + str(round(theta, 4)))

            futures.append(
                nonlinearity_step(
                    embedding=embedding,
                    theta=theta,
                    projector=projector,
                    target=target,
                    times=times,
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
        rhos.iloc[i] = result

    return rhos


def nonlinearity_step(
        embedding: Embedding,
        theta: float,
        projector: Projector,
        target: str,
        times: pd.DatetimeIndex or pd.Index,
        steps: int,
        step_size: int,
) -> float:
    projector.kernel.theta = theta

    # Projection inputs
    x = embedding.get_points(times)

    # Projection outputs
    times = projector.build_prediction_index(
        frequency=embedding.frequency,
        index=times,
        steps=steps,
        step_size=step_size,
    ).get_level_values(level=1)

    y = embedding.get_points(times=times)

    # Projection
    y_hat = projector.project(embedding=embedding, points=x, steps=steps, step_size=step_size, leave_out=True)

    return y_hat.droplevel(level=0)[target].corr(y[target])


@ray.remote
def nonlinearity_parallel_step(
        embedding: Embedding,
        theta: float,
        projector: Projector,
        target: str,
        times: pd.DatetimeIndex,
        steps: int,
        step_size: int,
) -> float:
    return nonlinearity_step(
        embedding=embedding,
        theta=theta,
        projector=projector,
        target=target,
        times=times,
        steps=steps,
        step_size=step_size,
    )
