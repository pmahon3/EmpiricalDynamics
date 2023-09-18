import ray
import pandas as pd
import numpy as np
import logging

from tqdm import tqdm
from copy import deepcopy
from ray.util.multiprocessing import Pool

from edynamics.modelling_tools.embeddings import Embedding
from edynamics.modelling_tools.projectors import Projector
from edynamics.modelling_tools.observers import Observer, lag


def gnn_optimizer(
    embedding: Embedding,
    target: str,
    projector: Projector,
    points: pd.DataFrame,
    observers: [Observer],
    steps: int = 1,
    step_size: int = 1,
    improvement_threshold: float = -np.inf,
    compute_pool: Pool = None,
    verbose: bool = False,
):
    """
    Searches for the set of lags which maximize Pearson's coefficient for a set of predictions and sets the Embedding
    observers to those lags.

    :param embedding: the delay Embedding.
    :param target: the variable to predict
    :param projector: which prediction method to use.
    :param points: a dataframe, indexed by time, of the points from which to predict.
    :param improvement_threshold: the minimum improvement from the previous round to continue optimizing.
    :param observers: the set of observers, excluding the target, to optimize over
    :param steps: the number of steps in a multistep prediction to make where successive predictions are made using
        previous predictions.
    :param step_size: the step size of each prediction as a multiple of the sampling frequency of the data.
    :param compute_pool: a ray computing pool if parallel computing.
    :param verbose: if true print out the result of each optimization round.
    :return: a dataframe summarizing the performance of each iteration. For each iteration the best lag added from the
    previous iteration is removed and appears as a nan in the row.
    """

    logging.info(msg="Performing gnn optimization...")

    # track best skill
    best_skill = 0

    # Initialize the embedding with the t=0 lag (i.e. just the regular time series)
    embedding.observers = [lag(variable_name=target, tau=0)]

    # todo: run first iteration as base line
    for i in range(len(observers)):
        # add a slot for a new lag_
        embedding.observers = embedding.observers + [None]

        # loop over moves_remaining lags
        futures = []
        if compute_pool is not None:
            args = []

            for observer in observers:
                embedding_copy = deepcopy(embedding)
                embedding_copy.observers[-1] = observer
                embedding_copy.compile()

                args.append(
                    [embedding_copy, target, projector, points, steps, step_size]
                )

            futures = compute_pool.starmap(_gnn_observer_parallel_step.remote, args)

        else:
            pbar = tqdm(observers, leave=False)
            for _, observer in enumerate(pbar):
                pbar.set_description(observer.observation_name)
                # add a new observer
                embedding_copy = deepcopy(embedding)
                embedding_copy.observers[-1] = observer
                embedding_copy.compile()

                futures.append(
                    _gnn_observer_step(
                        embedding=embedding_copy,
                        target=target,
                        projector=projector,
                        points=points,
                        steps=steps,
                        step_size=step_size,
                    )
                )

        if compute_pool is not None:
            results = []
            for result in tqdm(futures, leave=False):
                results.append(ray.get(result))
        else:
            results = futures

        # Get best performer
        max_idx = results.index(max(results))
        maximum = (observers[max_idx], results[max_idx])
        improvement = results[max_idx] - best_skill

        # remove the best performing observer from the potential lags and add to the block lags
        observers.remove(maximum[0])
        embedding.observers[-1] = maximum[0]
        best_skill = maximum[1]

        if verbose:
            print(maximum[0].observation_name + " " + str(improvement))

        # check early stopping improvement
        sigs = len(str(improvement_threshold).split(".")[1])
        if round(improvement, sigs) < improvement_threshold:
            embedding_.observers = embedding_.observers[:-1]
            print(
                "Early Stopping:\t"
                + str([obs.observation_name for obs in embedding_.observers])
            )
            break

    logging.info("Gnn lag optimization complete.")
    return embedding_.observers


def _gnn_observer_step(
    embedding: Embedding,
    target: str,
    projector: Projector,
    points: pd.DataFrame,
    steps: int = 1,
    step_size: int = 1,
) -> float:
    # predict
    x = embedding_.get_points(times=points.index)
    y_hat = projector.predict(
        embedding=embedding, points=x, steps=steps, step_size=step_size
    )

    # compute prediction skill
    y = embedding.get_points(times=y_hat.droplevel(level=0).index)
    rho = y_hat[target].droplevel("Current_Time").corr(y[target])

    return rho


@ray.remote
def _gnn_observer_parallel_step(
    embedding: Embedding,
    target: str,
    projector: Projector,
    points: pd.DataFrame,
    steps: int = 1,
    step_size: int = 1,
) -> float:
    return _gnn_observer_step(
        embedding=embedding,
        target=target,
        projector=projector,
        points=points,
        steps=steps,
        step_size=step_size,
    )
