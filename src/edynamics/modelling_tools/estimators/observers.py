import logging
from copy import deepcopy

import numpy as np
import pandas as pd
import ray
from ray.util.multiprocessing import Pool
from tqdm import tqdm

from edynamics.modelling_tools.embeddings import Embedding
from edynamics.modelling_tools.observers import Observer, Lag
from edynamics.modelling_tools.projectors import Projector, KNearestNeighbours


def greedy_nearest_neighbour(
        embedding: Embedding,
        target: str,
        projector: Projector,
        times: pd.DatetimeIndex,
        observers: [Observer],
        steps: int = 1,
        step_size: int = 1,
        improvement_threshold: float = -np.inf,
        compute_pool: Pool = None,
        verbose: bool = False,
):
    """
    Searches for the set of lags which maximize Pearson's coefficient for a set of predictions and sets the Embedding
    observers to those lags. Uses a greedy nearest neighbour approach by successively computing and adding the observer
    which improves prediction skill the most.

    :param embedding: the delay Embedding.
    :param target: the variable to predict
    :param projector: which prediction method to use.
    :param pd.DatetimeIndex times: the times to predict from.
    :param improvement_threshold: the minimum improvement from the previous round to continue optimizing.
    :param observers: the set of observers to optimize over
    :param steps: the number of steps in a multistep prediction to make where successive predictions are made using
        previous predictions.
    :param step_size: the step size of each prediction as a multiple of the sampling frequency of the DATA.
    :param compute_pool: a ray computing pool if parallel computing.
    :param verbose: if true print out the result of each optimization round.
    :return: a list of the selected observers added at each round and a list of the corresponding model performance
    measured by Pearson's correlation coefficient
    """

    logging.info(msg="STARTING gnn optimization...")

    # track best skill
    performance = []

    # Initialize the embedding with the t=0 lag (i.e. just the regular time series)
    embedding.set_observers(observers=[Lag(variable_name=target, tau=0)], compile_block=True)

    # Comput baseline performance
    performance.append(_gnn_observer_step(
        embedding=embedding,
        target=target,
        projector=projector,
        points=embedding.block.loc[times],
        steps=steps,
        step_size=step_size,
    ))
    best_skill = performance[0]

    if verbose:
        logging.info(f'\nObserver:\t\t{str(list(map(str, embedding.observers)))}\n'
                     f'Performance (rho):\t{performance[0]}')

    for i in range(len(observers)):
        # add a slot for a new lag_
        embedding.observers = embedding.observers + [None]

        # loop over moves_remaining lags
        futures = []
        if compute_pool is None:
            pbar = tqdm(observers, leave=False)
            for _, observer in enumerate(pbar):
                pbar.set_description(observer.observation_name)

                # add a new observer
                embedding.observers[-1] = observer
                embedding.compile()
                points = embedding.block.loc[times]

                futures.append(
                    _gnn_observer_step(
                        embedding=embedding,
                        target=target,
                        projector=projector,
                        points=points,
                        steps=steps,
                        step_size=step_size,
                    )
                )


        else:
            args = []
            for observer in observers:
                embedding_copy = deepcopy(embedding)
                embedding_copy.observers[-1] = observer
                embedding_copy.compile()
                points = embedding_copy.block.loc[times]

                args.append(
                    (embedding_copy, target, projector, points, steps, step_size)
                )

            futures = compute_pool.starmap(_gnn_observer_parallel_step.remote, args)

        if compute_pool is None:
            results = futures
        else:
            results = []
            for result in tqdm(futures, leave=False):
                results.append(ray.get(result))

        # Get best performer
        max_idx = results.index(max(results))
        maximum = (observers[max_idx], results[max_idx])
        improvement = results[max_idx] - best_skill

        # remove the best performing observer from the potential lags and add to the block lags
        observers.remove(maximum[0])
        embedding.observers[-1] = maximum[0]
        best_skill = maximum[1]
        performance.append(best_skill)

        if verbose:
            logging.info(f'\nObservers:\t\t{str(list(map(str, embedding.observers)))}\n'
                         f'Improvement (rho):\t{str(improvement)}')

        # check early stopping improvement
        if improvement < improvement_threshold:
            logging.info(
                f'\nEarly Stopping:\t\t{str([obs.observation_name for obs in embedding.observers])}\n'
                f'Performance (rho):\t{performance}')
            break

    embedding.set_observers(embedding.observers[:-1], compile_block=True)
    logging.info("Gnn lag optimization COMPLETE.")
    return  performance[:-1]


def _gnn_observer_step(
        embedding: Embedding,
        target: str,
        projector: Projector,
        points: pd.DataFrame,
        steps: int = 1,
        step_size: int = 1,
) -> float:
    # predict
    x = points
    y_hat = projector.project(embedding=embedding, points=x, steps=steps, step_size=step_size, leave_out=True)

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
