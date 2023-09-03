import ray
import pandas as pd

from ray.util.multiprocessing import Pool

from edynamics.modelling_tools.embeddings import embedding
from edynamics.modelling_tools.projectors import projector
from edynamics.modelling_tools.observers import lag

from copy import deepcopy
from tqdm import tqdm
from ray.util.multiprocessing import Pool


def dimensionality(embedding_: embedding,
                   projector_: projector,
                   target: str,
                   points: pd.DataFrame,
                   dimensions: int = 10,
                   steps: int = 1,
                   step_size: int = 1,
                   compute_pool: Pool = None) -> pd.DataFrame:
    """
    Successively adds lags from 0, -1, -2, up to -n where n is given by 'dimensions', evaluating the prediction
    skill.

    :param Embedding embedding: the delay embedded block.
    :param projector projector_: which prediction method to use.
    :param str target: which embedding variable to predict.
    :param pd.DataFrame points: the points for which to evaluate the prediction skill on.
    :param int dimensions: the maximum number of lags to add to the model.
    :param int steps: the number of steps in a multistep prediction to make where successive predictions are made
        using previous predictions.
    :param int step_size: the step size of each prediction as a multiple of the sampling frequency of the data.
    :param compute_pool: a ray computing pool if parallel computing.
    :return: a dataframe indexed by the number of dimensions and corresponding prediction skill, as measured by
        Pearson's correlation coefficient.
    """
    # Save original observation functions
    original_observers = embedding_.observers

    # initialize dataframe to return
    rhos = pd.DataFrame(data=[None for _ in range(dimensions)],
                        index=[i + 1 for i in range(dimensions)],
                        columns=['rho'])

    # create pool of lags to draw on, one for each dimension, lag multiples of -1
    lags = [lag(variable_name=target, tau=-i) for i in range(0, dimensions)]

    # Run predictions for each dimension
    futures = []
    if compute_pool is not None:
        args = []
        for i in range(dimensions):
            embedding_copy = deepcopy(embedding_)
            embedding_copy.observers = lags[:i + 1]
            embedding_copy.compile()
            points_ = embedding_copy.get_points(times=points.index)

            args.append([embedding_copy,
                         projector_,
                         target,
                         points_,
                         steps,
                         step_size])

        futures = compute_pool.starmap(dimensionality_parallel_step.remote, args)

    else:
        pbar = tqdm(range(dimensions), leave=True)
        for i in pbar:
            pbar.set_description('E = ' + str(i + 1))
            embedding_copy = deepcopy(embedding_)
            embedding_copy.observers = lags[:i + 1]
            embedding_copy.compile()
            points_ = embedding_copy.get_points(times=points.index)

            futures.append(dimensionality_step(embedding_=embedding_copy,
                                               projector_=projector_,
                                               target=target,
                                               points=points_,
                                               steps=steps,
                                               step_size=step_size))

    if compute_pool is not None:
        results = []
        for result in tqdm(futures):
            results.append(ray.get(result))

    else:
        results = futures

    for i, result in enumerate(results):
        rhos.loc[i + 1] = result

    # restore original observers
    Embedding.observers = original_observers

    return rhos


def dimensionality_step(embedding_: embedding,
                        projector_: projector,
                        target: str,
                        points: pd.DataFrame,
                        steps: int,
                        step_size: int) -> float:
    y_hat = projector_.predict(embedding_=embedding_,
                               points=points,
                               steps=steps,
                               step_size=step_size)

    y = embedding.data.loc[y_hat.index.droplevel(level=0)][target]

    return y_hat.droplevel(level=0)[target].corr(y)


@ray.remote
def dimensionality_parallel_step(embedding_: embedding,
                                 projector_: projector,
                                 target: str,
                                 points: pd.DataFrame,
                                 steps: int,
                                 step_size: int) -> float:
    return dimensionality_step(embedding_=embedding_,
                               projector_=projector_,
                               target=target,
                               points=points,
                               steps=steps,
                               step_size=step_size)
