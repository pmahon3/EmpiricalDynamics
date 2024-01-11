import random
import pandas as pd
import numpy as np

from scipy.spatial.distance import cdist
from tqdm import tqdm
from datetime import datetime

from edynamics.modelling_tools.embeddings import Embedding
from edynamics.modelling_tools.kernels import exponential
from edynamics.modelling_tools.kernels import Kernel


def convergent_cross_mapping(
    embedding_y: Embedding,
    embedding_x: Embedding,
    target: str,
    library_times: [pd.Timestamp],
    prediction_times: [pd.Timestamp],
    n_partitions: int,
    kernel: Kernel = exponential(theta=1),
) -> pd.DataFrame:
    """Cross maps Embedding y to Embedding x, providing the cross map curve to determine whether x is a convergent
    cross map cause of y.

    :param embedding_x: a coordinate delay Embedding of variable x.
    :param embedding_y: a coordinate delay Embedding of variable y.
    :param target: the target variable in Embedding x.
    :param library_times: the set of points to draw random library samples from.
    :param prediction_times: the times to test cross mapping on.
    :param n_partitions: the number of random partitions ranging from size N/n_partitions to N where N is the length of
    the prediction set defined by prediction start and prediction end.
    :param kernel: the weighting kernel for the cross map estimate of the target.
    :return: a pandas dataframe of the convergent cross map profile of y cross mapped to x. Indexed by library size.
    """

    # Set libraries and compile
    # initialize partition sizes
    partitions = np.linspace(
        len(prediction_times) / n_partitions,
        len(library_times),
        n_partitions,
        dtype=int,
    )

    rhos = pd.DataFrame(index=range(1, n_partitions + 1), columns=["rho"])

    pbar = tqdm(enumerate(partitions), leave=True, total=len(partitions))
    for i, partition in pbar:
        pbar.set_description("Partition Size = " + str(partitions[i]))

        lib_times = random.sample(list(library_times), k=partitions[i])

        embedding_x.set_library(library_times=lib_times)
        embedding_x.compile()

        embedding_y.set_library(library_times=lib_times)
        embedding_y.compile()

        rhos.loc[i + 1] = _cross_map_step(
            embedding_y=embedding_y,
            embedding_x=embedding_x,
            target=target,
            indices=prediction_times,
            kernel=kernel,
        )

    return rhos


def _cross_map_step(
    embedding_y: Embedding,
    embedding_x: Embedding,
    target: str,
    indices: pd.DatetimeIndex,
    kernel: Kernel,
) -> float:
    """Perform a cross mapping run for a given set of times.

    :param embedding_x: a coordinate delay Embedding of variable x.
    :param embedding_y: a coordinate delay Embedding of variable y.
    :param target: the target variable in Embedding x.
    :param kernel: the weighting kernel for the cross map estimate of the target.
    """
    cross_mapped_points = pd.DataFrame(
        index=indices, columns=embedding_x.block.columns, dtype=float
    )

    # Perform the cross mapping
    for time in indices:
        knn_idxs = embedding_y.get_k_nearest_neighbours(
            embedding_y.get_points(time).values[0],
            knn=embedding_y.dimension,
            max_time=max(embedding_y.library_times),
        )
        knn_times = embedding_y.block.index[knn_idxs]

        # todo: generalize Norm.distance function to handle this use case
        point = embedding_y.get_points(time).values
        weights = kernel.weigh(cdist(point, embedding_y.block.loc[knn_times].values)[0])

        cross_mapped_points.loc[time] = (
            np.matmul(weights, embedding_x.block.loc[knn_times]) / weights.sum()
        ).values

    x = embedding_x.get_points(indices)[target]
    x_hat = cross_mapped_points[target]
    return x_hat.corr(x)
