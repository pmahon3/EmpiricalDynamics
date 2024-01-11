import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from edynamics.modelling_tools.embeddings import Embedding
from edynamics.modelling_tools.kernels import Kernel, Exponential
from edynamics.modelling_tools.norms import Norm, Minkowski
from .projector import Projector


class KNearestNeighbours(Projector):
    """
    --------------------------------------------------------------------------------------------------------------------
    SUMMARY
    --------------------------------------------------------------------------------------------------------------------
    The knn class is a subclass of the Projector class and is used for performing k-nearest neighbor projections in a
    state space embedding. It takes a set of points and predicts their future values based on the k nearest neighbors in
    the embedding.

    --------------------------------------------------------------------------------------------------------------------
    EXAMPLE USAGE
    --------------------------------------------------------------------------------------------------------------------

    # Create an instance of the knn class
        knn_projector = knn(norm=Norm(), kernel=Kernel(), k=3)

    # Perform k-nearest neighbor projections
        predictions = knn_projector.predict(embedding, points, steps=1, step_size=1)

    --------------------------------------------------------------------------------------------------------------------
    MAIN FUNCTIONALITIES
    --------------------------------------------------------------------------------------------------------------------
    Perform k-nearest neighbor projections for a set of points in a state space embedding.
    Use the k nearest neighbors to calculate weights for the projections.
    Update the projected values based on the observed data.

    --------------------------------------------------------------------------------------------------------------------
    METHODS
    --------------------------------------------------------------------------------------------------------------------

    __init__(self, norm: Norm, kernel: Kernel, k: int):

        Initializes the knn projector with a norm, kernel, and the number of nearest neighbors to consider.

    predict(self, embedding: Embedding, points: pd.DataFrame, steps: int, step_size: int) -> pd.DataFrame:

        Performs k-nearest neighbor projections for a set of points in a state space embedding. Returns a DataFrame of
        the projected values.

    --------------------------------------------------------------------------------------------------------------------
    FIELDS
    --------------------------------------------------------------------------------------------------------------------

        norm: Norm:         The norm used for distance calculations.
        kernel: Kernel:     The kernel used for weighting the projections.
        k: int:             The number of nearest neighbors to consider for each point.

    """

    def __init__(self, k: int, norm: Norm = Minkowski(p=2), kernel: Kernel = Exponential(theta=0)):
        super().__init__(norm=norm, kernel=kernel)
        self.k = k

    def project(
            self, embedding: Embedding, points: pd.DataFrame, steps: int, step_size: int
    ) -> pd.DataFrame:
        """
        Perform a k projection for each of the given points.

        :param embedding: the state space Embedding.
        :param points: the points to be projected.
        :param steps: the number of prediction steps to make out from for each point. By default 1.
            period.
        :param step_size: the number to steps, of length given by the frequency of the block, to predict.
        :return: the k projected points
        """
        if self.k is None:
            self.k = embedding.dimension + 1

        indices = self.build_prediction_index(
            frequency=embedding.frequency,
            index=points.index,
            steps=steps,
            step_size=step_size,
        )
        predictions = pd.DataFrame(
            index=indices, columns=embedding.block.columns, dtype=float
        )

        if len(predictions.index.droplevel(0).intersection(embedding.library_times)):
            warnings.warn(f"The following reference time indices are included in the library times:\n"
                          f"\t{embedding.library_times.intersection(predictions.index.droplevel(0))}\n"
                          f"This may result in unexpected behaviour.")

        for i in range(len(points)):
            reference_time = indices[i * steps][0]
            point = points.iloc[i].values
            for j in range(steps):
                try:
                    prediction_time = indices[i * steps + j][-1]
                    knn_idxs = embedding.get_k_nearest_neighbours(
                        point=point, max_time=reference_time, knn=self.k
                    )

                    # optimize: would self.Norm.distance_matrix be more direct here? It might be but using kernel.weigh
                    #  allows for class specific error handling according to the kernel and its functional form
                    weights = self.kernel.weigh(
                        distance_matrix=cdist(
                            point[np.newaxis, :], embedding.block.iloc[knn_idxs].values
                        )
                    )

                    predictions.loc[(reference_time, prediction_time)] = (
                            np.dot(
                                weights, embedding.block.iloc[knn_idxs + step_size].values
                            )
                            / weights.sum()
                    )

                    if steps > 1:
                        predictions = self.update_values(
                            embedding=embedding,
                            predictions=predictions,
                            current_time=reference_time,
                            prediction_time=prediction_time,
                        )
                        point = predictions.loc[(reference_time, prediction_time)].values
                except IndexError:
                    continue

        return predictions
