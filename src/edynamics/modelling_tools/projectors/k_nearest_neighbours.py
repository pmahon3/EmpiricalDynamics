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

    def __init__(self, k: int = None, norm: Norm = Minkowski(p=2), kernel: Kernel = Exponential(theta=0)):
        super().__init__(norm=norm, kernel=kernel)
        self.k = k

    def project(
            self, embedding: Embedding, points: pd.DataFrame, steps: int, step_size: int, leave_out: bool = False
    ) -> pd.DataFrame:
        """
        Perform a k projection for each of the given points.

        :param embedding: the state space Embedding.
        :param points: the points to be projected.
        :param steps: the number of prediction steps to make out from for each point. By default 1.
            period.
        :param step_size: the number to steps, of length given by the frequency of the block, to predict.
        :param leave_out: if true return the matrices of coefficients used to integrate each prediction.
        :return: the k projected points
        """

        # k can be set as none for certain optimization schemes and set adaptively according the optimization schemes
        #   manipulation of the embedding. In this case it is typically desired that k be set to:
        #   k=embedding.dimension + 1, and then returned to none for the next iteration.
        revert = False
        if self.k is None:
            self.k = embedding.dimension + 1
            revert = True

        indices = self.build_prediction_index(
            frequency=embedding.frequency,
            index=points.index,
            steps=steps,
            step_size=step_size,
        )
        predictions = pd.DataFrame(
            index=indices, columns=embedding.block.columns, dtype=float
        )

        for i in range(len(points)):
            reference_time = indices[i * steps][0]
            point = points.iloc[i].values
            for j in range(steps):
                prediction_time = indices[i * steps + j][-1]

                # if the current point is in the library and performing a leave one out prediction then exclude the
                #   point at the reference time from the k nearest neighbours, otherwise just get the k nearest
                #   neighbours
                #
                # check if the current point is in the embedding
                try:
                    embedding.block.loc[reference_time, :]

                    if leave_out:
                        knn_count = [2, self.k + 1]

                    else:
                        knn_count = self.k

                    knn_idxs = embedding.get_k_nearest_neighbours(point=point, knn=knn_count)
                # if the current point isn't in the embedding just get the k nearest neighbours
                except KeyError:
                    knn_idxs = embedding.get_k_nearest_neighbours(point=point, knn=self.k)

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

                predictions = self.update_values(
                    embedding=embedding,
                    predictions=predictions,
                    current_time=reference_time,
                    prediction_time=prediction_time,
                )

                if steps > 1:
                    point = predictions.loc[(reference_time, prediction_time)].values

        # see note at beginning of this function
        if revert:
            self.k = None

        return predictions

    def __repr__(self):
        return f"K-Nearest Neighbours Projector:\n" \
               f"\tNorm:\t{repr(self.norm)}\n" \
               f"\tKernel:\t{repr(self.kernel)}\n" \
               f"\tK:\t{self.k}"

    def __str__(self):
        return f"K-Nearest Neighbours Projector:\n" \
               f"\tNorm:\t{str(self.norm)}\n" \
               f"\tKernel:\t{str(self.kernel)}\n" \
               f"\tK:\t{self.k}"

    def __eq__(self, other):
        if isinstance(other, KNearestNeighbours):
            return self.k == other.k and super.__eq__(self, other)
        return False
