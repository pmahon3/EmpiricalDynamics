import numpy as np
import pandas as pd

from edynamics.modelling_tools.embeddings import Embedding
from edynamics.modelling_tools.kernels import Kernel
from edynamics.modelling_tools.kernels import Exponential
from edynamics.modelling_tools.norms import Norm
from edynamics.modelling_tools.norms import Minkowski
from .projector import Projector


class WeightedLeastSquares(Projector):
    def __init__(
            self, norm: Norm = Minkowski(p=2), kernel: Kernel = Exponential(theta=0.0)
    ):
        super().__init__(norm=norm, kernel=kernel)

    def project(
            self,
            embedding: Embedding,
            points: pd.DataFrame,
            steps: int,
            step_size: int,
            leave_out: bool = True
    ) -> pd.DataFrame:
        """
        Performs a single or multistep weighted lease squares projections from each of the given points.

        :param embedding: the delay embedded system.
        :param points: an n-by-m pandas dataframe of m-dimensional lagged coordinate vectors, stored row-wise, to be
            projected according to the library of points.
        :param steps: the number of prediction steps to make out from for each point. By default 1.
        :param step_size: the number to steps, of length given by the frequency of the block, to prediction.
        :param leave_out: if true return the matrices of coefficients used to integrate each prediction.
        :return pd.DataFrame: a dataframe of the predicted embedded points. The DATA block is multi-indexed; the first
            index level is the 'current_time', the maximum time of observed DATA that was used to make the prediction,
            the second index level is the 'prediction_time', the time of the predicted point.
        """

        indices = self.build_prediction_index(
            frequency=embedding.frequency,
            index=points.index,
            steps=steps,
            step_size=step_size,
        )

        # Run the predictions
        futures = []
        for i, point in enumerate(points.values):
            futures.append(
                self._wls_multi_step(embedding=embedding, point=point, indices=indices[i * steps: i * steps + steps],
                                     steps=steps, step_size=step_size, leave_out=leave_out)
            )

        # Retrieve results
        projections = pd.DataFrame(
            index=indices, columns=embedding.block.columns, dtype=float
        )
        for result in futures:
            projections.loc[result.index] = result.values

        return projections

    def _wls_multi_step(
            self,
            embedding: Embedding,
            point: np.array,
            indices: pd.MultiIndex,
            steps: int,
            step_size: int,
            leave_out: bool = True
    ) -> pd.DataFrame:
        """
        Performs a single step weighted least squares projection from the given point, modifying the output dataframe in
        place. Useful for parallelization.

        :param np.array point: the starting point for the prediction period.
        :param indices pd.MultiIndex: the indices for projected points.
        :param int steps: the number predictions to conduct for this prediction period
        :param int step_size: the number to steps, of length given by the frequency of the block, to prediction.
        :return pd.DataFrame: a dataframe of the predicted embedded points. The DATA block is multi-indexed; the first
            index level.
            is the 'current_time', the maximum time of observed DATA that was used to make the prediction, the second
            index level is the 'prediction_time', the time of the predicted point.
        """
        predictions = pd.DataFrame(
            index=indices, columns=embedding.block.columns, dtype=float
        )

        current_time = indices[0][0]
        prediction_time = indices[0][-1]

        # X is the library of inputs, the embedded points up to the starting point of the prediction period. In addition
        # if the time indices of the points to predict from are in the library, they can be excluded from the library
        # as well.

        mask = embedding.library_times <= current_time

        if leave_out:
            mask = ~embedding.library_times.isin(indices.droplevel(0)) & mask

        # y is the library of outputs, the Embedding points at time t + step_size
        X = embedding.block.loc[mask][:-step_size]
        y = embedding.block.loc[mask][step_size:]

        for j in range(steps):
            # Compute the weights
            distance_matrix = self.norm.distance_matrix(
                embedding=embedding, points=point[np.newaxis, :], times=embedding.library_times[mask]
            )[:-step_size]
            weights = self.kernel.weigh(distance_matrix=distance_matrix)

            # A is the product of the weights and the library X points, A = w * X
            A = weights * X.values
            # B is the product of the weights and the library y points, A = w * y
            B = weights * y.values
            # Solve for C in B=AC via SVD
            C = np.linalg.lstsq(A, B, rcond=None)[0]

            predictions.loc[(current_time, prediction_time)] = np.matmul(point, C)

            # replace predictions for lagged variables for either actual values or previous predicted values
            predictions = self.update_values(
                embedding=embedding,
                predictions=predictions,
                current_time=current_time,
                prediction_time=prediction_time,
            )

            # If there are still more steps to go for this prediction period update variables
            if j < steps - 1:
                # Update current point predicting from
                point = predictions.loc[(current_time, prediction_time)].values

                prediction_time = indices[j + 1][-1]

        return predictions

    def __str__(self):
        return f"Weighted Least Squares Projector:\n" \
               f"\tNorm:\t{str(self.norm)}\n" \
               f"\tKernel:\t{str(self.kernel)}"

    def __repr__(self):
        return f"Weighted Least Squares Projector:\n" \
               f"\tNorm:\t{repr(self.norm)}\n" \
               f"\tKernel:\t{repr(self.kernel)}"
