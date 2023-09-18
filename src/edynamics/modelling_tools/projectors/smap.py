from .projector import Projector
import pandas as pd
import numpy as np

from scipy.linalg import pinv

from edynamics.modelling_tools.embeddings import Embedding
from edynamics.modelling_tools.norms import Norm
from edynamics.modelling_tools.kernels import Kernel
from edynamics.modelling_tools.norms import minkowski
from edynamics.modelling_tools.kernels import exponential


class smap(Projector):
    def __init__(self,
                 norm: Norm = minkowski(p=2),
                 kernel: Kernel = exponential(theta=0.0)):

        super().__init__(norm=norm, kernel=kernel)

    def predict(self, embedding: Embedding, points: pd.DataFrame, steps: int, step_size: int) -> pd.DataFrame:
        """
        Perform an S-Map projection from the given point.

        :param embedding: the delay embedded system.
        :param points: an n-by-m pandas dataframe of m-dimensional lagged coordinate vectors, stored row-wise, to be
            projected according to the library of points.
        :param steps: the number of prediction steps to make out from for each point. By default 1.
        :param step_size: the number to steps, of length given by the frequency of the block, to prediction.
        :return pd.DataFrame: a dataframe of the predicted embedded points. The data block is multi-indexed; the first
            index level is the 'current_time', the maximum time of observed data that was used to make the prediction,
            the second index level is the 'prediction_time', the time of the predicted point.
        """

        indices = self.build_prediction_index(frequency=embedding.frequency,
                                              index=points.index,
                                              steps=steps,
                                              step_size=step_size)

        # Run the predictions
        futures = []
        for i, point in enumerate(points.values):
            futures.append(self._smap_step(embedding=embedding,
                                           point=point,
                                           indices=indices[i * steps:i * steps + steps],
                                           steps=steps,
                                           step_size=step_size))

        # Retrieve results
        projections = pd.DataFrame(index=indices, columns=embedding.block.columns, dtype=float)
        for result in futures:
            projections.loc[result.index] = result.values

        return projections

    def _smap_step(self,
                   embedding: Embedding,
                   point: np.array,
                   indices: pd.MultiIndex,
                   steps: int,
                   step_size: int) -> pd.DataFrame:
        """
        Makes a smap projection for a given prediction period, modifying the output dataframe in place. Useful for
        parallelization.

        :param np.array point: the starting point for the prediction period.
        :param indices pd.MultiIndex: the indices for projected points.
        :param int steps: the number predictions to conduct for this prediction period
        :param int step_size: the number to steps, of length given by the frequency of the block, to prediction.
        :return pd.DataFrame: a dataframe of the predicted embedded points. The data block is multi-indexed; the first
            index level.
            is the 'current_time', the maximum time of observed data that was used to make the prediction, the second
            index level is the 'prediction_time', the time of the predicted point.
        """
        predictions = pd.DataFrame(index=indices,
                                   columns=embedding.block.columns,
                                   dtype=float)

        current_time = indices[0][0]
        prediction_time = indices[0][-1]

        # X is the library of inputs, the embedded points up to the starting point of the prediction period
        # y is the library of outputs, the Embedding points at time t+1
        X = embedding.block.loc[embedding.library_times][:-step_size]
        y = embedding.block.loc[embedding.library_times][step_size:]

        for j in range(steps):
            # Compute the weights
            distance_matrix = self.norm.distance_matrix(embedding=embedding,
                                                        points=point[np.newaxis, :],
                                                        max_time=current_time)
            weights = self.weigher.weigh(distance_matrix=distance_matrix)[:-step_size]
            weights[np.isnan(weights)] = 0.0

            # A is the product of the weights and the library X points, A = w * X
            A = weights * X.values
            # B is the product of the weights and the library y points, A = w * y
            B = weights * y.values
            # Solve for C in B=AC via SVD
            C = np.linalg.lstsq(A, B, rcond=None)[0]

            predictions.loc[(current_time, prediction_time)] = np.matmul(point, C)

            # replace predictions for lagged variables for either actual values or previous predicted values
            predictions = self.update_values(embedding=embedding,
                                             predictions=predictions,
                                             current_time=current_time,
                                             prediction_time=prediction_time)

            # If there are still more steps to go for this prediction period update variables
            if j < steps - 1:
                # Update current point predicting from
                point = predictions.loc[(current_time, prediction_time)].values

                prediction_time = indices[j + 1][-1]

        return predictions
