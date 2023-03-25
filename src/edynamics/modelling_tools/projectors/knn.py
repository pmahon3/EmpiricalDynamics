from .projector import projector

import numpy as np
import pandas as pd

from edynamics.modelling_tools.embeddings import Embedding
from edynamics.modelling_tools.norms import norm
from edynamics.modelling_tools.weighers import weigher

from scipy.spatial.distance import cdist


class knn(projector):
    def __init__(self,
                 norm_: norm,
                 weigher_: weigher,
                 k: int):

        super().__init__(norm_=norm_,
                         weigher_=weigher_)
        self.k = k

    def predict(self,
                embedding: Embedding,
                points: pd.DataFrame,
                steps: int = 1,
                step_size: int = 1) -> pd.DataFrame:
        """
        Perform a k projection for each of the given points.

        :param embedding: the state space embedding.
        :param points: the points to be projected.
        :param steps: the number of prediction steps to make out from for each point. By default 1.
            period.
        :param step_size: the number to steps, of length given by the frequency of the block, to predict.
        :return: the k projected points
        """
        if self.k is None:
            self.k = embedding.dimension

        indices = self.build_prediction_index(frequency=embedding.frequency,
                                              index=points.index,
                                              steps=steps,
                                              step_size=step_size)
        predictions = pd.DataFrame(index=indices,
                                   columns=embedding.block.columns,
                                   dtype=float)

        for i in range(len(points)):
            current_time = indices[i * steps][0]
            point = points.iloc[i].values
            for j in range(steps):
                try:
                    prediction_time = indices[i * steps + j][-1]
                    knn_idxs = embedding.get_k_nearest_neighbours(
                        point=point,
                        max_time=current_time,
                        knn=self.k)

                    # todo can self.norm.distance_matrix be modified for this use case?
                    weights = self.weigher.weigh(distance_matrix=cdist(point[np.newaxis, :],
                                                                       embedding.block.iloc[knn_idxs].values))

                    predictions.loc[(current_time, prediction_time)] = \
                        np.matmul(weights, embedding.block.iloc[knn_idxs + step_size].values) / weights.sum()

                    if steps > 1:
                        predictions = self.update_values(embedding=embedding,
                                                         predictions=predictions,
                                                         current_time=current_time,
                                                         prediction_time=prediction_time)
                        point = predictions.loc[(current_time, prediction_time)].values
                except IndexError:
                    continue

        return predictions
