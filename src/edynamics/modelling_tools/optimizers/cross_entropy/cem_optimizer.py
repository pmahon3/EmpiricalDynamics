import numpy as np
import pandas as pd

from edynamics.modelling_tools.embeddings import Embedding
from edynamics.modelling_tools.projectors.projector import projector
from edynamics.modelling_tools.observers import observer

import numpy as np


class CEMOptimizer:
    def __init__(self,
                 projector_: projector,
                 embedding: Embedding,
                 points: pd.DataFrame,
                 steps: int,
                 step_size: int,
                 observers: [observer],
                 pop_size=50,
                 elite_frac=0.2,
                 num_iter=100,
                 mu_init: np.array = None,
                 sigma_init: np.array = None):

        # Optimization function params
        self.embedding = embedding
        self.projector = projector_
        self.points = points
        self.steps = steps
        self.step_size = step_size
        self.observers = observers

        # Optimizer params
        self.pop_size = pop_size
        self.elite_frac = elite_frac
        self.num_iter = num_iter
        self.mu_init = mu_init
        self.sigma_init = sigma_init

    def optimize(self):
        # Initialize mean and standard deviation
        mu = self.mu_init or np.zeros(self.embedding.dimension)
        sigma = self.sigma_init or np.ones(self.embedding.dimension)

        for i in range(self.num_iter):
            # Generate population of candidate solutions
            pop = np.random.normal(mu, sigma, size=(self.pop_size, self.embedding.dimension))

            # Evaluate performance of candidate solutions
            fitness = np.array([])

            # Select elite solutions
            elite_idx = fitness.argsort()[:int(self.pop_size * self.elite_frac)]
            elite_pop = pop[elite_idx]

            # Update distribution parameters
            mu = elite_pop.mean(axis=0)
            sigma = elite_pop.std(axis=0)

        # Return best solution found
        best_idx = fitness.argmin()
        best_sol = pop[best_idx]
        return best_sol, fitness[best_idx]
