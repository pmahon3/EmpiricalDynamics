import pandas as pd

from node import node
from edynamics.modelling_tools.embeddings import Embedding
from edynamics.modelling_tools.observers import observer, lag
from edynamics.modelling_tools.projectors import projector


class mcst:
    def __init__(self,
                 target: str,
                 observers):
        self.target = target
        self.head = node(observer_=lag(variable_name=target, tau=0))
        self.observers = observers
        self.candidate_nodes: [node] = None
        self.current = self.head

    def compile(self,
                embedding: Embedding,
                projector_: projector,
                points: pd.DataFrame,
                steps: int = 1,
                step_size: int = 1,
                verbose: bool = False):
        pass

    def selection_step(self):
        current_path = self.current.get_parents([])
