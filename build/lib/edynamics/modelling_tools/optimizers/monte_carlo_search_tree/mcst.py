from edynamics.modelling_tools.observers.observers import observer

from typing import Type
import numpy as np


class mcst:
    """
    Literature:
        -https://arxiv.org/pdf/2010.11523.pdf - Combinatorial Monte Carlo Search Trees
        -https://link.springer.com/article/10.1007/s11071-022-07280-2 - Monte Carlo Search Tree for SSR selection
    """
    def __init__(self, f: observer, children: [observer], moves_remaining: int, parent: Type['mcst'] = None):
        self.observer = f
        self.skill = np.nan
        self.moves_remaining = moves_remaining
        self.parent = parent
        self.children = [mcst(f=child,
                              children=[new_child for new_child in children if new_child != child],
                              parent=self,
                              moves_remaining=self.moves_remaining - 1)
                         for child in children]
        self._softmax_beta = 1.0
        self._visits = 0

    def expansion(self):
        if not self.is_terminal():
            for child in self.children:
                child.set_skill(child.compute_skill())
        else:
            pass

    def selection(self):
        """
        selection selects a child node based on child skill, Gamma, with probability of selection for each child, i,
        given by softmax(Gamma_i)
        :return: mcst - subtree where the selected child node is root.
        """
        child_skills = np.array([child.skill for child in self.children])
        softmax_probs = np.exp(-1.0 * self._softmax_beta * child_skills) / \
                        np.sum(np.exp(-1.0 * self._softmax_beta * child_skills))

        return np.random.choice(self.children, 1, p=softmax_probs)

    def simulation(self):
        pass

    def update(self):
        pass

    def compute_skill(self) -> float:
        pass

    def set_skill(self, skill: float) -> None:
        self.skill = skill

    def visit_node(self):
        self._visits += 1

    def get_visits(self):
        return self._visits

    def is_terminal(self):
        if self.moves_remaining == 0:
            return True
        return False

    def __str__(self, levels=0):
        return f"mcst(\n\tf={str(self.observer)},\n\tskill={self.skill},\n\tmoves_remaining={self.moves_remaining}\n)"
