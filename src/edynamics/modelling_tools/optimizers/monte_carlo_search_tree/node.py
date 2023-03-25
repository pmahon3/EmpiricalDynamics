from __future__ import annotations

from edynamics.modelling_tools.observers import observer, lag

import numpy as np


class node:
    def __init__(self, observer_: observer = None, parent: node = None, children: [node] = None, V: float = np.nan, n: int = 0):
        # observation function
        self.observer = observer_
        # parent node
        self.parent = parent

        # children
        if children is None:
            self.children = []
        if children is not None:
            self.children = children
            for child in children:
                child.parent = self

        #
        if parent is not None and self not in parent.children:
            parent.children.append(self)

        # average value of the subtree rooted at this node
        self.V = V
        # number of times this node has been visited
        self.n = n
        # upper confidence bound:

    def get_level(self, level: int = 0):
        if self.parent is None:
            return 0
        else:
            return level + self.parent.get_level(level + 1)

    def add_child(self, new_child: node):
        if new_child not in self.children:
            self.children.append(new_child)
        new_child.parent = self

    def get_parents(self):
        if self.parent is None:
            return []
        else:
            parents = [self.parent]
            result = self.parent.get_parents()
            for parent in result:
                parents.append(parent)
            return parents

    def compute_value(self):
        pass

    def __str__(self):
        return 'Node: ' + self.observer.__str__()

if __name__ == '__main__':
    a = node()
    b = node(parent=a)
    c = node(parent=b)
    c.get_parents()
