from edynamics.modelling_tools.observers.observer import observer


class observer_node:
    def __init__(self, data: observer, children: [observer] = None):
        self.data = data

        if children is None:
            children = []
        self.children = children

    def is_leaf(self):
        if len(self.children) == 0:
            return True
        else:
            return False

    def __hash__(self):
        return hash(self.data.__hash__())

    def __eq__(self, other):
        return self.data.__eq__(other)
