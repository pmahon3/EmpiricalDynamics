from edynamics.modelling_tools.observers.observer import observer


class observer_node:
    def __init__(self, content: observer, children: [observer] = None):
        self.content = content

        if children is None:
            children = []
        self.children = children

    def is_leaf(self):
        if len(self.children) == 0:
            return True
        else:
            return False

    def __hash__(self):
        return hash(self.content)

    def __eq__(self, other):
        return self.content.__eq__(other)
