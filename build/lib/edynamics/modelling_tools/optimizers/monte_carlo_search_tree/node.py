from abc import ABC, abstractmethod


class node(ABC):

    def __init__(self, content):
        self.content = content

    @property
    def content(self):
        return self._content

    @content.setter
    def content(self, value):
        self._content = value

    @abstractmethod
    def is_leaf(self):
        pass

    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass
