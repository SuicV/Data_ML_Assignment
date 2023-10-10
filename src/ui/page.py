from abc import ABC, abstractmethod

class Page(ABC):
    def __init__(self, name, title):
        self.name = name
        self.title = title

    @abstractmethod
    def display(self):
        pass