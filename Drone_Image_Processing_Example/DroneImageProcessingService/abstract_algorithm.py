from abc import ABC, abstractmethod


class Algorithm(ABC):
    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def deinitialize(self):
        pass

    @abstractmethod
    def process(self, input_data):
        pass

    