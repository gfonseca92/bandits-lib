from abc import abstractmethod, ABC


class BaseLabeler(ABC):
    @abstractmethod
    def draw(self):
        pass