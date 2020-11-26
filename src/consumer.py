from ABC import abstractmethod
from abc import ABC

from anomalyDetection import AnomalyDetectionAbstract

class ConsumerAbstract(ABC):
    anomaly: "AnomalyDetectionAbstract"
    memory_size: int

    def __init__(self, memory_size: int = 0) -> None:
        self.memory_size = memory_size

    @abstractmethod
    def _stopping_condition_(self) -> bool:
        pass

    @abstractmethod
    def _read_next_(self) -> None:
        pass

    @abstractmethod
    def configure(self) -> None:
        pass

    def read(self) -> None:
        while(self._stopping_condition_):
            self._read_next_()
            self.anomaly.check()

class ConsumerKafka(ConsumerAbstract):

    def __init__(self) -> None:
        super().__init__()

class ConsumerJSON(ConsumerAbstract):

    def __init__(self) -> None:
        super().__init__()
