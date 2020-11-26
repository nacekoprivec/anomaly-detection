from ABC import abstractmethod
from abc import ABC
from typing import NoReturn

from anomalyDetection import AnomalyDetectionAbstract

class ConsumerAbstract(ABC):
    anomaly: "AnomalyDetectionAbstract"

    def __init__(self) -> None:
        pass

    @abstractmethod
    def _stopping_condition_(self) -> bool:
        pass

    @abstractmethod
    def _read_next_(self) -> None:
        pass

    def read(self) -> None:
        while(self._stopping_condition_):
            self._read_next_()
            self.anomaly.check()

class ConsumerKafka(ConsumerAbstract):

    def __init__(self) -> None:
        super().__init__()