from ABC import abstractclassmethod, abstractmethod
from abc import ABC
from typing import Any, List

from output import OutputAbstract


class AnomalyDetectionAbstract(ABC):
    memory_size: int
    memory: List[Any]
    outputs: List["OutputAbstract"]

    def __init__(self, memory_size: int = 5,
                 outputs: List["OutputAbstract"] = []) -> None:
        self.memory_size = memory_size
        self.memory = []
        self.outputs = outputs

    @abstractmethod
    def check(self) -> None:
        pass

    @abstractmethod
    def configure(self) -> None:
        pass


class BorderCheck(AnomalyDetectionAbstract):
    UL: float
    LL: float
    value_index: int
    warning_stages: List[float]

    def __init__(self, memory_size: int = 5,
                 outputs: List["OutputAbstract"] = [], UL: float = 0,
                 LL: float = 0, value_index: int = 0,
                 warning_stages: List[float] = [0.9]) -> None:
        super().__init__(memory_size)
        self.LL = LL
        self.UL = UL
        self.value_index = value_index
        self.warning_stages = warning_stages
        self.warning_stages.sort()

    def check(self, new: List[Any]) -> None:
        # inserts new element and deletes old
        self.memory.insert(0, new)
        self.memory = self.memory[:self.memory_size]

        value = new[self.value_index]

        value_normalized = 2*(value - (self.UL + self.LL)/2) / \
            (self.UL - self.LL)
        status = "OK"

        for stage in range(len(self.warning_stages)):
            if(value_normalized > self.warning_stages[stage]):
                status = "Warning" + stage + \
                    ": measurement close to upper limit."
            elif(value_normalized < -self.warning_stages[stage]):
                status = "Warning" + stage + \
                    ": measurement close to lower limit."

        for output in self.outputs:
            output.send_out(status=status, value=value)
