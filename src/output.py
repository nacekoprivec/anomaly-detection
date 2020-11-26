from ABC import abstractmethod
from abc import ABC
from typing import Any


class OutputAbstract(ABC):

    def __init__(self) -> None:
        pass

    @abstractmethod
    def send_out(self, value: Any, status: str) -> None:
        pass


class terminalOutput(OutputAbstract):

    def __init__(self) -> None:
        super().__init__()

    def send_out(self, value: Any, status: str = "") -> None:
        o = status + "(value: " + value + ")"
        print(o)
