from abc import abstractmethod
from abc import ABC
from typing import Any, Dict


class OutputAbstract(ABC):

    def __init__(self) -> None:
        pass

    @abstractmethod
    def configure(self, conf: Dict[Any, Any]) -> None:
        pass

    @abstractmethod
    def send_out(self, value: Any, status: str, timestamp: Any) -> None:
        pass


class TerminalOutput(OutputAbstract):

    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()

    def configure(self, conf: Dict[Any, Any] = None) -> None:
        # Nothing to configure
        pass

    def send_out(self,  value: Any, status: str = "",
                 timestamp: Any = 0) -> None:
        o = status + "(value: " + str(value) + ")"
        print(o)
