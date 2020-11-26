from ABC import abstractmethod
from abc import ABC
from typing import Any, Dict


class OutputAbstract(ABC):

    def __init__(self) -> None:
        pass

    @abstractmethod
    def configure(self) -> None:
        pass

    @abstractmethod
    def send_out(self, value: Any, status: str) -> None:
        pass


class TerminalOutput(OutputAbstract):

    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()

    def configure(self, conf: Dict[Any, Any] = None) -> None:
        # Nothing to configure
        pass

    def send_out(self, value: Any, status: str = "") -> None:
        o = status + "(value: " + value + ")"
        print(o)


class GraphOutput(OutputAbstract):
    num_of_points: int

    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(conf=conf)
        else:
            # TODO make up default configuration 
            default = {}
            self.configure(conf=default)

    def configure(self, conf: Dict[Any, Any] = None) -> None:
        # TODO: configure parameters
        pass

    def send_out(self, value: Any, status: str = "") -> None:
        # TODO basicly live plotter from MA1
        # would probably be smart to save last num_of_points values that are shown on graph
        pass
