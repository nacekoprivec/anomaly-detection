from abc import abstractclassmethod, abstractmethod
from abc import ABC
from typing import Any, Dict, List
import numpy as np

from src.output import OutputAbstract, TerminalOutput, GraphOutput, HistogramOutput



class AnomalyDetectionAbstract(ABC):
    memory_size: int
    memory: List[Any]
    outputs: List["OutputAbstract"]

    def __init__(self) -> None:
        self.memory = []

    @abstractmethod
    def message_insert(self, message_value: Dict[Any, Any]) -> None:
        pass

    @abstractmethod
    def configure(self, conf: Dict[Any, Any]) -> None:
        pass


class BorderCheck(AnomalyDetectionAbstract):
    UL: float
    LL: float
    warning_stages: List[float]

    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(conf)
        else:
            default = {
                "UL": 5,
                "LL": 0,
                "warning_stages": [0.9],
                "output": ["TerminalOutput()"],
                "output_conf": [
                    {}
                ]
            }
            self.configure(default)

    def configure(self, conf: Dict[Any, Any] = None) -> None:
        self.LL = conf["LL"]
        self.UL = conf["UL"]

        self.warning_stages = conf["warning_stages"]
        self.warning_stages.sort()

        # initialize all outputs
        self.outputs = [eval(o) for o in conf["output"]]

        # configure all outputs
        output_configurations = conf["output_conf"]
        for o in range(len(self.outputs)):
            self.outputs[o].configure(output_configurations[o])

    def message_insert(self, message_value: Dict[Any, Any]) -> None:
        value = float(message_value['test_value'])

        value_normalized = 2*(value - (self.UL + self.LL)/2) / \
            (self.UL - self.LL)
        status = "OK"

        if(value_normalized > 1):
            status = "Error: measurement above upper limit"
        elif(value_normalized < -1):
            status = "Error: measurement below lower limit"
        else:
            for stage in range(len(self.warning_stages)):
                if(value_normalized > self.warning_stages[stage]):
                    status = "Warning" + str(stage) + \
                        ": measurement close to upper limit."
                elif(value_normalized < -self.warning_stages[stage]):
                    status = "Warning" + str(stage) + \
                        ": measurement close to lower limit."
                else:
                    break

        for output in self.outputs:
            output.send_out(status=status, value=value)


class EMA(AnomalyDetectionAbstract):
    UL: float
    LL: float
    value_index: int

    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(conf)
        else:
            default = {
                "N": 5,
                "num_of_points": 50,
                "UL": 10,
                "LL": 0,
                "title": "Title",
                "output": ["TerminalOutput()"],
                "output_conf": [
                    {}
                ]
            }
            self.configure(default)

    def configure(self, conf: Dict[Any, Any] = None) -> None:
        self.N = conf['N']
        self.smoothing = 2/(self.N+1)
        self.UL = conf['UL']
        self.LL = conf['LL']
        self.title = conf['title']
        self.lines = [[], [], [], [], []]
        self.EMA = []
        self.sigma = []
        self.numbers = []
        self.timestamps = []

        self.outputs = [eval(o) for o in conf["output"]]

        # configure all outputs
        output_configurations = conf["output_conf"]
        for o in range(len(self.outputs)):
            self.outputs[o].configure(output_configurations[o])

    def message_insert(self, message_value: Dict[Any, Any]):
        self.numbers.append(float(message_value['test_value']))
        self.timestamps.append(float(message_value['timestamp']))
        if(len(self.EMA) == 0):
            self.EMA.append(self.numbers[-1])
        else:
            new = self.numbers[-1] * self.smoothing + self.EMA[-1] *\
                (1-self.smoothing)
            self.EMA.append(new)

        if(len(self.numbers) == 1):
            self.sigma.append(0)
        elif(len(self.numbers) < self.N):
            self.sigma.append(np.std(self.numbers))
        else:
            self.sigma.append(np.std(self.numbers[-self.N:]))

        for output in self.outputs:
            output.send_out(timestamp=float(message_value['timestamp']), value=message_value["test_value"])
        return (self.EMA[-1], self.sigma[-1])
