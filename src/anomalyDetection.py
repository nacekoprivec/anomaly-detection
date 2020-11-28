from abc import abstractclassmethod, abstractmethod
from abc import ABC
from typing import Any, Dict, List
import numpy as np

from output import OutputAbstract, TerminalOutput, GraphOutput


class AnomalyDetectionAbstract(ABC):
    memory_size: int
    memory: List[Any]
    outputs: List["OutputAbstract"]

    def __init__(self) -> None:
        self.memory = []

    #@abstractmethod
    def check(self) -> None:
        pass

    #@abstractmethod
    def configure(self) -> None:
        pass


class BorderCheck(AnomalyDetectionAbstract):
    UL: float
    LL: float
    value_index: int
    warning_stages: List[float]

    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(conf)
        else:
            default={
                "memory_size": 5,
                "UL": 5,
                "LL": 0,
                "warning_stages": [0.9],
                "value_index": 0,
                "output": [TerminalOutput()],
                "output_conf": [
                    {}
                ]
            }
            self.configure(default)
        

    def configure(self, conf: Dict[Any, Any] = None) -> None:
        self.memory_size = conf["memory_size"]
        self.LL = conf["LL"]
        self.UL = conf["UL"]
        self.value_index = conf["value_index"]

        self.warning_stages = conf["warning_stages"]
        self.warning_stages.sort()

        self.outputs = conf["output"]

        # configure all outputs
        output_configurations = conf["output_conf"]
        for o in range(len(self.outputs)):
            self.outputs[o].configure(output_configurations[o])

    def check(self, new: List[Any]) -> None:
        # TODO: warning for runnig average
        # inserts new element and deletes old
        self.memory.insert(0, new)
        self.memory = self.memory[:self.memory_size]

        value = new[self.value_index]

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
                    status = "Warning" + stage + \
                        ": measurement close to upper limit."
                elif(value_normalized < -self.warning_stages[stage]):
                    status = "Warning" + stage + \
                        ": measurement close to lower limit."
                else:
                    break

        for output in self.outputs:
            output.send_out(status=status, value=value)

class EMA(AnomalyDetectionAbstract):
    UL: float
    LL: float
    value_index: int
    warning_stages: List[float]

    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(conf)
        else:
            default={
                "N": 5,
                "num_of_points": 50,
                "UL": 10,
                "LL": 0,
                "title": "Title",
                "output": [TerminalOutput()],
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

    #def check(self, new: List[Any]) -> None:

    def message_insert(self, message):
        self.numbers.append(float(message.value['test_value']))
        self.timestamps.append(float(message.value['timestamp']))
        if(len(self.EMA) == 0):
            self.EMA.append(self.numbers[-1])
        else:
            self.EMA.append(self.numbers[-1] * self.smoothing + self.EMA[-1]*(1-self.smoothing))

        if(len(self.numbers) == 1):
            self.sigma.append(0)
        elif(len(self.numbers)<self.N):
            self.sigma.append(np.std(self.numbers))
        else:
            self.sigma.append(np.std(self.numbers[-self.N:]))
        return (self.EMA[-1], self.sigma[-1])
        

