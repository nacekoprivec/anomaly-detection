from abc import abstractclassmethod, abstractmethod
from abc import ABC
from typing import Any, Dict, List
import numpy as np

from src.output import OutputAbstract, TerminalOutput
from src.visualization import VisualizationAbstract, GraphVisualization,\
    HistogramVisualization


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
    outputs: List["OutputAbstract"]
    visualizations: List["VisualizationAbstract"]

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

        if ("visualization" in conf):
            self.visualizations = [eval(v) for v in conf["visualization"]]
            # configure all visualizations
            visualization_configurations = conf["visualization_conf"]
            for v in range(len(self.visualizations)):
                self.visualizations[v].configure(visualization_configurations[v])
        else:
            self.visualizations = []

    def message_insert(self, message_value: Dict[Any, Any]) -> None:
        value = float(message_value["test_value"])
        timestamp = message_value["timestamp"]

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
        
        lines = [value]
        for visualization in self.visualizations:
            visualization.update(value=lines, timestamp=timestamp)


class EMA(AnomalyDetectionAbstract):
    UL: float
    LL: float
    N: int
    smoothing: float
    EMA: List[float]
    sigma: List[float]
    numbers: List[float]
    timestamp: List[Any]
    visualizations: List["VisualizationAbstract"]
    outputs: List["OutputAbstract"]

    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(conf)
        else:
            default = {
                "N": 5,
                "num_of_points": 50,
                "output": ["TerminalOutput()"],
                "output_conf": [
                    {}
                ]
            }
            self.configure(default)

    def configure(self, conf: Dict[Any, Any] = None) -> None:
        self.N = conf['N']
        self.smoothing = 2/(self.N+1)
        self.EMA = []
        self.sigma = []
        self.numbers = []
        self.timestamps = []

        self.outputs = [eval(o) for o in conf["output"]]
        # configure all outputs
        output_configurations = conf["output_conf"]
        for o in range(len(self.outputs)):
            self.outputs[o].configure(output_configurations[o])

        if ("visualization" in conf):
            self.visualizations = [eval(v) for v in conf["visualization"]]
            # configure all visualizations
            visualization_configurations = conf["visualization_conf"]
            for v in range(len(self.visualizations)):
                self.visualizations[v].configure(visualization_configurations[v])
        else:
            self.visualizations = []

    def message_insert(self, message_value: Dict[Any, Any]):
        self.numbers.append(float(message_value['test_value']))
        self.timestamps.append(message_value['timestamp'])
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
            output.send_out(timestamp=float(message_value['timestamp']),
                            value=message_value["test_value"])

        mean = self.EMA[-1]
        sigma = self.sigma[-1]
        lines = [self.numbers[-1], mean, mean+sigma, mean-sigma]
        timestamp = self.timestamps[-1]
        for visualization in self.visualizations:
            visualization.update(value=lines, timestamp=timestamp)
