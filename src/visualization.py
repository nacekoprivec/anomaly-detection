from abc import ABC, abstractmethod
from typing import Any, Dict, List
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')


class VisualizationAbstract(ABC):
    @abstractmethod
    def configure(self, conf: Dict[Any, Any] = None) -> None:
        pass

    @abstractmethod
    def update(self, timestamp: Any, value: List[Any]) -> None:
        pass


class GraphVisualization(VisualizationAbstract):
    num_of_points: int

    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(conf=conf)
        else:
            default = {"num_of_points": 50,
                       "num_of_lines": 1,
                       "linestyles": ['ro']}
            self.configure(conf=default)

    def configure(self, conf: Dict[Any, Any] = None) -> None:
        self.num_of_points = conf["num_of_points"]
        self.num_of_lines = conf["num_of_lines"]
        self.linestyles = conf["linestyles"]
        self.lines = [[] for _ in range(self.num_of_lines)]
        pass

    def update(self, value: List[Any], timestamp: Any = 0) -> None:
        assert self.num_of_lines <= len(value), "Configuration specifies more lines that were given."
        # value is an array
        # [lastvalue, current_moving_average, current_moving_average+sigma,
        # current_moving_average-sigma] how many of those you actually need
        # depends on num of lines
        x_data = []
        y_data = value.copy()

        # define or update lines
        if self.lines[0] == []:
            plt.ion()
            fig_graph = plt.figure(figsize=(13, 6))
            ax_graph = [None] * self.num_of_lines
            for i in range(self.num_of_lines):
                ax_graph[i] = fig_graph.add_subplot(111)
            x_data = [float(timestamp)]
            y_data = value.copy()
            for i in range(self.num_of_lines):
                self.lines[i], = ax_graph[i].plot(x_data, y_data[i],
                                            self.linestyles[i], alpha=0.8)
                plt.show()

        if (len(self.lines[0].get_data()[0]) < self.num_of_points):
            x_data = [None] * self.num_of_points
            x_data = np.append(x_data, self.lines[0].get_data()[0])
            x_data = np.append(x_data, float(timestamp))
            x_data = x_data[-self.num_of_points:]
            for i in range(self.num_of_lines):
                y_data[i] = [None]*self.num_of_points
                y_data[i] = np.append(y_data[i], self.lines[i].get_data()[1])
                y_data[i] = np.append(y_data[i], value[i])
                y_data[i] = y_data[i][-self.num_of_points:]
        else:
            x_data = self.lines[0].get_data()[0]
            x_data = np.append(x_data, float(timestamp))
            x_data = x_data[-self.num_of_points:]
            for i in range(self.num_of_lines):
                y_data[i] = self.lines[i].get_data()[1]
                y_data[i] = np.append(y_data[i], value[i])
                y_data[i] = y_data[i][-self.num_of_points:]

        for i in range(self.num_of_lines):
            self.lines[i].set_ydata(y_data[i])
            self.lines[i].set_xdata(x_data)

        # plot limits correction
        if(value is not None):
            if (min(value) <= self.lines[0].axes.get_ylim()[0]) or (max(value) >= self.lines[0].axes.get_ylim()[1]):
                plt.subplot(111).set_ylim([min(filter(lambda x: x is not None, self.lines[0].get_data()[1])) - 1,
                                          max(filter(lambda x: x is not None, self.lines[0].get_data()[1])) + 1])

            plt.subplot(111).set_xlim([min(filter(lambda x: x is not None, x_data)) - 1, max(filter(lambda x: x is not None, x_data))+1])
        plt.pause(0.1)


class HistogramVisualization(VisualizationAbstract):
    num_of_bins: int
    range: List[int]
    bins: Any
    bin_vals: List[Any]
    line: List[Any]

    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(conf=conf)
        else:
            default = {"num_of_bins": 50,
                       "range": [0, 10]}
            self.configure(conf=default)

    def configure(self, conf: Dict[Any, Any] = None) -> None:
        self.num_of_bins = conf["num_of_bins"]
        self.range = conf["range"]
        self.bins = np.linspace(self.range[0], self.range[1], self.num_of_bins)
        self.bin_vals = np.zeros(len(self.bins))
        self.line = []

    def update(self, value: List[Any], timestamp: Any = 0) -> None:
        if (self.line == []):
            fig_hist = plt.figure(figsize=(13, 6))
            ax_hist = fig_hist.add_subplot(111)

            self.bin_vals[np.digitize([value[0]], self.bins)] = 1
            self.line, = ax_hist.step(self.bins, self.bin_vals)
        else:
            self.bin_vals[np.digitize(value[0], self.bins)] += 1
            self.line.set_ydata(self.bin_vals)

        if(value[0] is not None):
            if (max(self.bin_vals) >= self.line.axes.get_ylim()[1]):
                plt.subplot(111).set_ylim([0, max(self.bin_vals)+1])

        plt.pause(0.1)
