from abc import abstractmethod, ABC
from typing import Any, Dict, List, Tuple, Union
import numpy as np
import statistics
import sys
import math
import os
import json
import ast
from statistics import mean
from datetime import datetime, time
import pickle
from pandas.core.frame import DataFrame
from scipy.signal.lti_conversion import _atleast_2d_or_none
import sklearn.ensemble
from scipy import signal
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from ast import literal_eval

sys.path.insert(0,'./src')
sys.path.insert(1, 'C:/Users/Matic/SIHT/anomaly_det/anomalyDetection/')
from anomalyDetection import AnomalyDetectionAbstract
from output import OutputAbstract, TerminalOutput, FileOutput, KafkaOutput
from visualization import VisualizationAbstract, GraphVisualization,\
    HistogramVisualization, StatusPointsVisualization
from normalization import NormalizationAbstract, LastNAverage,\
    PeriodicLastNAverage

class EMA(AnomalyDetectionAbstract):
    name: str = "EMA"

    UL: float
    LL: float
    N: int
    smoothing: float
    EMA: List[float]
    sigma: List[float]
    numbers: List[float]
    timestamp: List[Any]

    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(conf)

    def configure(self, conf: Dict[Any, Any] = None,
                  configuration_location: str = None,
                  algorithm_indx: int = None) -> None:
        super().configure(conf, configuration_location=configuration_location,
                          algorithm_indx=algorithm_indx)

        self.LL = conf["LL"]
        self.UL = conf["UL"]
        self.warning_stages = conf["warning_stages"]
        self.warning_stages.sort()
        self.N = conf['N']
        self.smoothing = 2/(self.N+1)
        self.EMA = []
        self.sigma = []
        self.numbers = []
        self.timestamps = []

    def message_insert(self, message_value: Dict[Any, Any]):
        super().message_insert(message_value)

        value = message_value["ftr_vector"]
        value = value[0]

        self.numbers.append(value)
        self.timestamps.append(message_value['timestamp'])
        if (len(self.numbers) > self.N):
            self.numbers = self.numbers[-self.N:]
            self.timestamps = self.timestamps[-self.N:]

        #Calculate exponential moving average
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


        #Normalize the moving average to the range LL - UL
        value_normalized = 2*(self.EMA[-1] - (self.UL + self.LL)/2) / \
            (self.UL - self.LL)
        status = self.OK
        status_code = self.OK_CODE

        #Perform error and warning checks
        if(value_normalized > 1):
            status = "Error: EMA above upper limit"
            status_code = -1
        elif(value_normalized < -1):
            status = "Error: EMA below lower limit"
            status_code = -1
        else:
            for stage in range(len(self.warning_stages)):
                if(value_normalized > self.warning_stages[stage]):
                    status = "Warning" + str(stage) + \
                        ": EMA close to upper limit."
                    status_code = 0
                elif(value_normalized < -self.warning_stages[stage]):
                    status = "Warning" + str(stage) + \
                        ": EMA close to lower limit."
                    status_code = 0
                else:
                    break

        self.status = status
        self.status_code = status_code

        for output in self.outputs:
            output.send_out(timestamp=message_value["timestamp"],
                            algorithm=self.name, status=status,
                            value=message_value['ftr_vector'],
                            status_code=status_code)

        #send EMA and +- sigma band to visualization
        
        # mean = self.EMA[-1]
        #sigma = self.sigma[-1]
        lines = [self.numbers[-1]]
        timestamp = self.timestamps[-1]
        if(self.visualization is not None):
            self.visualization.update(value=lines, timestamp=message_value["timestamp"],
            status_code = status_code)
