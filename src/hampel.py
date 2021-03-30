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

class Hampel(AnomalyDetectionAbstract):
    name: str = "Hampel"

    W: int
    new_value: float
    K: float


    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(conf)

    def configure(self, conf: Dict[Any, Any] = None,
                  configuration_location: str = None,
                  algorithm_indx: int = None) -> None:
        super().configure(conf, configuration_location=configuration_location,
                          algorithm_indx=algorithm_indx)
        if ('W' in conf):
            self.W = conf['W']
            self.memory = [None] * (2*self.W + 1)
        else:
            self.W = None

        self.K = conf["K"]
        self.n_sigmas = conf["n_sigmas"]
        self.count = 0


    def message_insert(self, message_value: Dict[Any, Any]):
        super().message_insert(message_value)
        # extract from message
        value = message_value["ftr_vector"]
        value = value[0]
        timestamp = message_value["timestamp"]

        status = self.OK
        status_code = self.OK_CODE

        suggested_value = None

        if(self.W is not None):
            self.memory.append(value)
            self.memory = self.memory[-(2*self.W + 1):]

        #Nothing is done until there is enough values to apply the window
        if (self.count < (2*self.W + 1)):
            if(self.memory[self.W + 1] is not None):
                suggested_value = self.memory[self.W + 1]
                status = self.UNDEFINED
                status_code = self.UNDEFIEND_CODE
            else:
                suggested_value = None
                status = self.UNDEFINED
                status_code = self.UNDEFIEND_CODE
        
        else:
            median = np.median(self.memory)
            S0 = self.K * np.median(np.abs(self.memory - median))
            if(np.abs(self.memory[self.W+1] - median) > self.n_sigmas * S0):
                suggested_value = median
                status = "Anomaly detected"
                status_code = self.ERROR_CODE
            else:
                suggested_value = self.memory[self.W+1]
                status = self.OK
                status_code = self.OK_CODE
  



        # Outputs and visualizations
        output_value = self.memory[self.W+1]
        if(output_value is not None):
            for output in self.outputs:
                output.send_out(timestamp=timestamp, status=status,
                                suggested_value=suggested_value,
                                value=value,
                                status_code=status_code, algorithm=self.name)

        if(suggested_value is not None):
            if(self.visualization is not None):
                # print(status_code)
                lines = [suggested_value]
                self.visualization.update(value=lines, timestamp=timestamp,
                                        status_code=status_code)
        
        self.count += 1
