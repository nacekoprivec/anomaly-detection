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

class Filtering(AnomalyDetectionAbstract):
    name: str = "Filtering"

    UL: float
    LL: float
    value_normalized: float
    filtered: float
    result: float

    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(conf)

    def configure(self, conf: Dict[Any, Any] = None,
                  configuration_location: str = None,
                  algorithm_indx: int = None) -> None:
        super().configure(conf, configuration_location=configuration_location,
                          algorithm_indx=algorithm_indx)

        self.mode = conf["mode"]
        self.filter_order = conf["filter_order"]
        self.cutoff_frequency = conf["cutoff_frequency"]
        self.LL = conf["LL"]
        self.UL = conf["UL"]
        self.warning_stages = conf["warning_stages"]
        self.warning_stages.sort()
        self.filtered = []
        self.numbers = []
        self.timestamps = []

        #Initalize the digital filter
        self.b, self.a = signal.butter(self.filter_order, self.cutoff_frequency)
        self.z = signal.lfilter_zi(self.b, self.a)

    def message_insert(self, message_value: Dict[Any, Any]):
        super().message_insert(message_value)

        self.last_measurement = message_value['ftr_vector'][0]

        #Update filter output after last measurement
        filtered_value, self.z = signal.lfilter(self.b, self.a, x = [self.last_measurement], zi = self.z)

        self.filtered = filtered_value[0]
        
        #Normalize filter outputs
        value_normalized = 2*(self.filtered - (self.UL + self.LL)/2) / \
            (self.UL - self.LL)
        status = "OK"
        status_code = 1

        #Perform error and warning checks
        if(self.mode == 1):
            deviation = (self.last_measurement - self.filtered)/(self.UL - self.LL)
            if(deviation > 1):
                status = "Error: Large deviation"
                status_code = -1
            elif(value_normalized < -1):
                status = "Error: Large deviation"
                status_code = -1
            else:
                for stage in range(len(self.warning_stages)):
                    if(deviation > self.warning_stages[stage]):
                        status = "Warning" + str(stage) + \
                            ": Significant deviation."
                        status_code = 0
                    elif(deviation < -self.warning_stages[stage]):
                        status = "Warning" + str(stage) + \
                            ": Significant deviation."
                        status_code = 0
                    else:
                        break
        else:
            if(value_normalized > 1):
                status = "Error: Filtered signal above upper limit"
                status_code = -1
            elif(value_normalized < -1):
                status = "Error: Filtered signal below lower limit"
                status_code = -1
            else:
                for stage in range(len(self.warning_stages)):
                    if(value_normalized > self.warning_stages[stage]):
                        status = "Warning" + str(stage) + \
                            ": Filtered signal close to upper limit."
                        status_code = 0
                    elif(value_normalized < -self.warning_stages[stage]):
                        status = "Warning" + str(stage) + \
                            ": Filtered signal close to lower limit."
                        status_code = 0
                    else:
                        break
        
        self.status = status
        self.status_code = status_code

        if(self.mode == 0):
            result = self.filtered
        else:
            result = self.last_measurement - self.filtered

        for output in self.outputs:
            output.send_out(timestamp=message_value["timestamp"],
                            status=status, value=message_value['ftr_vector'][0], 
                            status_code=status_code, algorithm=self.name)

        self.result = result
        lines = [result, self.last_measurement]
        #timestamp = self.timestamps[-1]
        if(self.visualization is not None):
            self.visualization.update(value=lines, timestamp=message_value["timestamp"],
            status_code = status_code)
