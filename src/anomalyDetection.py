from abc import abstractmethod, ABC
from typing import Any, Dict, List, Tuple, Union
import numpy as np
import statistics
import sys
import math
import os
import json
from statistics import mean
from datetime import datetime, time
import pickle
from pandas.core.frame import DataFrame
from scipy.signal.lti_conversion import _atleast_2d_or_none
import sklearn.ensemble
from scipy import signal
#from tensorflow.keras import backend as K
#import tensorflow as tf
#from tensorflow import keras
import pandas as pd
from ast import literal_eval

sys.path.insert(0,'./src')
sys.path.insert(1, 'C:/Users/Matic/SIHT/anomaly_det/anomalyDetection/')
from output import OutputAbstract, TerminalOutput, FileOutput, KafkaOutput
from visualization import VisualizationAbstract, GraphVisualization,\
    HistogramVisualization, StatusPointsVisualization
from normalization import NormalizationAbstract, LastNAverage,\
    PeriodicLastNAverage


class AnomalyDetectionAbstract(ABC):
    configuration_location: str

    # needed if there are more anomaly detection algorithms
    algorithm_indx: int

    memory_size: int
    memory: List[List[Any]]
    averages: List[List[int]]
    periodic_averages: List[List[Tuple[int, List[int]]]]
    shifts: List[List[int]]
    time_features: List[str]
    name: str

    use_cols: List[int]
    input_vector_size: int
    outputs: List["OutputAbstract"]
    visualization: "VisualizationAbstract"
    normalization: "NormalizationAbstract"

    # Statuses
    UNDEFINED = "Undefined"
    ERROR = "Error"
    WARNING = "Warning"
    OK = "OK"

    # Status codes
    UNDEFIEND_CODE = 2
    ERROR_CODE = -1
    WARNING_CODE = 0
    OK_CODE = 1

    def __init__(self) -> None:
        self.memory = []

    @abstractmethod
    def message_insert(self, message_value: Dict[Any, Any]) -> None:
        # print("test value: " + str(message_value['ftr_vector']))
        # print(message_value['ftr_vector'])
        print(message_value['ftr_vector'])
        print(len(message_value['ftr_vector']))
        print(self.input_vector_size)
        assert len(message_value['ftr_vector']) == self.input_vector_size, \
            "Given test value does not satisfy input vector size"

    @abstractmethod
    def configure(self, conf: Dict[Any, Any],
                  configuration_location: str = None,
                  algorithm_indx: int = None) -> None:
        self.configuration_location = configuration_location
        
        # If algorithm is initialized from consumer kafka it has this
        # specified
        self.algorithm_indx = algorithm_indx
        
        # FEATURE CONSTRUCTION CONFIGURATION
        self.input_vector_size = conf["input_vector_size"]

        if("braila_fall_feature" in conf):
            self.braila_fall = conf["braila_fall_feature"]
        else:
            self.braila_fall = []
        
        if("averages" in conf):
            self.averages = conf["averages"]
        else:
            self.averages = []

        if("periodic_averages" in conf):
            self.periodic_averages = conf["periodic_averages"]
        else:
            self.periodic_averages = []

        if("shifts" in conf):
            self.shifts = conf["shifts"]
        else:
            self.shifts = []

        if("time_features" in conf):
            self.time_features = conf["time_features"]
        else:
            self.time_features = []

        # Finds the largest element among averages and shifts
        if(len(self.shifts) == 0):
            max_shift = 0
        else:
            max_shifts = []
            for shifts in self.shifts:
                if(len(shifts) == 0):
                    max_shifts.append(0)
                else:
                    max_shifts.append(max(shifts))
            max_shift = max(max_shifts)+1

        if (len(self.averages) == 0):
            max_average = 0
        else:
            max_averages = []
            for averages in self.averages:
                if(len(averages) == 0):
                    max_averages.append(0)
                else:
                    max_averages.append(max(averages))
            max_average = max(max_averages)

        if (len(self.periodic_averages) == 0):
            max_periodic_average = 0
        else:
            max_periodic_average = 0
            for feature_avgs in self.periodic_averages:
                for period_tuple in feature_avgs:
                    period = period_tuple[0]
                    # assumes period specifies at least one average
                    max_avg = max(period_tuple[1])
                    # Memory required to calculate the average
                    required_memory = 1+(period * (max_avg-1))
                    if(required_memory > max_periodic_average):
                        max_periodic_average = required_memory

        # one because of feature construction memory management
        self.memory_size = max(max_shift, max_average, max_periodic_average, 1)

        # OUTPUT/VISUALIZATION INITIALIZATION & CONFIGURATION
        self.outputs = [eval(o) for o in conf["output"]]
        output_configurations = conf["output_conf"]
        for o in range(len(self.outputs)):
            self.outputs[o].configure(output_configurations[o])
        if ("visualization" in conf):
            self.visualization = eval(conf["visualization"])
            visualization_configurations = conf["visualization_conf"]
            self.visualization.configure(visualization_configurations)
        else:
            self.visualization = None

        # NORMALIZATION INITIALIZATION & CONFIGURATION
        if("normalization" in conf):
            self.normalization = eval(conf["normalization"])
            normalization_configuration = conf["normalization_conf"]
            self.normalization.configure(normalization_configuration) 
        else:
            self.normalization = None
        
        # If specified save which columns to use
        if("use_cols" in conf):
            self.use_cols = conf["use_cols"]
        else:
            self.use_cols = None
    
    def change_last_record(self, value: List[Any]) -> None:
        self.memory[-1] = value

    def training_feature_construction(self, data, timestamps) -> List[Any]:
        # Saves memory so it can be restored
        memory_backup = self.memory
        self.memory = []

        # Aditional feature construction
        features = []
        for i in range(len(data)):
            value = data[i].tolist()
            timestamp = timestamps[i]
            feature_vector = self.feature_construction(value=value,
                                                      timestamp=timestamp)

            if(feature_vector is not False):
                features.append(np.array(feature_vector))

        self.memory = memory_backup
        return features

    def feature_construction(self, value: List[Any], 
                             timestamp: str) -> Union[Any, bool]:

        # Add new value to memory and slice it
        self.memory.append(value)
        self.memory = self.memory[-self.memory_size:]

        if(len(self.memory) < self.memory_size):
            # The memory does not contain enough records for all shifts and 
            # averages to be created
            return False

        # create new value to be returned
        new_value = value.copy()

        # Create average features
        new_value.extend(self.average_construction())

        # Create periodic averages
        new_value.extend(self.periodic_average_construction())

        # Create shifted features
        new_value.extend(self.shift_construction())

        # Create time features
        new_value.extend(self.time_features_construction(timestamp))

        return new_value

    def average_construction(self) -> None:
        averages = []
        np_memory = np.array(self.memory)

        # Loop through all features
        for feature_index in range(len(self.averages)):
            # Loop through all horizons we want the average of
            for interval in self.averages[feature_index]:
                # Add last interval numbers to values array
                values = []
                for sample_indx in range(len(self.memory)):
                    if(sample_indx == interval):
                        break
                    values.append(self.memory[-(sample_indx+1)][feature_index])
                averages.append(mean(values))

        return averages

    def periodic_average_construction(self) -> None:
        # construct periodic averages

        periodic_averages = []
        # Loop through features
        for feature_indx in range(len(self.periodic_averages)):
            # Loop through indexes for different features
            for period_indx in range(len(self.periodic_averages[feature_indx])):
                # Extract period and a list of how N-s (number of samples to
                # take from this sequence)
                period = self.periodic_averages[feature_indx][period_indx][0]
                averages = self.periodic_averages[feature_indx][period_indx][1]

                # Loop through N-s (number of samples to take from this
                # sequence)
                for average in averages:
                    # Construct a list of semples with this period from memory
                    periodic_list = []
                    # Loop through samples (in opposite direction) and if they
                    # are of right period add them to the list
                    for i in range(self.memory_size):
                        if(len(periodic_list) == average):
                            # Enough samples
                            break
                        if(i%period==0):
                            periodic_list.append(self.memory[self.memory_size-(i+1)][feature_indx])
                    
                    # print("periodic list:")
                    # print(periodic_list)
                    
                    # Append average of the list to features
                    avg = mean(periodic_list)
                    periodic_averages.append(avg)

        return periodic_averages

    def shift_construction(self) -> None:
        shifts = []

        # Loop through all features
        for feature_index in range(len(self.shifts)):
            # Loop through all shift values
            for look_back in self.shifts[feature_index]:
                shifts.append(self.memory[self.memory_size-(look_back+1)][feature_index])

        return shifts


    def time_features_construction(self, tmstp: Any) -> None:
        time_features = []
        
        if(type(tmstp) is not str):
            tmstp = datetime.utcfromtimestamp(tmstp).strftime('%Y-%m-%d %H:%M:%S')
        if(len(str(tmstp)) == 19):
            dt = datetime.strptime(tmstp, '%Y-%m-%d %H:%M:%S')
        else:
            dt = datetime.strptime(tmstp, '%Y-%m-%d %H:%M:%S.%f')

        # Requires datetime format
        # Check for keywords specified in time_features
        if("month" in self.time_features):
            time_features.append(int(dt.month))
        if ("day" in self.time_features):
            time_features.append(int(dt.day))
        if ("weekday" in self.time_features):
            time_features.append(int(dt.weekday()))
        if ("hour" in self.time_features):
            time_features.append(int(dt.hour))
        if ("minute" in self.time_features):
            time_features.append(int(dt.minute))


        return time_features

    def normalization_output_visualization(self, status_code: int,
                                           status: str, value: List[Any],
                                           timestamp: Any) -> None:
        # Normalize if needed or just add the value
        normalized = None
        if(self.normalization is not None):
            if(status_code == -1):
                normalized = self.normalization.get_normalized(value=value)
                if(normalized != False):
                    self.change_last_record(value=normalized)
                else:
                    normalized = None
            else:
                self.normalization.add_value(value=value)
        
        for output in self.outputs:
            output.send_out(timestamp=timestamp, status=status,
                            suggested_value=normalized,
                            value=value,
                            status_code=status_code, algorithm=self.name)

        if(self.visualization is not None):
            lines = [value[0]]
            self.visualization.update(value=lines, timestamp=timestamp,
                                      status_code=status_code)


class BorderCheck(AnomalyDetectionAbstract):
    """ works with 1D data and checks if the value is above, below or close
    to guven upper and lower limits 
    """
    UL: float
    LL: float
    warning_stages: List[float]
    name: str = "Border check"

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

    def message_insert(self, message_value: Dict[Any, Any]) -> None:
        super().message_insert(message_value)
        value = message_value["ftr_vector"]
        value = value[0]
        timestamp = message_value["timestamp"]

        value_normalized = 2*(value - (self.UL + self.LL)/2) / \
            (self.UL - self.LL)
        status = "OK"
        status_code = 1

        if(value_normalized > 1):
            status = "Error: measurement above upper limit"
            status_code = -1
        elif(value_normalized < -1):
            status = "Error: measurement below lower limit"
            status_code = -1
        else:
            for stage in range(len(self.warning_stages)):
                if(value_normalized > self.warning_stages[stage]):
                    status = "Warning" + str(stage) + \
                        ": measurement close to upper limit."
                    status_code = 0
                elif(value_normalized < -self.warning_stages[stage]):
                    status = "Warning" + str(stage) + \
                        ": measurement close to lower limit."
                    status_code = 0
                else:
                    break
        
        self.status = status
        self.status_code = status_code

        for output in self.outputs:
            output.send_out(timestamp=timestamp, status=status, value=message_value["ftr_vector"],
                            status_code=status_code, algorithm=self.name)

        if(self.visualization is not None):
            lines = [value]
            self.visualization.update(value=lines, timestamp=timestamp,
                                      status_code=status_code)


class Welford(AnomalyDetectionAbstract):
    name: str = "Welford"

    UL: float
    LL: float
    N: int
    count: int
    memory: List[float]
    X: float
    mean: float
    s: float
    warning_stages: List[float]

    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(conf)

    def configure(self, conf: Dict[Any, Any] = None,
                  configuration_location: str = None,
                  algorithm_indx: int = None) -> None:
        super().configure(conf, configuration_location=configuration_location,
                          algorithm_indx=algorithm_indx)

        if ('N' in conf):
            self.N = conf['N']
            self.memory = [None] * self.N
        else:
            self.N = None

        self.UL = self.LL = None
        self.count = 0

        self.X = conf["X"]
        self.warning_stages = conf["warning_stages"]
        self.warning_stages.sort()

    def message_insert(self, message_value: Dict[Any, Any]):
        super().message_insert(message_value)
        # extract from message
        value = message_value["ftr_vector"]
        value = value[0]
        timestamp = message_value["timestamp"]

        # First value is undefined
        if (self.count == 0):
            self.mean = value
            self.s = 0
            status = self.UNDEFINED
            status_code = self.UNDEFIEND_CODE
        # Infinite stream and has at least 2 elements so far, or finite stream
        # with full memory
        elif ((self.N is None and self.count > 1) or
             (self.N is not None and self.N <= self.count)):
            value_normalized = 2*(value - (self.UL + self.LL)/2) / \
                (self.UL - self.LL)
            status = self.OK
            status_code = self.OK_CODE

            if(value_normalized > 1):
                status = "Error: measurement above upper limit"
                status_code = -1
            elif(value_normalized < -1):
                status = "Error: measurement below lower limit"
                status_code = -1
            else:
                for stage in range(len(self.warning_stages)):
                    if(value_normalized > self.warning_stages[stage]):
                        status = "Warning" + str(stage) + \
                            ": measurement close to upper limit."
                        status_code = 0
                    elif(value_normalized < -self.warning_stages[stage]):
                        status = "Warning" + str(stage) + \
                            ": measurement close to lower limit."
                        status_code = 0
                    else:
                        break
        else:
            status = self.UNDEFINED
            status_code = self.UNDEFIEND_CODE

        self.status = status
        self.status_code = status_code

        # Outputs and visualizations
        for output in self.outputs:
            output.send_out(timestamp=timestamp, status=status,
                            value=message_value["ftr_vector"],
                            status_code=status_code, algorithm=self.name)

        if(self.visualization is not None):
            lines = [value]
            self.visualization.update(value=lines, timestamp=timestamp,
                                      status_code=status_code)
        
        self.count += 1
        # adjust mean and stdev for the current window
        if(self.N is not None):
            self.memory.append(value)
            self.memory = self.memory[-self.N:]
            # If the memory is not full calculate stdev, mean LL and UL
            if(self.count >= self.N):
                self.mean = statistics.mean(self.memory)
                self.s = statistics.stdev(self.memory)

                # if standard deviation is 0 it causes error in the next
                # iteration (UL and LL are the same) so a small number is
                # choosen instead 
                if(self.s == 0):
                    self.s = np.nextafter(0, 1)

                self.LL = self.mean - self.X*self.s
                self.UL = self.mean + self.X*self.s
        # Adjust mean and stdev for all data till this point
        elif(self.count > 1):
            # Actual welfords algorithm formulas
            new_mean = self.mean + (value - self.mean)*1./self.count
            new_s = self.s + (value - self.mean)* \
                        (value - new_mean)
            self.mean = new_mean
            self.s = new_s

            # if standard deviation is 0 it causes error in the next
            # iteration (UL and LL are the same) so a small number is
            # choosen instead 
            if(self.s == 0):
                self.s = np.nextafter(0, 1)

            self.LL = self.mean - self.X*(math.sqrt(self.s/self.count))
            self.UL = self.mean + self.X*(math.sqrt(self.s/self.count))


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


class IsolationForest(AnomalyDetectionAbstract):
    name: str = "Isolation forest"
    
    # method specific
    N: int
    max_samples: int
    model_name: str
    memory: List[float]
    isolation_score: float
    warning_stages: List[float]

    # retrain information
    samples_from_retrain: int
    retrain_interval: int
    samples_for_retrain: int
    retrain_file: str
    trained: bool
    memory_dataframe: DataFrame

    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(conf)

    def configure(self, conf: Dict[Any, Any] = None,
                  configuration_location: str = None,
                  algorithm_indx: int = None) -> None:
        super().configure(conf, configuration_location=configuration_location,
                          algorithm_indx=algorithm_indx)

        # Train configuration
        self.max_features = conf["train_conf"]["max_features"]
        self.model_name = conf["train_conf"]["model_name"]
        self.max_samples = conf["train_conf"]["max_samples"]
        self.input_vector_size = conf["input_vector_size"]

        # Retrain configuration
        if("retrain_interval" in conf):
            self.retrain_interval = conf["retrain_interval"]
            self.retrain_file = conf["retrain_file"]
            self.samples_from_retrain = 0
            if("samples_for_retrain" in conf):
                self.samples_for_retrain = conf["samples_for_retrain"]
            else:
                self.samples_for_retrain = None

            # Retrain memory initialization
            if("train_data" in conf):
                self.memory_dataframe = pd.read_csv(conf["train_data"],
                                                    skiprows=1,
                                                    delimiter=",")
                if(self.samples_for_retrain is not None):
                    self.memory_dataframe = self.memory_dataframe.iloc[-self.samples_for_retrain:]
            else:
                columns = ["timestamp"]
                for i in range(self.input_vector_size):
                    columns.append(str(i))
                self.memory_dataframe = pd.DataFrame(columns=columns)
        else:
            self.retrain_interval = None
            self.samples_for_retrain = None
            self.memory_dataframe = None

        # Initialize model
        self.trained = False
        if("load_model_from" in conf):
            self.model = self.load_model(conf["load_model_from"])
        elif("train_data" in conf):
            self.train_model(conf["train_data"])
        elif(self.retrain_interval is not None):
            self.model = sklearn.ensemble.IsolationForest(
                max_samples=self.max_samples, max_features=self.max_features)
        else:
            raise Exception("The configuration must specify either \
                            load_model_from, train_data or train_interval")

    def message_insert(self, message_value: Dict[Any, Any]) -> None:
        super().message_insert(message_value)

        if(self.use_cols is not None):
            value = []
            for el in range(len(message_value["ftr_vector"])):
                if(el in self.use_cols):
                    value.append(message_value["ftr_vector"][el])
        else:
            value = message_value["ftr_vector"]

        timestamp = message_value["timestamp"]
        feature_vector = super().feature_construction(value=value,timestamp=timestamp)
        # print("feature vector:", feature_vector)

        if (not feature_vector or not self.trained):
            # If this happens the memory does not contain enough samples to
            # create all additional features.
            # print("undefined")
            status = self.UNDEFINED
            status_code = self.UNDEFIEND_CODE
            # Send undefined message to output
            for output in self.outputs:
                output.send_out(timestamp=message_value['timestamp'],
                                value=value, status=status,
                                status_code=status_code,
                                algorithm=self.name)
            
            # And to visualization
            if(self.visualization is not None):
                lines = [value[0]]
                self.visualization.update(value=lines, timestamp=timestamp,
                                          status_code=2)

            # Add to normalization
            if(self.normalization is not None):
                self.normalization.add_value(value=value)

        else:
            feature_vector = np.array(feature_vector)
            # Model prediction
            # print(feature_vector)
            isolation_score = self.model.predict(feature_vector.reshape(1, -1))
            #print("isol_score: " + str(isolation_score))
            if(isolation_score == 1):
                status = self.OK
                status_code = self.OK_CODE
            elif(isolation_score == -1):
                status = "Error: outlier detected"
                status_code = -1
            else:
                status = self.UNDEFINED
                status_code = self.UNDEFIEND_CODE

            self.normalization_output_visualization(status=status,
                                                    status_code=status_code,
                                                    value=value,
                                                    timestamp=timestamp)
        self.status = status
        self.status_code = status_code

        # Add to memory for retrain and execute retrain if needed 
        if (self.retrain_interval is not None):
            # print(self.samples_from_retrain)
            # print(self.memory_dataframe.shape)
            # Add to memory

            samples_in_memory = self.memory_dataframe.shape[0]
            to_save = [timestamp] + value
            self.memory_dataframe.loc[samples_in_memory] = to_save
            if(self.samples_for_retrain is not None):
                self.memory_dataframe = self.memory_dataframe.iloc[-self.samples_for_retrain:]
            self.samples_from_retrain += 1

            # Retrain if needed (and possible)
            if(self.samples_from_retrain >= self.retrain_interval and
                (self.samples_for_retrain == self.memory_dataframe.shape[0] or
                self.samples_for_retrain is None)):
                self.samples_from_retrain = 0
                self.train_model(train_dataframe=self.memory_dataframe)

    def save_model(self, filename: str) -> None:
        with open("models/" + filename, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self, filename: str) -> None:
        with open(filename, 'rb') as f:
            clf = pickle.load(f)
        return(clf)

    def train_model(self, train_file: str = None,
                    train_dataframe: DataFrame = None) -> None:  
        if(train_dataframe is not None):
            # print("RETRAIN")
            df = train_dataframe

            path = self.retrain_file
            df.to_csv(path,index=False)

            # Change the config file so the next time the model will train from that file
            with open("configuration/" + self.configuration_location) as conf:
                whole_conf = json.load(conf)
                whole_conf["anomaly_detection_conf"][self.algorithm_indx]["train_data"] = path
            
            with open("configuration/" + self.configuration_location, "w") as conf:
                json.dump(whole_conf, conf)

            df = df.to_numpy()
        elif(train_file is not None):
            # Load data from location stored in "filename"
            
            # Changed 25.3.
            # df = pd.read_csv(train_file, skiprows=1, delimiter = ",")

            df_ = pd.read_csv(train_file, skiprows=0, delimiter = ",", usecols = (0, 1,), converters={'ftr_vector': literal_eval})
            vals = df_['ftr_vector'].values
            vals = np.array([np.array(xi) for xi in vals])
            
            timestamps = np.array(df_['timestamp'].values)
            timestamps = np.reshape(timestamps, (-1, 1))
            df = np.concatenate([timestamps,vals], axis = 1)
        else:
            raise Exception("train_file or train_dataframe must be specified.")
        
        timestamps = np.array(df[:,0])
        data = np.array(df[:,1:(1 + self.input_vector_size)])

        # Requires special feature construction so it does not mess with the
        # feature-construction memory
        features = self.training_feature_construction(data=data,
                                                      timestamps=timestamps)

        # Fit IsolationForest model to data (if there was enoug samples to
        # construct at leat one feature)
        if(len(features) > 0):
            self.model = sklearn.ensemble.IsolationForest(
                max_samples = self.max_samples,
                max_features = self.max_features
                ).fit(features)

            self.save_model(self.model_name)
            self.trained = True


class PCA(AnomalyDetectionAbstract):
    name: str = "PCA"

    N: int
    N_components: int
    N_past_data: int
    PCA_transformed: List[float]

    # Retrain information
    retrain_file: str

    isolation_forest: "Isolation_forest"

    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(conf)

    def configure(self, conf: Dict[Any, Any] = None,
                  configuration_location: str = None,
                  algorithm_indx: int = None) -> None:
        super().configure(conf, configuration_location=configuration_location,
                          algorithm_indx=algorithm_indx)

        # Train configuration
        self.max_features = conf["train_conf"]["max_features"]
        self.model_name = conf["train_conf"]["model_name"]
        self.max_samples = conf["train_conf"]["max_samples"]
        self.N_components = conf["train_conf"]["N_components"]

        # Retrain configuration
        if("retrain_interval" in conf):
            self.retrain_interval = conf["retrain_interval"]
            self.samples_from_retrain = 0
            if("samples_for_retrain" in conf):
                self.samples_for_retrain = conf["samples_for_retrain"]
            else:
                self.samples_for_retrain = None

            # Retrain memory initialization
            if("train_data" in conf):
                self.memory_dataframe = pd.read_csv(conf["train_data"],
                                                    skiprows=1,
                                                    delimiter=",", usecols = (0, 1,))
                if(self.samples_for_retrain is not None):
                    self.memory_dataframe = self.memory_dataframe.iloc[-self.samples_for_retrain:]
            else:
                columns = ["timestamp"]
                for i in range(self.input_vector_size):
                    columns.append(str(i))
                self.memory_dataframe = pd.DataFrame(columns=columns)
        else:
            self.retrain_interval = None
            self.samples_for_retrain = None
            self.memory_dataframe = None

        # Initialize model
        if("load_model_from" in conf):
            self.load_model(conf["load_model_from"])
        elif("train_data" in conf):
            self.train_model(train_file = conf["train_data"])
        else:
            raise Exception("Model or train dataset must be specified to\
                            initialize model.")

    def message_insert(self, message_value: Dict[Any, Any]) -> None:
        super().message_insert(message_value)

        if(self.use_cols is not None):
            value = []
            for el in range(len(message_value["ftr_vector"])):
                if(el in self.use_cols):
                    value.append(message_value["ftr_vector"][el])
        else:
            value = message_value["ftr_vector"]

        timestamp = message_value["timestamp"]

        feature_vector = super().feature_construction(value=value,
                                                      timestamp=timestamp)
        

        if (feature_vector == False):
            # If this happens the memory does not contain enough samples to
            # create all additional features.
            self.status = self.UNDEFINED
            self.status_code = self.UNDEFIEND_CODE
            
            # Send undefined message to output
            for output in self.outputs:
                output.send_out(timestamp=message_value['timestamp'],
                                value=None)
            
            # And to visualization
            if(self.visualization is not None):
                lines = [value[0]]
                #self.visualization.update(value=[None], timestamp=timestamp,
                #                          status_code=2)
            return
        else:
            feature_vector = np.array(feature_vector)
            # print(feature_vector)
            #Model prediction
            PCA_transformed = self.PCA.transform(feature_vector.reshape(1, -1))
            IsolationForest_transformed =  self.IsolationForest.predict(PCA_transformed.reshape(1, -1))
            if(IsolationForest_transformed == 1):
                status = self.OK
                status_code = self.OK_CODE
            elif(IsolationForest_transformed == -1):
                status = "Error: outlier detected"
                status_code = -1
            else:
                status = self.UNDEFINED
                status_code = self.UNDEFIEND_CODE

            self.normalization_output_visualization(status=status,
                                                    status_code=status_code,
                                                    value=value,
                                                    timestamp=timestamp)
        self.status = status
        self.status_code = status_code

        # Add to memory for retrain and execute retrain if needed 
        if (self.retrain_interval is not None):
            # print(self.samples_from_retrain)
            # print(self.memory_dataframe.shape)
            # Add to memory
            to_save = [timestamp] + value
            samples_in_memory = self.memory_dataframe.shape[0]
            self.memory_dataframe.loc[samples_in_memory] = to_save
            self.memory_dataframe = self.memory_dataframe.iloc[-self.samples_for_retrain:]
            self.samples_from_retrain += 1

            # Retrain if needed (and possible)
            if(self.samples_from_retrain >= self.retrain_interval and
                self.samples_for_retrain == self.memory_dataframe.shape[0]):
                self.samples_from_retrain = 0
                self.train_model(train_dataframe=self.memory_dataframe)
            return

    def save_model(self, filename):
        with open("models/" + filename + "_PCA", 'wb') as f:
            #print("Saving PCA")
            pickle.dump(self.PCA, f)

        with open("models/" + filename + "_IsolationForest", 'wb') as f:
            #print("Saving isolationForest")
            pickle.dump(self.IsolationForest, f) 

    def load_model(self, filename):
        with open(filename + "_PCA", 'rb') as f:
            clf = pickle.load(f)
        self.PCA = clf
        with open(filename + "_IsolationForest", 'rb') as f:
            clf = pickle.load(f)
        self.IsolationForest = clf

    def train_model(self, train_file: str = None, train_dataframe: DataFrame = None) -> None:  
        #print("TrainingModel")
        if(train_dataframe is None):
            df = pd.read_csv(train_file, skiprows=1, delimiter = ",")
        else:
            df = train_dataframe
        
        df = df.to_numpy()
        timestamps = np.array(df[:,0])
        data = np.array(df[:,1:(1 + self.input_vector_size)])

        # Requires special feature construction so it does not mess with the
        # feature-construction memory
        features = self.training_feature_construction(data=data,
                                                      timestamps=timestamps)

        # Fit IsolationForest model to data (if there was enoug samples to
        # construct at leat one feature)
        if(len(features) > 0):
            N_components = self.N_components

            self.PCA = sklearn.decomposition.PCA(n_components = N_components)
            self.PCA.fit(features)
            transformed = self.PCA.transform(features)

            self.IsolationForest = sklearn.ensemble.IsolationForest(
                max_samples = self.max_samples,
                max_features = self.max_features
                ).fit(transformed)

            self.save_model(self.model_name)


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


class GAN(AnomalyDetectionAbstract):
    name: str = "GAN"

    N_shifts: int
    N_latent: int
    GAN_error: List[float]

    isolation_forest: "Isolation_forest"

    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(conf)

    def configure(self, conf: Dict[Any, Any] = None,
                  configuration_location: str = None,
                  algorithm_indx: int = None) -> None:
        super().configure(conf, configuration_location=configuration_location,
                          algorithm_indx=algorithm_indx)

        # Train configuration
        self.N_shifts = conf["train_conf"]["N_shifts"]
        self.N_latent = conf["train_conf"]["N_latent"]
        self.model_name = conf["train_conf"]["model_name"]
        self.max_features = conf["train_conf"]["max_features"]
        self.max_samples = conf["train_conf"]["max_samples"]
        self.K = conf["train_conf"]["K"]

        # Retrain configuration
        if("retrain_interval" in conf):
            self.retrain_interval = conf["retrain_interval"]
            self.samples_from_retrain = 0
            if("samples_for_retrain" in conf):
                self.samples_for_retrain = conf["samples_for_retrain"]
            else:
                self.samples_for_retrain = None

            # Retrain memory initialization
            if("train_data" in conf):

                df_ = pd.read_csv(conf["train_data"], skiprows=1, delimiter = ",", usecols = (0, 1,)).values
                vals = df_[:,1]
            
                #values = np.lib.stride_tricks.sliding_window_view(values, (self.input_vector_size))
                values = [vals[x:x+self.input_vector_size] for x in range(len(vals) - self.input_vector_size + 1)]
    
                timestamps = [df_[:,0][-len(values):]]
                df = np.concatenate((np.array(timestamps).T,values), axis=1)

                self.memory_dataframe = pd.DataFrame(df, index = None)
                if(self.samples_for_retrain is not None):
                    self.memory_dataframe = self.memory_dataframe.iloc[-self.samples_for_retrain:]
            else:
                columns = ["timestamp"]
                for i in range(self.input_vector_size):
                    columns.append(str(i))
                self.memory_dataframe = pd.DataFrame(columns=columns)
        else:
            self.retrain_interval = None
            self.samples_for_retrain = None
            self.memory_dataframe = None

        # Initialize model
        if("load_model_from" in conf):
            self.load_model(conf["load_model_from"])
        elif("train_data" in conf):
            self.train_model(train_file = conf["train_data"])
        else:
            raise Exception("Model or train dataset must be specified to\
                            initialize model.")

    def message_insert(self, message_value: Dict[Any, Any]) -> None:
        super().message_insert(message_value)

        if(self.use_cols is not None):
            value = []
            for el in range(len(message_value["ftr_vector"])):
                if(el in self.use_cols):
                    value.append(message_value["ftr_vector"][el])
        else:
            value = message_value["ftr_vector"]

        timestamp = message_value["timestamp"]

        feature_vector = list((np.array(value) - self.avg)/(self.max - self.min))

        if (feature_vector == False):
            # If this happens the memory does not contain enough samples to
            # create all additional features.

            # Send undefined message to output
            for output in self.outputs:
                output.send_out(timestamp=message_value['timestamp'],
                                value=None)
            
            # And to visualization
            if(self.visualization is not None):
                lines = [value[-1]]
                #self.visualization.update(value=[None], timestamp=timestamp,
                #                          status_code=2)
            return
        else:
            feature_vector = np.array(feature_vector)
            # print(feature_vector)
            #Model prediction
            prediction = self.GAN.predict(feature_vector.reshape(1, self.N_shifts+1))[0]
            self.GAN_error = self.mse(np.array(prediction),np.array(feature_vector))


            #print("GAN error: " + str(self.GAN_error))
            #IsolationForest_transformed =  self.IsolationForest.predict(self.GAN_error.reshape(-1, 1))
            
            if(self.GAN_error < self.threshold):
                status = self.OK
                status_code = self.OK_CODE
            elif(self.GAN_error >= self.threshold):
                status = "Error: outlier detected (GAN)"
                status_code = -1
            else:
                status = self.UNDEFINED
                status_code = self.UNDEFIEND_CODE

            self.normalization_output_visualization(status=status,
                                                    status_code=status_code,
                                                    value=value,
                                                    timestamp=timestamp)
        
        self.status = status
        self.status_code = status_code

        # Add to memory for retrain and execute retrain if needed 
        if (self.retrain_interval is not None):
            # print(self.samples_from_retrain)
            #print(self.memory_dataframe[0])
            # Add to memory
            to_save = [timestamp] + value
            samples_in_memory = self.memory_dataframe.shape[0]

            self.memory_dataframe.loc[samples_in_memory] = to_save
            self.memory_dataframe = self.memory_dataframe.iloc[-self.samples_for_retrain:]
            self.samples_from_retrain += 1

            # Retrain if needed (and possible)
            if(self.samples_from_retrain >= self.retrain_interval and
                self.samples_for_retrain == self.memory_dataframe.shape[0]):
                self.samples_from_retrain = 0
                self.train_model(train_dataframe=self.memory_dataframe)
            return

    @staticmethod
    def mse(pre_GAN, post_GAN):
        #mean squared error - loss
        mse = np.sum((np.add(np.array(pre_GAN), -np.array(post_GAN))**2))
        return(mse)

    def save_model(self, filename):
        self.GAN.save("models/" + filename + "_GAN")
        #print("Saving GAN")
        

    def load_model(self, filename):
        self.GAN = keras.models.load_model(filename + "_GAN")

    def train_model(self, train_file: str = None, train_dataframe: DataFrame = None) -> None:  
        #print("TrainingModel")
        if(train_dataframe is None):
            df_ = pd.read_csv(train_file, skiprows=0, delimiter = ",", usecols = (0, 1,), converters={'ftr_vector': literal_eval})
            vals = df_['ftr_vector'].values
            vals = np.array([np.array(xi) for xi in vals])
            self.min = min(min(vals, key=min))
            self.max = max(max(vals, key=max))
            self.avg = (self.min + self.max)/2

            values = (np.array(vals) - self.avg)/(self.max - self.min)
            
            #values = np.lib.stride_tricks.sliding_window_view(values, (self.input_vector_size))
            #values = [vals[x:x+self.input_vector_size] for x in range(len(vals) - self.input_vector_size + 1)]
            timestamps = np.array(df_['timestamp'].values)
            timestamps = np.reshape(timestamps, (-1, 1))
            df = np.concatenate([timestamps,values], axis = 1)
        else:
            df = train_dataframe
        
        #df = df.to_numpy()
        timestamps = np.array(df[:,0])
        data = np.array(df[:,1:(1 + self.input_vector_size)])

        # Requires special feature construction so it does not mess with the
        # feature-construction memory
        features = self.training_feature_construction(data=data,
                                                      timestamps=timestamps)


        # Fit IsolationForest model to data (if there was enoug samples to
        # construct at leat one feature)
        if(len(features) > 0):

            original_dim = np.prod(np.array(features).shape [1:]) # dimenzija vhodnih podatkov
            hidden_dim = 10 # skriti sloj z 64 node -i
            latent_dim = self.N_latent # 2D latentni prostor
            inputs = keras.Input(shape =(original_dim ,))
            h1 = keras.layers.Dense(hidden_dim, activation ='linear')(inputs)
            h2 = keras.layers.Dense(hidden_dim, activation ='tanh')(h1)
            h3 = keras.layers.Dense(hidden_dim, activation ='tanh')(h2)

            h4 = keras.layers.Dense(latent_dim, activation ='tanh')(h3)

            encoder = keras.Model(inputs, outputs = [h4, h4], name = 'encoder')
            latent_inputs = keras.Input(shape =latent_dim, name ='z_sampling')

            x1 = keras.layers.Dense(hidden_dim , activation ='tanh')(latent_inputs)
            x2 = keras.layers.Dense(hidden_dim , activation ='relu')(x1)
            x3 = keras.layers.Dense(hidden_dim , activation ='relu')(x2)

            outputs = keras.layers.Dense(original_dim , activation ='linear')(x3)
            decoder = keras.Model(latent_inputs, outputs, name ='decoder')

            outputs = decoder(encoder(inputs)[0])
            self.GAN = keras.Model(inputs, outputs, name ='vae')
            mse = tf.keras.losses.MeanSquaredError()

            GAN_loss = mse(inputs, outputs)
            self.GAN.add_loss(GAN_loss)
            self.GAN.compile(optimizer =tf.keras.optimizers.Adam(lr = 0.001, beta_1 = 0.95))
            features = np.array(features)
            self.GAN.fit(features,features, epochs =100, batch_size = 100, validation_data = None, verbose = 2)
            
            predictions = self.GAN.predict(features.reshape(len(features), self.N_shifts+1))
            
            GAN_transformed = [mse(np.array(features[i]), predictions[i]) for i in range(len(features))]
            self.threshold = self.K * max(GAN_transformed)

            self.save_model(self.model_name)


class LinearFit(AnomalyDetectionAbstract):
    name: str = "LinearFit"
    slope: float
    average: float
    memory: list
    UL: float
    LL: float
    N: int
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
        self.timestamps = []
        self.count = 0


        if ('N' in conf):
            self.N = conf['N']
            self.memory = []
        else:
            self.N = None

    def message_insert(self, message_value: Dict[Any, Any]):
        super().message_insert(message_value)

        value = message_value["ftr_vector"]
        value = value[0]
        timestamp = message_value["timestamp"]

        self.memory.append(value)
        self.timestamps.append(message_value['timestamp'])
        if (len(self.memory) > self.N):
            self.memory = self.memory[-self.N:]
            self.timestamps = self.timestamps[-self.N:]

        slope = None
        #Calculate the fit coefficients
        if(self.count < 1):
            pass
        else:
            x = np.array(range(len(self.memory)))
            y = self.memory
            slope, average = np.polyfit(x, y, deg = 1)
        

        #Normalize the slope to the range LL - UL
        status = self.UNDEFINED
        status_code = self.UNDEFIEND_CODE
        value_normalized = 0
        if(slope is not None):
            value_normalized = 2*(slope - (self.UL + self.LL)/2) / \
                (self.UL - self.LL)
            print(value_normalized)

            status = self.OK
            status_code = self.OK_CODE

            #Perform error and warning checks
            if(value_normalized > 1):
                status = "Error: slope above upper limit"
                status_code = -1
            elif(value_normalized < -1):
                status = "Error: slope below lower limit"
                status_code = -1
            else:
                for stage in range(len(self.warning_stages)):
                    if(value_normalized > self.warning_stages[stage]):
                        status = "Warning" + str(stage) + \
                            ": slope close to upper limit."
                        status_code = 0
                    elif(value_normalized < -self.warning_stages[stage]):
                        status = "Warning" + str(stage) + \
                            ": slope close to lower limit."
                        status_code = 0
                    else:
                        break


        self.status = status
        self.status_code = status_code

        #Send to outputs
        for output in self.outputs:
            output.send_out(timestamp=message_value["timestamp"],
                            algorithm=self.name, status=status,
                            value=message_value['ftr_vector'],
                            status_code=status_code)

        #send to visualization
        lines = [value_normalized]
        timestamp = self.timestamps[-1]
        if(self.visualization is not None):
            self.visualization.update(value=lines, timestamp=message_value["timestamp"],
            status_code = status_code)
        self.count += 1
