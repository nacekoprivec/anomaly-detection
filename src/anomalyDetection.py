from abc import abstractmethod, ABC
from typing import Any, Dict, List, Union
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
import pandas as pd

sys.path.insert(0,'./src')
from output import OutputAbstract, TerminalOutput, FileOutput, KafkaOutput
from visualization import VisualizationAbstract, GraphVisualization,\
    HistogramVisualization, StatusPointsVisualization
from normalization import NormalizationAbstract, LastNAverage,\
    PeriodicLastNAverage


class AnomalyDetectionAbstract(ABC):
    configuration_location: str

    memory_size: int
    memory: List[List[Any]]
    averages: List[List[int]]
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
        print(message_value['test_value'])
        # print(message_value['test_value'])
        assert len(message_value['test_value']) == self.input_vector_size, \
            "Given test value does not satisfy input vector size"

    @abstractmethod
    def configure(self, conf: Dict[Any, Any],
                  configuration_location: str = None) -> None:
        self.configuration_location = configuration_location
        
        # FEATURE CONSTRUCTION CONFIGURATION
        self.input_vector_size = conf["input_vector_size"]
        
        if("averages" in conf):
            self.averages = conf["averages"]
        else:
            self.averages = []

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
            max_shift = max(map(max, self.shifts))+1

        if (len(self.averages) == 0):
            max_average = 0
        else:
            max_average = max(map(max, self.averages))

        self.memory_size = max(max_shift, max_average)

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
                values = np_memory[-interval:, feature_index]
                averages.append(mean(values))

        return averages

    def shift_construction(self) -> None:
        shifts = []

        # Loop through all features
        for feature_index in range(len(self.shifts)):
            # Loop through all shift values
            for look_back in self.shifts[feature_index]:
                shifts.append(self.memory[self.memory_size-(look_back+1)][feature_index])

        return shifts

    def time_features_construction(self, timestamp: str) -> None:
        time_features = []
        
        if(len(str(timestamp)) == 19):
            dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
        else:
            dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S.%f')

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
        
        if("Braila_day_night" in self.time_features):
            start = time(23, 0, 0)
            end = time(4, 45, 0)
            now = time(dt.hour, dt.minute, dt.second)
            day_night = int(start <= now or now <= end)
            print(day_night)
            print(start)
            print(now)
            print(end)
            time_features.append(day_night)


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
                  configuration_location: str = None) -> None:
        super().configure(conf, configuration_location=configuration_location)
        self.LL = conf["LL"]
        self.UL = conf["UL"]

        self.warning_stages = conf["warning_stages"]
        self.warning_stages.sort()

    def message_insert(self, message_value: Dict[Any, Any]) -> None:
        super().message_insert(message_value)
        value = message_value["test_value"]
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

        for output in self.outputs:
            output.send_out(timestamp=timestamp, status=status, value=message_value["test_value"],
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
                  configuration_location: str = None) -> None:
        super().configure(conf, configuration_location=configuration_location)
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
        value = message_value["test_value"]
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

        # Outputs and visualizations
        for output in self.outputs:
            output.send_out(timestamp=timestamp, status=status,
                            value=message_value["test_value"],
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
                  configuration_location: str = None) -> None:
        super().configure(conf, configuration_location=configuration_location)
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
        self.numbers.append(message_value['test_value'][0])
        self.timestamps.append(message_value['timestamp'])
        
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

        for output in self.outputs:
            output.send_out(timestamp=message_value["timestamp"],
                            algorithm=self.name, status=status,
                            value=message_value['test_value'],
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
    trained: bool
    memory_dataframe: DataFrame

    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(conf)

    def configure(self, conf: Dict[Any, Any] = None,
                  configuration_location: str = None) -> None:
        super().configure(conf, configuration_location=configuration_location)

        # Train configuration
        self.N = conf["train_conf"]["max_features"]
        self.model_name = conf["train_conf"]["model_name"]
        self.max_samples = conf["train_conf"]["max_samples"]
        self.input_vector_size = conf["input_vector_size"]

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
                max_samples=self.max_samples, max_features=self.N)
        else:
            raise Exception("The configuration must specify either \
                            load_model_from, train_data or train_interval")

    def message_insert(self, message_value: Dict[Any, Any]) -> None:
        super().message_insert(message_value)

        if(self.use_cols is not None):
            value = []
            for el in range(len(message_value["test_value"])):
                if(el in self.use_cols):
                    value.append(message_value["test_value"][el])
        else:
            value = message_value["test_value"]

        timestamp = message_value["timestamp"]

        feature_vector = super().feature_construction(value=value,
                                                      timestamp=timestamp)

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
            #Model prediction
            print(feature_vector)
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

        # Add to memory for retrain and execute retrain if needed 
        if (self.retrain_interval is not None):
            # print(self.samples_from_retrain)
            # print(self.memory_dataframe.shape)
            # Add to memory
            to_save = [timestamp] + value
            samples_in_memory = self.memory_dataframe.shape[0]
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

            # Save dataframe to csv file
            name = "IsolationForest_last_" + str(self.samples_for_retrain)\
                  + "_samples.csv"
            dir = "./data"
            if not os.path.isdir(dir):
                os.makedirs(dir)
            path = dir + "/" + name
            df.to_csv(path,index=False)

            # Change the config file so the next time the model will train from that file
            with open("configuration/" + self.configuration_location) as conf:
                whole_conf = json.load(conf)
                whole_conf["anomaly_detection_conf"]["train_data"] = path
            
            with open("configuration/" + self.configuration_location, "w") as conf:
                json.dump(whole_conf, conf)

        elif(train_file is not None):
            # Load data from location stored in "filename"
            df = pd.read_csv(train_file, skiprows=1, delimiter = ",")
        else:
            raise Exception("train_file or train_dataframe must be specified.")
        
        df = df.to_numpy()
        timestamps = np.array(df[:,0])
        data = np.array(df[:,1:(1 + self.input_vector_size)])

        # Aditional feature construction
        features = []
        for i in range(len(data)):
            value = data[i].tolist()
            timestamp = timestamps[i]
            feature_vector = super().feature_construction(value=value,
                                                      timestamp=timestamp)

            if(feature_vector is not False):
                features.append(np.array(feature_vector))
        # Fit IsolationForest model to data
        self.model = sklearn.ensemble.IsolationForest(
            max_samples = self.max_samples,
            max_features = self.N
            ).fit(features)

        self.save_model(self.model_name)
        self.trained = True


class PCA(AnomalyDetectionAbstract):
    name: str = "PCA"

    N: int
    N_components: int
    N_past_data: int
    PCA_transformed: List[float]

    isolation_forest: "Isolation_forest"

    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(conf)

    def configure(self, conf: Dict[Any, Any] = None,
                  configuration_location: str = None) -> None:
        super().configure(conf, configuration_location=configuration_location)

        # Train configuration
        self.N = conf["train_conf"]["max_features"]
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
            for el in range(len(message_value["test_value"])):
                if(el in self.use_cols):
                    value.append(message_value["test_value"][el])
        else:
            value = message_value["test_value"]

        timestamp = message_value["timestamp"]

        feature_vector = super().feature_construction(value=value,
                                                      timestamp=timestamp)

        if (feature_vector == False):
            # If this happens the memory does not contain enough samples to
            # create all additional features.

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
            print(feature_vector)
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
        # ("here")
        with open("models/" + filename + "_PCA", 'wb') as f:
            print("Saving PCA")
            pickle.dump(self.PCA, f)

        with open("models/" + filename + "_IsolationForest", 'wb') as f:
            print("Saving isolationForest")
            pickle.dump(self.IsolationForest, f)
        
        

    def load_model(self, filename):
        with open(filename + "_PCA", 'rb') as f:
            clf = pickle.load(f)
        self.PCA = clf
        with open(filename + "_IsolationForest", 'rb') as f:
            clf = pickle.load(f)
        self.IsolationForest = clf

    def train_model(self, train_file: str = None, train_dataframe: DataFrame = None) -> None:  
        # print("TrainingModel")
        if(train_dataframe is None):
            df = pd.read_csv(train_file, skiprows=1, delimiter = ",")
        else:
            df = train_dataframe
        
        df = df.to_numpy()
        timestamps = np.array(df[:,0])
        data = np.array(df[:,1:(1 + self.input_vector_size)])

        # Aditional feature construction
        features = []
        for i in range(len(data)):
            value = data[i].tolist()
            timestamp = timestamps[i]
            feature_vector = super().feature_construction(value=value,
                                                      timestamp=timestamp)

            if(feature_vector is not False):
                features.append(np.array(feature_vector))
        # Fit IsolationForest model to data

        N_components = self.N_components

        self.PCA = sklearn.decomposition.PCA(n_components = N_components)
        self.PCA.fit(features)
        transformed = self.PCA.transform(features)

        self.IsolationForest = sklearn.ensemble.IsolationForest(
            max_samples = self.max_samples,
            max_features = self.N
            ).fit(transformed)

        self.save_model(self.model_name)


class Filtering(AnomalyDetectionAbstract):
    name: str = "Filtering"

    UL: float
    LL: float
    value_normalized: float
    filtered: List[float]
    result: float

    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(conf)

    def configure(self, conf: Dict[Any, Any] = None,
                  configuration_location: str = None) -> None:
        super().configure(conf, configuration_location=configuration_location)
        self.mode = conf["mode"]
        self.LL = conf["LL"]
        self.UL = conf["UL"]
        self.warning_stages = conf["warning_stages"]
        self.warning_stages.sort()
        self.filtered = []
        self.numbers = []
        self.timestamps = []

        #Initalize the digital filter
        self.b, self.a = signal.butter(conf["filter_order"], conf["cutoff_frequency"])
        self.z = signal.lfilter_zi(self.b, self.a)

    def message_insert(self, message_value: Dict[Any, Any]):
        super().message_insert(message_value)

        self.last_measurement = message_value['test_value'][0]

        #Update filter output after last measurement
        filtered_value, self.z = signal.lfilter(self.b, self.a, x = [self.last_measurement], zi = self.z)

        self.filtered.append(filtered_value[0])
        
        #Normalize filter outputs
        value_normalized = 2*(self.filtered[-1] - (self.UL + self.LL)/2) / \
            (self.UL - self.LL)
        status = "OK"
        status_code = 1

        #Perform error and warning checks
        if(self.mode == 1):
            deviation = (self.last_measurement - self.filtered[-1])/(self.UL - self.LL)
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

        if(self.mode == 0):
            result = self.filtered[-1]
        else:
            result = self.last_measurement - self.filtered[-1]

        for output in self.outputs:
            output.send_out(timestamp=message_value["timestamp"],
                            status=status, value=message_value['test_value'][0], 
                            status_code=status_code, algorithm=self.name)

        
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
                  configuration_location: str = None) -> None:
        super().configure(conf, configuration_location=configuration_location)
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
        value = message_value["test_value"]
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
        
