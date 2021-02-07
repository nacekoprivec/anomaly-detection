from abc import abstractmethod, ABC
from typing import Any, Dict, List, Union
import numpy as np
import statistics
import sys
import math
from statistics import mean
from datetime import datetime
import pickle
import sklearn.ensemble
from scipy import signal
import pandas as pd

sys.path.insert(0,'./src')
from output import OutputAbstract, TerminalOutput, FileOutput, KafkaOutput
from visualization import VisualizationAbstract, GraphVisualization,\
    HistogramVisualization, StatusPointsVisualization



class AnomalyDetectionAbstract(ABC):
    memory_size: int
    memory: List[List[Any]]
    averages: List[List[int]]
    shifts: List[List[int]]
    time_features: List[str]
    name: str

    input_vector_size: int
    outputs: List["OutputAbstract"]

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
        assert len(message_value['test_value']) == self.input_vector_size, \
            "Given test value does not satisfy input vector size"

    @abstractmethod
    def configure(self, conf: Dict[Any, Any]) -> None:
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
    
    def feature_construction(self, value: List[Any], 
                             timestamp: str) -> Union[None, bool]:

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

        return time_features


class BorderCheck(AnomalyDetectionAbstract):
    """ works with 1D data and checks if the value is above, below or close
    to guven upper and lower limits 
    """
    UL: float
    LL: float
    warning_stages: List[float]
    outputs: List["OutputAbstract"]
    visualization: List["VisualizationAbstract"]
    name: str = "Border check"

    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(conf)


    def configure(self, conf: Dict[Any, Any] = None) -> None:
        super().configure(conf)
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

        # If configuration is specified configure it
        if ("visualization" in conf):
            # print(conf["visualization"])
            self.visualization = eval(conf["visualization"])
            visualization_configurations = conf["visualization_conf"]
            self.visualization.configure(visualization_configurations)
        else:
            self.visualization = None

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
    UL: float
    LL: float
    N: int
    count: int
    memory: List[float]
    X: float
    mean: float
    s: float
    warning_stages: List[float]
    visualization: List["VisualizationAbstract"]
    outputs: List["OutputAbstract"]
    name: str = "Welford"

    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(conf)

    def configure(self, conf: Dict[Any, Any] = None) -> None:
        super().configure(conf)
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

        self.outputs = [eval(o) for o in conf["output"]]
        # configure all outputs
        output_configurations = conf["output_conf"]
        for o in range(len(self.outputs)):
            self.outputs[o].configure(output_configurations[o])

        if ("visualization" in conf):
            self.visualization = eval(conf["visualization"])
            visualization_configurations = conf["visualization_conf"]
            self.visualization.configure(visualization_configurations)
        else:
            self.visualization = None

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
    UL: float
    LL: float
    N: int
    smoothing: float
    EMA: List[float]
    sigma: List[float]
    numbers: List[float]
    timestamp: List[Any]
    visualization: List["VisualizationAbstract"]
    outputs: List["OutputAbstract"]
    name: str = "EMA"

    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(conf)

    def configure(self, conf: Dict[Any, Any] = None) -> None:
        super().configure(conf)
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

        self.outputs = [eval(o) for o in conf["output"]]
        # configure all outputs
        output_configurations = conf["output_conf"]
        for o in range(len(self.outputs)):
            self.outputs[o].configure(output_configurations[o])

        if ("visualization" in conf):
            self.visualization = eval(conf["visualization"])
            visualization_configurations = conf["visualization_conf"]
            self.visualization.configure(visualization_configurations)
        else:
            self.visualization = None

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
        status = "OK"
        status_code = 1

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
        mean = self.EMA[-1]
        sigma = self.sigma[-1]
        lines = [self.numbers[-1], mean, mean+sigma, mean-sigma]
        timestamp = self.timestamps[-1]
        if(self.visualization is not None):
            self.visualization.update(value=lines, timestamp=message_value["timestamp"],
            status_code = status_code)


class IsolationForest(AnomalyDetectionAbstract):
    N: int
    memory: List[float]
    isolation_score: float
    warning_stages: List[float]
    visualization: List["VisualizationAbstract"]
    outputs: List["OutputAbstract"]
    name: str = "Isolation forest"

    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(conf)

    def configure(self, conf: Dict[Any, Any] = None) -> None:
        super().configure(conf)

        # Outputs and visualization initialization and configuration
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


        if("load_model_from" in conf):
            self.model = self.load_model(conf["load_model_from"])
        elif("train_data" in conf):
            self.train_model(conf)

    def message_insert(self, message_value: Dict[Any, Any]) -> None:
        super().message_insert(message_value)

        value = message_value["test_value"]
        timestamp = message_value["timestamp"]

        feature_vector = super().feature_construction(value=value,
                                                      timestamp=timestamp)

        if (feature_vector == False):
            # If this happens the memory does not contain enough samples to
            # create all additional features.
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
            return
        else:
            feature_vector = np.array(feature_vector)

            #Model prediction
            isolation_score = self.model.predict(feature_vector.reshape(1, -1))
            print("isol_score: " + str(isolation_score))
            if(isolation_score == 1):
                status = self.OK
                status_code = self.OK_CODE
            elif(isolation_score == -1):
                status = "Error: outlier detected"
                status_code = -1
            else:
                status = self.UNDEFINED
                status_code = self.UNDEFIEND_CODE
            


            # TODO: fix sendout calls, status and status_codes
            for output in self.outputs:
                output.send_out(timestamp=timestamp, status=status,
                            value=message_value["test_value"],
                            status_code=status_code, algorithm=self.name)
            

            if(self.visualization is not None):
                lines = [value]
                self.visualization.update(value=lines, timestamp=timestamp,
                                      status_code=status_code)


    def save_model(self, filename):
        with open("models/" + filename, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            clf = pickle.load(f)
        return(clf)

    def train_model(self, conf):
        #load data from location stored in "filename"
        df = pd.read_csv(conf["train_data"], skiprows=1, delimiter = ",")
        df = df.to_numpy()
        timestamps = np.array(df[:,0])
        data = np.array(df[:,1])
        features = []
        N = conf["train_conf"]["max_features"]
        for i in range(len(data)):
            value = [data[i]]
            timestamp = timestamps[i]
            feature_vector = super().feature_construction(value=value,
                                                      timestamp=timestamp)
            if(feature_vector is not False):
                features.append(np.array(feature_vector))

        #fit IsolationForest model to data
        self.model = sklearn.ensemble.IsolationForest(
            max_samples = conf["train_conf"]["max_samples"],
            max_features = N
            ).fit(features)

        self.save_model(conf["train_conf"]["model_name"])


class PCA(AnomalyDetectionAbstract):
    N: int
    N_components: int
    N_past_data: int
    PCA_transformed: List[float]
    visualization: List["VisualizationAbstract"]
    outputs: List["OutputAbstract"]
    name: str = "PCA"

    isolation_forest: "Isolation_forest"

    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(conf)

    def configure(self, conf: Dict[Any, Any] = None) -> None:
        super().configure(conf)

        # Outputs and visualization initialization and configuration
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

        if("load_model_from" in conf):
            self.model = self.load_model(conf["load_model_from"])
        elif("train_data" in conf):
            self.train_model(conf)

    def message_insert(self, message_value: Dict[Any, Any]) -> None:
        super().message_insert(message_value)

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
            # TODO: probably  feature_vector = np.array(feature vector)?
            feature_vector = np.array(self.shift_construction())

            #Model prediction
            PCA_transformed = self.model.transform(feature_vector.reshape(1, -1))

            # TODO: Call message insert isolation forest
            return

    def save_model(self, filename):
        with open("models/" + filename, 'wb') as f:
            pickle.dump(self.model, f)
        
        self.isolation_forest.save_model()

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            clf = pickle.load(f)
        return(clf)

    def train_model(self, conf):
        # load data from location stored in "train_data"
        data = np.loadtxt(conf["train_data"], skiprows=1, delimiter = ",", usecols=(1,))
        # TODO featuere construction
        features = []
        N_components = conf["train_conf"]["N_components"]
        N_past_data = conf["train_conf"]["N_past_data"]
        for i in range(N_past_data, len(data)):
            features.append(np.array(data[i-N_past_data:i]))

        #fit IsolationForest model to data
        self.model = sklearn.decomposition.PCA(n_components = N_components)
        self.model.fit(features)
        self.save_model(conf["train_conf"]["model_name"])


class Filtering(AnomalyDetectionAbstract):
    UL: float
    LL: float
    value_normalized: float
    filtered: List[float]
    result: float
    visualization: List["VisualizationAbstract"]
    outputs: List["OutputAbstract"]
    name: str = "Filtering"

    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(conf)

    def configure(self, conf: Dict[Any, Any] = None) -> None:
        super().configure(conf)
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

        self.outputs = [eval(o) for o in conf["output"]]
        # configure all outputs
        output_configurations = conf["output_conf"]
        for o in range(len(self.outputs)):
            self.outputs[o].configure(output_configurations[o])

        if ("visualization" in conf):
            self.visualization = eval(conf["visualization"])
            visualization_configurations = conf["visualization_conf"]
            self.visualization.configure(visualization_configurations)
        else:
            self.visualization = None

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