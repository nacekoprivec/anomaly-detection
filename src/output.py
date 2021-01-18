from abc import abstractmethod
from abc import ABC
import json
import csv
from json import dumps
import os
from typing import Any, Dict
from kafka import KafkaProducer
#from kafka.admin import KafkaAdminClient, NewTopic


class OutputAbstract(ABC):

    def __init__(self) -> None:
        pass

    @abstractmethod
    def configure(self, conf: Dict[Any, Any]) -> None:
        pass

    @abstractmethod
    def send_out(self, value: Any, status: str, timestamp: Any,
                 status_code: int = None) -> None:
        pass


class TerminalOutput(OutputAbstract):

    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(conf=conf)

    def configure(self, conf: Dict[Any, Any] = None) -> None:
        # Nothing to configure
        pass

    def send_out(self,  value: Any, status: str = "",
                 timestamp: Any = 0, status_code: int = None,
                 algorithm: str = "Unknown") -> None:
        o = status + "(value: " + str(value) + ")"
        print(o)


class FileOutput(OutputAbstract):
    file_name: str
    file_path: str
    mode: str

    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(conf=conf)

    def configure(self, conf: Dict[Any, Any] = None) -> None:
        self.file_name = conf["file_name"]
        self.mode = conf["mode"]
        self.file_path = "log/" + self.file_name

        # make log folder if one does not exist
        dir = "./log"
        if not os.path.isdir(dir):
            os.makedirs(dir)

        # If mode is write clear the file
        if(self.mode == "w"):
            if(self.file_name[-4:] == "json"):
                with open(self.file_path, "w") as f:
                    d = {
                        "data": []
                    }
                    json.dump(d, f)
            elif(self.file_name[-3:] == "txt"):
                open(self.file_path, "w").close()
            elif(self.file_name[-3:] == "csv"):
                with open(self.file_path, "w", newline="") as csvfile:
                    fieldnames = ["timestamp", "status", "status_code", "value", "algorithm"]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()

    def send_out(self,  value: Any = None, status: str = "",
                 timestamp: Any = None, status_code: int = None,
                 algorithm: str = "Unknown") -> None:
        if(self.file_name[-4:] == "json"):
            self.write_JSON(value=value, status=status,
                            timestamp=timestamp, status_code=status_code,
                            algorithm=algorithm)
        elif(self.file_name[-3:] == "txt"):
            self.write_txt(value=value, status=status,
                           timestamp=timestamp, status_code=status_code,
                           algorithm=algorithm)
        elif(self.file_name[-3:] == "csv"):
            self.write_csv(value=value, status=status,
                           timestamp=timestamp, status_code=status_code,
                           algorithm=algorithm)
        else:
            print("Output file type not supported.")

    def write_JSON(self,  value: Any, timestamp: Any,
                   status: str = "", status_code: int = None,
                   algorithm: str = "Unknown") -> None:
        # Construct the object to write
        to_write = {"algorithm": algorithm}
        if (value is not None):
            to_write["value"] = value
        if (status != ""):
            to_write["status"] = status
        if (timestamp is not None):
            to_write["timestamp"] = timestamp
        if (status_code is not None):
            to_write["status_code"] = status_code

        with open(self.file_path) as json_file:
            data = json.load(json_file)
            temp = data["data"]
            temp.append(to_write)
        with open(self.file_path, "w") as f:
            json.dump(data, f)

    def write_txt(self,  value: Any, status: str = "",
                  timestamp: Any = 0, status_code: int = None,
                  algorithm: str = "Unknown") -> None:
        with open(self.file_path, "a") as txt_file:
            o = timestamp + " " + status + "(value: " + str(value) + ")\n"
            txt_file.write(o)

    def write_csv(self,  value: Any, status: str = "",
                  timestamp: Any = 0, status_code: int = None,
                  algorithm: str = "Unknown") -> None:
        # Construct the object to write
        to_write = {"algorithm": algorithm}
        to_write["value"] = value
        to_write["status"] = status
        to_write["timestamp"] = timestamp
        to_write["status_code"] = status_code

        with open(self.file_path, 'a', newline='') as csv_file:
            fieldnames = ["timestamp", "status", "status_code", "value", "algorithm"]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writerow(to_write)

class KafkaOutput(OutputAbstract):

    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        # print(conf)
        if(conf is not None):
            self.configure(conf=conf)

    def configure(self, conf: Dict[Any, Any]) -> None:
        self.output_topic = conf['output_topic']
        self.output_metric = conf['output_metric']

        self.producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                         value_serializer=lambda x: 
                         dumps(x).encode('utf-8'))

    def send_out(self, value: Any = None, status: str = "",
                 timestamp: Any = None, status_code: int = None,
                 algorithm: str = "Unknown") -> None:
        # Build a object to be sent out
        to_write = {"algorithm": algorithm}
        if (value is not None):
            to_write["value"] = value
        if (status != ""):
            to_write["status"] = status
        if (timestamp is not None):
            to_write["timestamp"] = timestamp
        if (status_code is not None):
            to_write["status_code"] = status_code

        self.producer.send(self.output_topic, value=status_code)