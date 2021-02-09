from abc import ABC, abstractmethod
import csv
import json
import sys

from typing import Any, Dict, List

from src.anomalyDetection import AnomalyDetectionAbstract, EMA, BorderCheck,\
        IsolationForest, Welford, Filtering, PCA

from kafka import KafkaConsumer, TopicPartition
from json import loads
import matplotlib.pyplot as plt
from time import sleep
import numpy as np


class ConsumerAbstract(ABC):
    anomaly: "AnomalyDetectionAbstract"
    last_message: Any

    def __init__(self) -> None:
        pass

    @abstractmethod
    def configure(self, con: Dict[Any, Any],
                  configuration_location: str) -> None:
        pass

    @abstractmethod
    def read(self) -> None:
        while(self._stopping_condition_):
            self._read_next_()
            self.anomaly.check(self.last_message)


class ConsumerKafka(ConsumerAbstract):
    consumer: KafkaConsumer

    def __init__(self, conf: Dict[Any, Any] = None,
                 configuration_location: str = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(con=conf)
        elif(configuration_location is not None):
            # Read config file
            with open("configuration/" + configuration_location) as data_file:
                conf = json.load(data_file)
            self.configure(con=conf)
        else:
            print("No configuration was given")

    def configure(self, con: Dict[Any, Any] = None) -> None:
        if(con is None):
            print("No configuration was given")
            return            

        self.topics = con['topics']
        self.consumer = KafkaConsumer(
                        bootstrap_servers=con['bootstrap_servers'],
                        auto_offset_reset=con['auto_offset_reset'],
                        enable_auto_commit=con['enable_auto_commit'],
                        group_id=con['group_id'],
                        value_deserializer=eval(con['value_deserializer']))
        self.consumer.subscribe(self.topics)
        self.anomaly = eval(con["anomaly_detection_alg"])
        anomaly_configuration = con["anomaly_detection_conf"]
        self.anomaly.configure(anomaly_configuration)

    def read(self) -> None:
        for message in self.consumer:
            value = message.value
            self.anomaly.message_insert(value)


class ConsumerFile(ConsumerAbstract):
    file_name: str
    file_path: str

    def __init__(self, conf: Dict[Any, Any] = None,
                 configuration_location: str = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(con=conf)
        elif(configuration_location is not None):
            # Read config file
            with open("configuration/" + configuration_location) as data_file:
                conf = json.load(data_file)
            self.configure(con=conf)
        else:
            print("No configuration was given")

    def configure(self, con: Dict[Any, Any] = None) -> None:
        self.file_name = con["file_name"]
        self.file_path = "./data/consumer/" + self.file_name

        self.anomaly = eval(con["anomaly_detection_alg"])
        anomaly_configuration = con["anomaly_detection_conf"]
        self.anomaly.configure(anomaly_configuration)

    def read(self) -> None:
        if(self.file_name[-4:] == "json"):
            self.read_JSON()
        elif(self.file_name[-3:] == "csv"):
            self.read_csv()
        else:
            print("Consumer file type not supported.")
            sys.exit(1)

    def read_JSON(self):
        with open(self.file_path) as json_file:
            data = json.load(json_file)
            tab = data["data"]
        for d in tab:
            self.anomaly.message_insert(d)

    def read_csv(self):
        with open(self.file_path, 'r') as read_obj:
            csv_reader = csv.reader(read_obj)

            header = next(csv_reader)

            try:
                timestamp_index = header.index("timestamp")
            except ValueError:
                timestamp_index = None
            other_indicies = [i for i, x in enumerate(header) if (x != "timestamp")]

            # Iterate over each row in the csv using reader object
            for row in csv_reader:
                d = {}
                if(timestamp_index is not None):
                    timestamp = row[timestamp_index]
                    try:
                        timestamp = float(timestamp)
                    except ValueError:
                        pass
                    d["timestamp"] = timestamp
                test_value = [float(row[i]) for i in other_indicies]

                d["test_value"] = test_value

                self.anomaly.message_insert(d)


class ConsumerFileKafka(ConsumerKafka, ConsumerFile):
    file_name: str
    file_path: str

    def __init__(self, conf: Dict[Any, Any] = None,
                 configuration_location: str = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(con=conf)
        elif(configuration_location is not None):
            # Read config file
            with open("configuration/" + configuration_location) as data_file:
                conf = json.load(data_file)
            self.configure(con=conf)
        else:
            print("No configuration was given")

    def configure(self, con: Dict[Any, Any] = None) -> None:
        # File configuration
        self.file_name = con["file_name"]
        self.file_path = "./data/consumer/" + self.file_name

        # Kafka configuration
        self.topics = con['topics']
        self.consumer = KafkaConsumer(
                        bootstrap_servers=con['bootstrap_servers'],
                        auto_offset_reset=con['auto_offset_reset'],
                        enable_auto_commit=con['enable_auto_commit'],
                        group_id=con['group_id'],
                        value_deserializer=eval(con['value_deserializer']))
        self.consumer.subscribe(self.topics)

        self.anomaly = eval(con["anomaly_detection_alg"])
        anomaly_configuration = con["anomaly_detection_conf"]
        self.anomaly.configure(anomaly_configuration)

    def read(self) -> None:
        ConsumerFile.read(self)
        ConsumerKafka.read(self)
