from abc import ABC, abstractmethod
import json

from typing import Any, Dict

from src.anomalyDetection import AnomalyDetectionAbstract, EMA, BorderCheck

from kafka import KafkaConsumer, TopicPartition
from pymongo import MongoClient
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

    def configure(self, con: Dict[Any, Any] = None,
                  configuration_location: str = None) -> None:
        if(con is not None):
            self.topics = con['topics']
            self.consumer = KafkaConsumer(
                            bootstrap_servers=con['bootstrap_servers'],
                            auto_offset_reset=con['auto_offset_reset'],
                            enable_auto_commit=con['enable_auto_commit'],
                            group_id=con['group_id'],
                            value_deserializer=eval(con['value_deserializer']))
            self.consumer.subscribe(self.topics)
        elif(configuration_location is not None):
            # Read config file
            with open("configuration/" + configuration_location) as data_file:
                conf = json.load(data_file)
            self.configure(con=conf)
        else:
            print("No configuration was given")
            return

        self.anomaly = eval(con["anomaly_detection_alg"])
        anomaly_configuration = con["anomaly_detection_conf"]
        self.anomaly.configure(anomaly_configuration)

    def read(self) -> None:
        for message in self.consumer:
            value = message.value
            self.anomaly.message_insert(value)


class ConsumerFile(ConsumerAbstract):
    file_type: str
    file_location: str

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

    def configure(self, conf: Dict[Any, Any] = None) -> None:
        pass

    def read(self) -> None:
        pass

# Main
# check_list = {'topic_one': check_for_topic_one,
#             'topic_two': check_for_topic_two}

# c = ConsumerKafka(config, topics = check_list.keys())


# while True:
#    msg = c._read_next()
#    if msg is None:
#        continue
#    if msg.error():
#        print("Consumer error: {}".format(msg.error()))
#        continue

    # send message to specified anomaly detection method
#    check_list[msg.topic()](msg)
