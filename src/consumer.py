from ABC import abstractmethod
from abc import ABC
from typing import Any, Dict

from anomalyDetection import AnomalyDetectionAbstract

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
    def _stopping_condition_(self) -> bool:
        pass

    @abstractmethod
    def _read_next_(self) -> None:
        pass

    @abstractmethod
    def configure(self) -> None:
        pass

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
            self.configure(conf)
        elif(configuration_location is not None):
            self.configure(configuration_location = configuration_location)
        else:
            # TODO: make up default configuration and call configure
            conf = {}
            self.configure(config = conf)
        
        # TODO: move this to config
        self.consumer.assign([TopicPartition(self.topic, 0)])

    def configure(con: Dict[Any, Any] = None, 
                  configuration_location: str = None) -> None:
        # TODO finish configuration
        # TODO from here config of anomaly detection is called also
        pass

    def _read_next_(self) -> None:
        self.c.seek_to_end(TopicPartition(self.topic, 0))
        last_message = self.consumer.message[-1]
        if(last_message.value is not None):
            self.last_message = [last_message.value]


class ConsumerJSON(ConsumerAbstract):

    def __init__(self) -> None:
        super().__init__()
