from ABC import abstractmethod
from abc import ABC
from typing import NoReturn

from anomalyDetection import AnomalyDetectionAbstract

from kafka import KafkaConsumer, TopicPartition
from pymongo import MongoClient
from json import loads
import matplotlib.pyplot as plt
from time import sleep
import numpy as np

class ConsumerAbstract(ABC):
    anomaly: "AnomalyDetectionAbstract"
    def __init__(self) -> None:
        pass

    @abstractmethod
    def _stopping_condition_(self) -> bool:
        pass

    @abstractmethod
    def _read_next_(self) -> None:
        pass

    def read(self) -> None:
        while(self._stopping_condition_):
            self._read_next_()
            self.anomaly.check()

class ConsumerKafka(ConsumerAbstract):

    def __init__(self, config, topic: str) -> None:
        super().__init__()
		self.topic = topic
		if(config != None):
			c = KafkaConsumer(config)
			c.assign([TopicPartition(self.topic, 0)])
			c.seek_to_end(TopicPartition(self.topic, 0))
		self.consumer = c
	
	def _read_next_(self):
		last_message = self.consumer.message[-1]
		return last_message.value
	


	