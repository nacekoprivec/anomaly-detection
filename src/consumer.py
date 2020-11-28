from abc import ABC
from abc import abstractmethod


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

    #@abstractmethod
    def stopping_condition(self) -> bool:
        pass

    #@abstractmethod
    def read_next(self) -> None:
        pass

    #@abstractmethod
    def configure(self) -> None:
        pass

    def read(self) -> None:
        while(self._stopping_condition_):
            self._read_next_()
            self.anomaly.check(self.last_message)


class ConsumerKafka(ConsumerAbstract):
    consumer: KafkaConsumer

    def __init__(self, conf: Dict[Any, Any] = None, 
                 configuration_location: str = None, topics = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(conf, topics = topics)
        elif(configuration_location is not None):
            self.configure(configuration_location = configuration_location)
        else:
            # TODO: make up default configuration and call configure
            conf = {}
            self.configure(con = conf, topics = topics)
        

    def configure(self, con: Dict[Any, Any] = None, 
                  configuration_location: str = None, topics = None) -> None:
        if(con is not None):
            self.topics = topics
            self.consumer = KafkaConsumer(
                            bootstrap_servers=con['bootstrap_servers'],
                            auto_offset_reset=con['auto_offset_reset'],
                            enable_auto_commit=con['enable_auto_commit'],
                            group_id=con['group_id'],
                            value_deserializer=con['value_deserializer'])
            self.consumer.subscribe(topics)
        elif(configuration_location is not none):
            pass
        else:
            pass
            # TODO
        # TODO finish configuration
        # TODO from here config of anomaly detection is called also
        pass

    def read_next(self) -> None:
        msg = self.consumer.poll(1.0)
        return msg


class ConsumerJSON(ConsumerAbstract):

    def __init__(self) -> None:
        super().__init__()



# Main
#check_list = {'topic_one': check_for_topic_one,
#             'topic_two': check_for_topic_two}

#c = ConsumerKafka(config, topics = check_list.keys())	


#while True:
#    msg = c._read_next()
#    if msg is None:
#        continue
#    if msg.error():
#        print("Consumer error: {}".format(msg.error()))
#        continue

    #send message to specified anomaly detection method
#    check_list[msg.topic()](msg)

