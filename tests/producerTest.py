import csv
import json
from time import sleep
from json import dumps
from kafka import KafkaProducer
import numpy as np
from datetime import datetime
import time

#Define producer
producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                         value_serializer=lambda x: 
                         dumps(x).encode('utf-8'))



#Send random data to topic - sine with noise in range 2-4

tab_data = [3, 4, 4, 4, 4, 5, 5, 5]
tab_data_csv = []

timestamp = time.time()

for e in range(100):
    timestamp += 60
    ran = float(np.random.normal(0, 1))
    
    data = {"ftr_vector" : [ran],
			"timestamp": timestamp}
    
    producer.send('input_stream', value=data)
    sleep(1) #one data point each second

