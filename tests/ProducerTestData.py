import csv
import json
from time import sleep
from json import dumps
from kafka import KafkaProducer
import numpy as np
from datetime import datetime
import pandas as pd

#Define producer
producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                         value_serializer=lambda x: 
                         dumps(x).encode('utf-8'))


"load real data from Continental, send it to kafka topic"

df = pd.read_csv("../data/Braila_new_data/braila_pressure5771.csv", delimiter = ",")
df['time'] = pd.to_datetime(df['time'],unit='s')

values = df['value']
times = df['time']

for i in range(len(values)):

    "Artificially add some anomalies"
    #if(i%20 == 0):
    #    ran = np.random.choice([-1, 1])*5
    #else:
    #    ran = 0
    value = values[i]
    data = {"test_value" : [value],
			"timestamp": str(times[i])}

	
    producer.send('anomaly_detection1', value=data)
    sleep(1) #one data point each second
