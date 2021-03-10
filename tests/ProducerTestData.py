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

df = pd.read_csv("../data/consumer/braila_test.csv", delimiter = ",")
#df['time'] = pd.to_datetime(df['time'],unit='s')
#values = df['value']
#times = df['time']

values = df['analog2'].values
times = df['timestamp']

for i in range(len(values)):

    "Artificially add some anomalies"
    #if(i%20 == 0):
    #    ran = np.random.choice([-1, 1])*5
    #else:
    #    ran = 0
    value = list(values[i:i+1])
    anomaly = 0
    if (i%20 == 0):
        anomaly = -0.03
    data = {"test_value" : value,
			"timestamp": str(times[i])}
    print(data)

	
    producer.send('anomaly_detection1', value=data)
    sleep(1) #one data point each second
