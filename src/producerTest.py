from time import sleep
from json import dumps
from kafka import KafkaProducer
import numpy as np
import datetime

#Define producer
producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                         value_serializer=lambda x: 
                         dumps(x).encode('utf-8'))



#Send random data to topic - sine with noise in range 2-4
					 
for e in range(1000):
    timestamp = e
    ran = float(np.random.normal(0, 0.1) + np.sin(0.1*e))
    data = {'test_value' : 3 + ran,
			'timestamp': e}
	
    producer.send('anomaly_detection', value=data)
    sleep(1) #one data point each second