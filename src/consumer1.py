from kafka import KafkaConsumer, TopicPartition
from pymongo import MongoClient
from json import loads
import matplotlib.pyplot as plt
from time import sleep
import numpy as np
from MA1 import live_plotter, warning_check

#Define consumer
consumer = KafkaConsumer(
	#'anomaly_detection',
     bootstrap_servers=['localhost:9092'],
     auto_offset_reset='latest',
     enable_auto_commit=False,
     group_id='my-group',
     value_deserializer=lambda x: loads(x.decode('utf-8')))
	 

"""client = MongoClient('localhost:27017')
collection = client.numtest.numtest"""

#assign topic to read from
tp = TopicPartition('anomaly_detection', 0)
consumer.assign([tp])

#set to read data which is sent to topic after consumer is started 
#(otherwise reads where it last left off)
consumer.seek_to_end(tp)

numbers = []
timestamps = []
#set EMA parameter
N = 5 

#initiate lines for live plotter
lines = [[], [], [], [], []]

EMA = []

#maximum number of points on live plot
num_of_points = 50

for message in consumer:
    consumer.commit()
    #collection.insert_one(message.value)
    print ("value=%s" % (message.value['test_value']))
    numbers.append(float(message.value['test_value']))
    timestamps.append(float(message.value['timestamp']))
    UL = message.value['UL']
    LL = message.value['LL']
    if(len(timestamps) > N):  #to ensure the moving average calculations work
        if(len(timestamps) < num_of_points): #ensure the imputs to live plotter are consistent
            #av = (message.value['UL'] + message.value['LL'])/2
            x_vec = [None]*num_of_points + timestamps
            y_vec = [None]*num_of_points + numbers
            EMA_vec = [None]*num_of_points + EMA
			
            x_vec = x_vec[-num_of_points:]
            y_vec = y_vec[-num_of_points:]
            EMA_vec = EMA_vec[-num_of_points:]
        else:
            x_vec = timestamps[-num_of_points:]
            y_vec = numbers[-num_of_points:]
            EMA_vec = EMA[-num_of_points:]
        lines, EMA = live_plotter(x_vec, y_vec, lines, EMA_vec, N,UL, LL, 'Točke, EMA, sigma-band')
    warning_check(message.value['test_value'], UL, LL)