from json import loads

from consumer import ConsumerAbstract, ConsumerKafka
from output import TerminalOutput, GraphOutput, HistogramOutput
from anomalyDetection import BorderCheck, EMA

configuration = {
    "bootstrap_servers": ['localhost:9092'],
    "auto_offset_reset": 'latest',
    "enable_auto_commit": False,
    "group_id": 'my-group',
    "value_deserializer": lambda x: loads(x.decode('utf-8')),
    "topic": "anomaly_detection",
    "anomaly_detection_alg": BorderCheck(),
    "anomaly_detection_conf": {
        "memory_size": 5,
        "UL": 4,
        "LL": 2,
        "warning_stages": [0.7, 0.9],
        "value_index": 0,
        "output": [TerminalOutput(), GraphOutput()],
        "output_conf": [
            {},
            {}
        ]
    }
}

grafConfiguration = {
    "num_of_points": 50,
    "num_of_lines": 4,
    "linestyles":['wo', 'r-', 'b--', 'b--']
}

emaConfiguration = {
    "N": 5,
    "num_of_points": 50,
    "UL": 4,
    "LL": 2,
    "title": "Title"
}

kafkaConfiguration = {
    "bootstrap_servers": ['localhost:9092'],
    "auto_offset_reset": 'latest',
    "enable_auto_commit": True,
    "group_id": 'my-group',
    "value_deserializer": lambda x: loads(x.decode('utf-8'))
}


consumer1 = ConsumerKafka(conf = kafkaConfiguration, topics = ['anomaly_detection'])

EMA = EMA(emaConfiguration)
#graf = GraphOutput(grafConfiguration)
hist = HistogramOutput()
for message in consumer1.consumer:
    value = message.value

    MovingAverage, sigma = EMA.message_insert(message)
    plotPoints = [value['test_value'], MovingAverage, MovingAverage + sigma, MovingAverage - sigma]
 

    hist.send_out(value['test_value'])
