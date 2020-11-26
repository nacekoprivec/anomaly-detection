from json import loads

from consumer import ConsumerKafka
from output import TerminalOutput, GraphOutput
from anomalyDetection import BorderCheck

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

con = ConsumerKafka()
con.configure(configuration)
con.read()