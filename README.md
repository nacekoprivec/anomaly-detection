# Anomaly detection for streaming data

## Usage
`python main.py [-h] [-c CONFIG] [--f]`

| Short   |      Long     |  Description |
|----------|-------------|------|
| `-h` | `--help` | show help |
| `-c CONFIG` | `--config CONFIG` | name of the config file located in configuration folder (example: `config.json`) |
| `-f` | `--file` | If this flag is used the program will read data from file specified in config file instead of kafka stream|

## Architecture
The anomaly detection program consists of three main types of components:
1. Consumer component
2. Anomaly detection component
3. Output component
Each component has many implementations that are interchangeable. Which ones are used depends on the task the program is solving.
There is also an optional Visualization component which doesn't effect the general workflow and is used for streaming data visualization.

### Configuration file
The program is configured through configuration file specified with -c flag (located in configuration folder). It is a JSON file with the following structure:
```
{
    ...
    consumer configuration
    ...
    "anomaly_detection_alg": "anomaly detection algorithm",
    "anomaly_detection_conf": {
        ...
        anomaly detection configuration
        ...
        "output": ["list of output components"],
        "output_conf": ["list of output components configurations"],
        "visualization": ["list of visualization components"], # optional
        "visualization_conf":["list of visualization components configurations"] # optional
    }
}
```
The specific configurations for components will be presented in following chapters
For more details see example configuration files in configuration folder.

### Consumer
Consumer components differ in where the data is read from.
1. **Kafka consumer:** Data is read from kafka stream from a specified topic. The conciguration file must specify following parameters:
   * bootstrap_servers: Kafka server (example: ["localhost:9092"])
   * auto_offset_reset": TODO (example: "latest")
   * enable_auto_commit": "True",
   * group_id": "my-group",
   * value_deserializer": "lambda x: loads(x.decode('utf-8'))",
   * topics": ["anomaly_detection"],

### Output

### Visualization

### Anomaly detection
 