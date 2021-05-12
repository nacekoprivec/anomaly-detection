from typing import Any, Dict, List
import sys

sys.path.insert(0,'./src')
from anomalyDetection import AnomalyDetectionAbstract
from output import OutputAbstract, TerminalOutput, FileOutput, KafkaOutput
from visualization import VisualizationAbstract, GraphVisualization,\
    HistogramVisualization, StatusPointsVisualization
from normalization import NormalizationAbstract, LastNAverage,\
    PeriodicLastNAverage


class Clustering(AnomalyDetectionAbstract):
    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(conf)


    def configure(self, conf: Dict[Any, Any] = None,
                  configuration_location: str = None,
                  algorithm_indx: int = None) -> None:
        super().configure(conf, configuration_location=configuration_location,
                          algorithm_indx=algorithm_indx)

    def message_insert(self, message_value: Dict[Any, Any]) -> Any:
        super().message_insert(message_value)

        # Check feature vector
        if(not self.check_ftr_vector(message_value=message_value)):
            status = self.UNDEFINED
            status_code = self.UNDEFIEND_CODE

            # Remenber status for unittests
            self.status = status
            self.status_code = status_code
            return status, status_code
