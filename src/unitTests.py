import sys
sys.path.insert(0,'./src')

import unittest
from anomalyDetection import BorderCheck

def create_model_instance(algorithm_str):
        
        
        configuration = {
        "input_vector_size": 2,
        "warning_stages": [0.7, 0.9],
        "UL": 4,
        "LL": 2,
        "output": ["FileOutput()"],
        "output_conf": [
            {
                "file_name": "sin.csv",
                "mode": "w"
            }
        ]
        }
        model =  eval(algorithm_str)
        model.configure(configuration)
        
        return model

class BCTestCase(unittest.TestCase):

    def setUp(self):
        self.model = create_model_instance("BorderCheck()")

class BCTestClassPropperties(BCTestCase):

    def test_UL(self):
        self.assertEqual(self.model.UL, 4)

    def test_LL(self):
        self.assertEqual(self.model.LL, 2)

    def test_WarningStages(self):
        self.assertEqual(self.model.warning_stages, [0.7, 0.9])


if __name__ == '__main__':
    unittest.main(verbosity=2)