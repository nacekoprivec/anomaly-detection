import sys
sys.path.insert(0,'./src')

from datetime import datetime

import unittest
from anomalyDetection import BorderCheck, Welford, EMA

def create_model_instance(algorithm_str, configuration):
        model =  eval(algorithm_str)
        model.configure(configuration)
        return model

def create_message(timestamp, value):
    message = {
        "timestamp" : timestamp,
        "test_value" : value
    }
    return message

class BCTestCase(unittest.TestCase):

    def setUp(self):
        configuration = {
        "input_vector_size": 1,
        "warning_stages": [0.7, 0.9],
        "UL": 4,
        "LL": 2,
        "output": [],
        "output_conf": [{}]
        }
        self.model = create_model_instance("BorderCheck()", configuration)

class BCTestClassPropperties(BCTestCase):

    def test_UL(self):
        self.assertEqual(self.model.UL, 4)

    def test_LL(self):
        self.assertEqual(self.model.LL, 2)

    def test_WarningStages(self):
        self.assertEqual(self.model.warning_stages, [0.7, 0.9])

class BCTestFunctionality(BCTestCase):

    def test_OK(self):
        message = create_message(str(datetime.now()), [3])
        self.model.message_insert(message)
        self.assertEqual(self.model.status_code, 1)

    def test_outliers(self):
        message = message = create_message(str(datetime.now()), [5])
        self.model.message_insert(message)
        self.assertEqual(self.model.status_code, -1)

        message = message = create_message(str(datetime.now()), [1])
        self.model.message_insert(message)
        self.assertEqual(self.model.status_code, -1)

        message = message = create_message(str(datetime.now()), [2.1])
        self.model.message_insert(message)
        self.assertEqual(self.model.status_code, 0)




class WelfordTestCase(unittest.TestCase):

    def setUp(self):
        configuration = {
        "input_vector_size": 1,
        "warning_stages": [0.7, 0.9],
        "N": 4,
        "X": 3,
        "output": [],
        "output_conf": [
            {}
        ],
        }
        self.model = create_model_instance("Welford()", configuration)

class WelfordTestClassPropperties(WelfordTestCase):

    def test_N(self):
        self.assertEqual(self.model.N, 4)

    def test_X(self):
        self.assertEqual(self.model.X, 3)

    def test_WarningStages(self):
        self.assertEqual(self.model.warning_stages, [0.7, 0.9])



class EMATestCase(unittest.TestCase):

    def setUp(self):
        configuration = {
        "input_vector_size": 1,
        "warning_stages": [0.7, 0.9],
        "UL": 4,
        "LL": 2,
        "N": 5,
        "output": [],
        "output_conf": [{}],
        }
        self.model = create_model_instance("EMA()", configuration)

class EMATestClassPropperties(EMATestCase):

    def test_UL(self):
        self.assertEqual(self.model.UL, 4)

    def test_LL(self):
        self.assertEqual(self.model.LL, 2)

    def test_N(self):
        self.assertEqual(self.model.N, 5)

    def test_WarningStages(self):
        self.assertEqual(self.model.warning_stages, [0.7, 0.9])

class EMATestFunctionality(EMATestCase):

    def test_OK(self):
        test_array = [3, 3, 3]
        for i in test_array:
            message = create_message(str(datetime.now()), [i])
            self.model.message_insert(message)
            self.assertEqual(self.model.status_code, 1)

    def test_Error(self):
        test_array = [3, 4, 4, 4, 4, 5, 5, 5]
        expected_status = [1, 1, 1, 0, 0, -1, -1, -1]
        for i in range(len(test_array)):
            message = create_message(str(datetime.now()), [test_array[i]])
            self.model.message_insert(message)
            self.assertEqual(self.model.status_code, expected_status[i])

    



if __name__ == '__main__':
    unittest.main(verbosity=2)