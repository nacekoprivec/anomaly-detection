import sys, os
sys.path.insert(0, './')
sys.path.append(os.path.join(os.path.dirname(__file__), "../models"))

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


from datetime import datetime
import numpy as np
import pandas as pd
import json
import shutil

import unittest
from anomalyDetection import BorderCheck, Welford, EMA, Filtering, IsolationForest,\
    GAN


def create_model_instance(algorithm_str, configuration, save = False):
        model =  eval(algorithm_str)
        model.configure(configuration)
        if (save):
            #save config to temporary file for retrain purposes
            if not os.path.isdir("configuration"):
                os.makedirs("configuration")
            
            filepath = "configuration/Test_config.txt"

            with open(filepath, 'w') as data_file:
                json.dump({"anomaly_detection_conf":[configuration]}, data_file)

            model.configure(configuration, "Test_config.txt", algorithm_indx = 0)
        else:
            model.configure(configuration, algorithm_indx = 0)

        return model

def create_message(timestamp, value):
    message = {
        "timestamp" : timestamp,
        "test_value" : value
    }
    return message

def create_testing_file(filepath, withzero = False):
    timestamps = [1459926000 + 3600*x for x in range(50)]
    values = [1]*50
    if(withzero):
        values[-1] = 0
    data = {
        'timestamp': timestamps,
        'test_value': values
    }
    testset = pd.DataFrame(data = data)
    testset.to_csv(filepath, index = False)

    return filepath


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
    #Check propperties setup.
    def test_UL(self):
        self.assertEqual(self.model.UL, 4)

    def test_LL(self):
        self.assertEqual(self.model.LL, 2)

    def test_WarningStages(self):
        self.assertEqual(self.model.warning_stages, [0.7, 0.9])

class BCTestFunctionality(BCTestCase):

    def test_OK(self):
        #Test a value at the center of the range. Should give OK status.
        message = create_message(str(datetime.now()), [3])
        self.model.message_insert(message)
        self.assertEqual(self.model.status_code, 1)

    def test_outliers(self):
        #Above UL. Should give Error (-1 status code).
        message = create_message(str(datetime.now()), [5])
        self.model.message_insert(message)
        self.assertEqual(self.model.status_code, -1)

        #Below LL. Should give Error.
        message = create_message(str(datetime.now()), [1])
        self.model.message_insert(message)
        self.assertEqual(self.model.status_code, -1)

        #Close to LL. Should give warning (0 status code)
        message = create_message(str(datetime.now()), [2.1])
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
    #Check propperties setup.
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
    #Check propperties setup.
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
        #Insert values in the middle of the range. All should have no error.
        test_array = [3, 3, 3]
        for i in test_array:
            message = create_message(str(datetime.now()), [i])
            self.model.message_insert(message)
            self.assertEqual(self.model.status_code, 1)

    def test_Error(self):
        #Check values which drift out of the range. Should transition from OK -> warning -> error
        test_array = [3, 4, 4, 4, 4, 5, 5, 5]
        expected_status = [1, 1, 1, 0, 0, -1, -1, -1]
        for i in range(len(test_array)):
            message = create_message(str(datetime.now()), [test_array[i]])
            self.model.message_insert(message)
            self.assertEqual(self.model.status_code, expected_status[i])
            
class Filtering1TestCase(unittest.TestCase):
    #Test case for filtering - mode 1
    def setUp(self):
        configuration = {
        "input_vector_size": 1,
        "mode": 1,
        "LL": 0,
        "UL": 1,
        "filter_order": 3,
        "cutoff_frequency":0.4,
        "warning_stages": [0.7, 0.9], 
        "output": [],
        "output_conf": [{}],
        }
        self.model = create_model_instance("Filtering()", configuration)

class Filtering0TestCase(unittest.TestCase):
    #Test case for filtering - mode 0
    def setUp(self):
        configuration = {
        "input_vector_size": 1,
        "mode": 0,
        "LL": 0,
        "UL": 1,
        "filter_order": 3,
        "cutoff_frequency":0.4,
        "warning_stages": [0.7, 0.9], 
        "output": [],
        "output_conf": [{}],
        }
        self.model = create_model_instance("Filtering()", configuration)

class Filtering1TestClassPropperties(Filtering1TestCase):
    #Check propperties setup.
    def test_UL(self):
        self.assertEqual(self.model.UL, 1)

    def test_LL(self):
        self.assertEqual(self.model.LL, 0)

    def test_N(self):
        self.assertEqual(self.model.filter_order, 3)

    def test_CutoffFrequency(self):
        self.assertEqual(self.model.cutoff_frequency, 0.4)

    def test_WarningStages(self):
        self.assertEqual(self.model.warning_stages, [0.7, 0.9])

    def test_Mode(self):
        self.assertEqual(self.model.mode, 1)

    def test_Kernel(self):
        #Test kernel coefficients
        self.assertAlmostEqual(self.model.a[0], 1, 8)
        self.assertAlmostEqual(self.model.a[1], -0.57724052, 8)
        self.assertAlmostEqual(self.model.a[2], 0.42178705, 8)
        self.assertAlmostEqual(self.model.a[3], -0.05629724, 8)

        self.assertAlmostEqual(self.model.b[0], 0.09853116, 8)
        self.assertAlmostEqual(self.model.b[1], 0.29559348, 8)
        self.assertAlmostEqual(self.model.b[2], 0.29559348, 8)
        self.assertAlmostEqual(self.model.b[3], 0.09853116, 8)

        self.assertAlmostEqual(self.model.z[0], 0.90146884, 8)
        self.assertAlmostEqual(self.model.z[1], 0.02863483, 8)
        self.assertAlmostEqual(self.model.z[2], 0.1548284, 8)

class Filtering1TestFunctionality(Filtering1TestCase):
    def test_Constant(self):
        #Test constant datastream.
        test_array = np.ones(10)
        for i in test_array:
            message = create_message(str(datetime.now), [i])
            self.model.message_insert(message)
            self.assertAlmostEqual(self.model.filtered, 1, 8)
            self.assertAlmostEqual(self.model.result, 0, 8)

    def test_Errors(self):
        #Test drifting datastream.
        test_array = [0, 0, 0, 1, 2, 2, 2]
        expected_status = [0, 1, 1, -1, -1, 1, 1]
        for i in range(len(test_array)):
            message = create_message(str(datetime.now), [test_array[i]])
            self.model.message_insert(message)
            self.assertEqual(self.model.status_code, expected_status[i])

class Filtering0TestFunctionality(Filtering0TestCase):
    def test_Constant(self):
        #Test constant datastream.
        test_array = np.ones(10)
        for i in test_array:
            message = create_message(str(datetime.now), [i])
            self.model.message_insert(message)
            self.assertAlmostEqual(self.model.filtered, 1, 8)
            self.assertAlmostEqual(self.model.result, 1, 8)

    def test_Errors(self):
        #Test drifting datastream.
        test_array = [0.5, 0.5, 0.5, 1, 1, 1, 2, 2, 2]
        expected_status = [0, 1, 1, 1, 1, 0, -1, -1, -1]
        for i in range(len(test_array)):
            message = create_message(str(datetime.now), [test_array[i]])
            self.model.message_insert(message)
            self.assertEqual(self.model.status_code, expected_status[i])

class IsolForestTestCase(unittest.TestCase):
    def setUp(self):
        create_testing_file("./unittest/IsolForestTestData.csv", withzero = True)

        configuration = {
        "train_data": "./unittest/IsolForestTestData.csv",
        "train_conf": {
            "max_features": 3,
            "max_samples": 5,
            "contamination": "0.1",
            "model_name": "IsolForestTestModel"
        },
        "retrain_interval": 10,
        "samples_for_retrain": 5,
        "input_vector_size": 1,
        "shifts": [[1,2,3,4]],
        "averages": [[1,2]],
        "output": [],
        "output_conf": [{}]
        }
        self.f = "models"

        #Create a temporary /models folder.
        if not os.path.isdir(self.f):
            os.makedirs(self.f)
        self.model = create_model_instance("IsolationForest()", configuration, save = True)

class IsolForestTestClassPropperties(IsolForestTestCase):
    #Check propperties setup.
    def test_MaxFeatures(self):
        self.assertEqual(self.model.max_features, 3)

    def test_MaxSamples(self):
        self.assertEqual(self.model.max_samples, 5)

    def test_RetrainInterval(self):
        self.assertEqual(self.model.retrain_interval, 10)

    def test_SamplesForRetrain(self):
        self.assertEqual(self.model.samples_for_retrain, 5)

class IsolForestTestFunctionality(IsolForestTestCase):
    def test_OK(self):
        #Insert same values as in train set (status should be 1).
        test_array = [1]*10
        expected_status = [2, 2, 2, 2, 1, 1, 1, 1, 1, 1]
        for i in range(len(test_array)):
            message = create_message(str(datetime.now()), [test_array[i]])
            self.model.message_insert(message)
            self.assertEqual(self.model.status_code, expected_status[i])

    def test_errors(self):
        #insert different values as in train set (status should be -1).
        test_array = [0]*10
        expected_status = [2, 2, 2, 2, -1, -1, -1, -1, -1, -1]
        for i in range(len(test_array)):
            message = create_message(str(datetime.now()), [test_array[i]])
            self.model.message_insert(message)
            self.assertEqual(self.model.status_code, expected_status[i])

    def test_cleanup(self):
        #Delete models folder and check.
        if os.path.isdir(self.f):
            shutil.rmtree(self.f)
        self.assertEqual(os.path.isdir(self.f), False)

class GANTestCase(unittest.TestCase):
 def setUp(self):
        create_testing_file("./unittest/GANTestData.csv", withzero = True)

        configuration = {
        "train_data": "./unittest/GANTestData.csv",
        "train_conf":{
            "max_features": 1,
            "max_samples": 5,
            "contamination": "auto",
            "model_name": "GAN_Test",
            "N_shifts": 9,
            "N_latent": 3
        },
        "retrain_interval": 15,
        "samples_for_retrain": 15,
        "input_vector_size": 10,
        "output": [],
        "output_conf": [{}]
        }
        self.f = "models"

        #Create a temporary /models folder.
        if not os.path.isdir(self.f):
            os.makedirs(self.f)
        self.model = create_model_instance("GAN()", configuration, save = True)

class GANTestClassPropperties(GANTestCase):
    #Check propperties setup.
    def test_Propperties(self):
        self.assertEqual(self.model.max_features, 1)
        self.assertEqual(self.model.max_samples, 5)
        self.assertEqual(self.model.N_shifts, 9)
        self.assertEqual(self.model.N_latent, 3)
        self.assertEqual(self.model.retrain_interval, 15)
        self.assertEqual(self.model.samples_for_retrain, 15)


class GANTestFunctionality(GANTestCase):
    def test_OK(self):
        #Insert same values as in train set (status should be 1).
        test_array = [1]*10
        for i in range(len(test_array)):
            message = create_message(str(datetime.now()), test_array)
            self.model.message_insert(message)
            self.assertEqual(self.model.status_code, 1)

    def test_errors(self):
        #Insert same values as in train set (status should be 1).
        test_array = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        for i in range(len(test_array)):
            message = create_message(str(datetime.now()), test_array)
            self.model.message_insert(message)
            self.assertGreater(self.model.GAN_error, 1)
    
    def test_cleanup(self):
        #Delete models folder and check.
        if os.path.isdir(self.f):
            shutil.rmtree(self.f)
        self.assertEqual(os.path.isdir(self.f), False)

if __name__ == '__main__':
    unittest.main(verbosity=2)