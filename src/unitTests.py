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
    GAN, PCA, Hampel, AnomalyDetectionAbstract
from normalization import LastNAverage, PeriodicLastNAverage


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
        "ftr_vector" : value
    }
    return message

def create_testing_file(filepath, withzero = False, FV_length = None):
    timestamps = [1459926000 + 3600*x for x in range(100)]

    values = [1.0]*100
    if(withzero):
        values[-1] = 0.0

    vals = []
    timest = []
    if (FV_length is not None):
        for i in range(FV_length, len(values)):
            vals.append(values[i-FV_length+1:i+1])
            timest.append(timestamps[i])
        values = vals
        timestamps = timest
    
    df = pd.DataFrame({'timestamp': timestamps, 'ftr_vector': values})
    df.to_csv(filepath, index = False)

    return filepath

def create_testing_file_feature_construction(filepath):
    timestamps = [1459926000 + 3600*x for x in range(20)]
    values = [[x, x+100] for x in range(20)]
    data = {
        'timestamp': timestamps,
        'ftr_vector': values
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

    def tearDown(self) -> None:
        if os.path.isdir("configuration"):
            shutil.rmtree("configuration")

        return super().tearDown()


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


class WelfordDefinedNTestCase(unittest.TestCase):

    def setUp(self):
        configuration = {
        "input_vector_size": 1,
        "warning_stages": [0.7, 0.9],
        "N": 4,
        "X": 2,
        "output": [],
        "output_conf": [
            {}
        ],
        }
        self.model = create_model_instance("Welford()", configuration)
    
    def tearDown(self) -> None:
        if os.path.isdir("configuration"):
            shutil.rmtree("configuration")

        return super().tearDown()


class WelfordDefinedNTestClassPropperties(WelfordDefinedNTestCase):
    #Check propperties setup.
    def test_N(self):
        self.assertEqual(self.model.N, 4)

    def test_X(self):
        self.assertEqual(self.model.X, 2)

    def test_WarningStages(self):
        self.assertEqual(self.model.warning_stages, [0.7, 0.9])


class WelfordDefinedNTestFunctionality(WelfordDefinedNTestCase):
    def test_OK(self):
        test_data = [1, 2, 3, 4, 1, 2]

        for data_indx in range(len(test_data)):
            message = create_message(str(datetime.now()), [test_data[data_indx]])
            self.model.message_insert(message)

            if(data_indx < 4):
                self.assertEqual(self.model.status_code, 2)
            else:
                self.assertEqual(self.model.status_code, 1)

    def test_error(self):
        test_data = [1, 2, 3, 4, -0.1, 5.73]

        for data_indx in range(len(test_data)):
            message = create_message(str(datetime.now()), [test_data[data_indx]])
            self.model.message_insert(message)

            if(data_indx < 4):
                self.assertEqual(self.model.status_code, 2)
            else:
                self.assertEqual(self.model.status_code, -1)


class WelfordUndefinedNTestCase(unittest.TestCase):

    def setUp(self):
        configuration = {
        "input_vector_size": 1,
        "X": 2,
        "warning_stages": [],
        "output": [],
        "output_conf": [
            {}
        ],
        }
        self.model = create_model_instance("Welford()", configuration)
    
    def tearDown(self) -> None:
        if os.path.isdir("configuration"):
            shutil.rmtree("configuration")

        return super().tearDown()


class WelfordUndefinedNTestClassPropperties(WelfordUndefinedNTestCase):
    #Check propperties setup.
    def test_X(self):
        self.assertEqual(self.model.X, 2)


class WelfordUndefinedNTestFunctionality(WelfordUndefinedNTestCase):
    def test_OK(self):
        test_data = [1, 2, 2.4, 2.6, 1, 3.1]

        for data_indx in range(len(test_data)):
            message = create_message(str(datetime.now()), [test_data[data_indx]])
            self.model.message_insert(message)
            
            # Check memory length
            self.assertEqual(self.model.count, data_indx+1)

            if(data_indx < 2):
                self.assertEqual(self.model.status_code, 2)
            else:
                self.assertEqual(self.model.status_code, 1)

    def test_error(self):
        test_data = [1, 2, 3, -1, 5, -2.5]

        for data_indx in range(len(test_data)):
            message = create_message(str(datetime.now()), [test_data[data_indx]])
            self.model.message_insert(message)

            # Check memory length
            self.assertEqual(self.model.count, data_indx+1)

            if(data_indx < 2):
                self.assertEqual(self.model.status_code, 2)
            else:
                self.assertEqual(self.model.status_code, -1)


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

    def tearDown(self) -> None:
        if os.path.isdir("configuration"):
            shutil.rmtree("configuration")

        return super().tearDown()


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

    def tearDown(self) -> None:
        if os.path.isdir("configuration"):
            shutil.rmtree("configuration")

        return super().tearDown()


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

    def tearDown(self) -> None:
        if os.path.isdir("configuration"):
            shutil.rmtree("configuration")

        return super().tearDown()


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
        if not os.path.isdir("unittest"):
            os.makedirs("unittest")

        create_testing_file("./unittest/IsolForestTestData.csv", withzero = True)

        configuration = {
        "train_data": "./unittest/IsolForestTestData.csv",
        "train_conf": {
            "max_features": 3,
            "max_samples": 5,
            "contamination": "0.1",
            "model_name": "IsolForestTestModel"
        },
        "retrain_file": "./unittest/IsolationForestRetrainData.csv",
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
    
    def tearDown(self):
        if os.path.isdir(self.f):
            shutil.rmtree(self.f)

        # Delete unittest folder
        shutil.rmtree("unittest")

        if os.path.isdir("configuration"):
            shutil.rmtree("configuration")


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


class GANTestCase(unittest.TestCase):
    def setUp(self):
            # Make unittest directory
            if not os.path.isdir("unittest"):
                os.makedirs("unittest")

            create_testing_file("./unittest/GANTestData.csv", withzero = True, FV_length=10)

            configuration = {
            "train_data": "./unittest/GANTestData.csv",
            "train_conf":{
                "max_features": 1,
                "max_samples": 5,
                "contamination": "auto",
                "model_name": "GAN_Test",
                "N_shifts": 9,
                "N_latent": 3,
                "K": 0.8
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


    def tearDown(self):
        if os.path.isdir(self.f):
            shutil.rmtree(self.f)
        
        # Delete unittest folder
        shutil.rmtree("unittest")

        if os.path.isdir("configuration"):
            shutil.rmtree("configuration")


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
            self.assertEqual(self.model.status_code, -1)


class PCATestCase(unittest.TestCase):
    def setUp(self):
        # Make unittest directory
        if not os.path.isdir("unittest"):
            os.makedirs("unittest")
            
        create_testing_file("./unittest/PCATestData.csv", withzero = True)

        configuration = {
        "train_data": "./unittest/PCATestData.csv",
        "train_conf":{
            "max_features": 3,
            "max_samples": 15,
            "contamination": "0.01",
            "model_name": "PCA_Test",
            "N_components": 3
        },
        "shifts": [[1, 2, 3, 4, 5, 6]],
        "retrain_interval": 15,
        "samples_for_retrain": 15,
        "input_vector_size": 1, 
        "output": [],
        "output_conf": [{}]
        }
        self.f = "models"

        #Create a temporary /models folder.
        if not os.path.isdir(self.f):
            os.makedirs(self.f)
        self.model = create_model_instance("PCA()", configuration, save = True)

    def tearDown(self):
        if os.path.isdir(self.f):
            shutil.rmtree(self.f)

        # Delete unittest folder
        shutil.rmtree("unittest")

        if os.path.isdir("configuration"):
            shutil.rmtree("configuration")


class PCATestClassPropperties(PCATestCase):
    def test_Propperties(self):
        self.assertEqual(self.model.max_features, 3)
        self.assertEqual(self.model.max_samples, 15)
        self.assertEqual(self.model.retrain_interval, 15)
        self.assertEqual(self.model.samples_for_retrain, 15)


class PCATestFunctionality(PCATestCase):
    def test_OK(self):
        #Insert same values as in train set (status should be 1).
        test_array = [1]*10
        expected_status = [2, 2, 2, 2, 2, 2, 1, 1, 1, 1]
        for i in range(len(test_array)):
            message = create_message(str(datetime.now()), [test_array[i]])
            self.model.message_insert(message)
            self.assertEqual(self.model.status_code, expected_status[i])

    def test_errors(self):
        #Insert same values as in train set (status should be 1).
        test_array = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        expected_status = [2, 2, 2, 2, 2, 2, 1, -1, 1, -1]
        for i in range(len(test_array)):
            message = create_message(str(datetime.now()), [test_array[i]])
            self.model.message_insert(message)
            self.assertEqual(self.model.status_code, expected_status[i])


class FeatureConstructionTestCase(unittest.TestCase):
    def setUp(self) -> None:
        # Border check does not use feature construction (or vector size > 1) 
        # but will be used because of simplicity
        configuration = {
            "input_vector_size": 2,
            "averages": [[2, 3], [2]],
            "periodic_averages": [[[2, [3]], [3, [2]]], []],
            "shifts": [[1, 2, 3, 4], []],
            "time_features": ["day", "month", "weekday", "hour", "minute"],
            "warning_stages": [0.7, 0.9],
            "UL": 4,
            "LL": 2,
            "output": [],
            "output_conf": [{}]
        }
        self.model = create_model_instance("BorderCheck()", configuration)

        self.f = "models"

        #Create a temporary /models folder.
        if not os.path.isdir(self.f):
            os.makedirs(self.f)
        self.model = create_model_instance("BorderCheck()", configuration)

        # Execute feature constructions (FV-s are saved and will be checked in
        # following tests)
        test_data = [[x, x+101] for x in range(10)]
        # timestamps are 1 day and 1 hour and 1 minute apart
        timestamps = timestamps = [1459926000 + (24*3600+60+3600)*x for x in range(20)]
        self.feature_vectors = []
        for sample_indx in range(10):
            feature_vector = self.model.feature_construction(value=test_data[sample_indx],
                                            timestamp=timestamps[sample_indx])
            self.feature_vectors.append(feature_vector)
        return super().setUp()

    def tearDown(self) -> None:
        if os.path.isdir(self.f):
            shutil.rmtree(self.f)

        if os.path.isdir("configuration"):
            shutil.rmtree("configuration")

        return super().tearDown()
    
    def test_shifts(self):
        # First 4 feature vecotrs are undefined since memory does not contain
        # enough samples to construct all features
        for x in self.feature_vectors[:4]:
            self.assertFalse(x)
        
        # Extract shifted features
        shifts = [fv[7:11] for fv in self.feature_vectors[4:]]
        
        # Test shifted features
        self.assertListEqual(shifts[0], [3, 2, 1, 0])

        self.assertListEqual(shifts[1], [4, 3, 2, 1])

        self.assertListEqual(shifts[2], [5, 4, 3, 2])

        self.assertListEqual(shifts[3], [6, 5, 4, 3])

        self.assertListEqual(shifts[4], [7, 6, 5, 4])

        self.assertListEqual(shifts[5], [8, 7, 6, 5])

    def test_averages(self):
        # First 4 feature vecotrs are undefined since memory does not contain
        # enough samples to construct all features
        for x in self.feature_vectors[:4]:
            self.assertFalse(x)
        
        # Extract average features
        averages = [fv[2:5] for fv in self.feature_vectors[4:]]
        
        # Test average features
        self.assertListEqual(averages[0], [3.5, 3, 104.5])

        self.assertListEqual(averages[1], [4.5, 4, 105.5])

        self.assertListEqual(averages[2], [5.5, 5, 106.5])

        self.assertListEqual(averages[3], [6.5, 6, 107.5])

        self.assertListEqual(averages[4], [7.5, 7, 108.5])

        self.assertListEqual(averages[5], [8.5, 8, 109.5])

    def test_periodic_averages(self):
        # First 4 feature vecotrs are undefined since memory does not contain
        # enough samples to construct all features
        for x in self.feature_vectors[:4]:
            self.assertFalse(x)
        
        # Extract periodic average features
        periodic_averages = [fv[5:7] for fv in self.feature_vectors[4:]]
        
        # Test periodic average features
        self.assertListEqual(periodic_averages[0], [2, 2.5])

        self.assertListEqual(periodic_averages[1], [3, 3.5])

        self.assertListEqual(periodic_averages[2], [4, 4.5])

        self.assertListEqual(periodic_averages[3], [5, 5.5])

        self.assertListEqual(periodic_averages[4], [6, 6.5])

        self.assertListEqual(periodic_averages[5], [7, 7.5])

    def test_time_features(self):
        # First 4 feature vecotrs are undefined since memory does not contain
        # enough samples to construct all features
        for x in self.feature_vectors[:4]:
            self.assertFalse(x)
        
        # Extract time features
        time_features = [fv[11:] for fv in self.feature_vectors[4:]]
        
        # Test time features
        self.assertListEqual(time_features[0], [4, 10, 6, 11, 4])

        self.assertListEqual(time_features[1], [4, 11, 0, 12, 5])

        self.assertListEqual(time_features[2], [4, 12, 1, 13, 6])

        self.assertListEqual(time_features[3], [4, 13, 2, 14, 7])

        self.assertListEqual(time_features[4], [4, 14, 3, 15, 8])

        self.assertListEqual(time_features[5], [4, 15, 4, 16, 9])


class AverageNormalizationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        configuration = {
            "N": 4
        }
        self.normalization = LastNAverage()
        self.normalization.configure(conf=configuration)
        return super().setUp()


class AverageNormalizationClassProperties(AverageNormalizationTestCase):
    def test_properties(self):
        # Test setup from configuration
        self.assertEqual(self.normalization.N, 4)


class AverageNormalizationFunctionality(AverageNormalizationTestCase):
    def test_normalization(self):
        # Testing data and expected results
        test_data = [[x, 11+2*x] for x in range(9)]
        result_data = [
            [1, 10],
            [1, 12.25],
            [1.25, 12.5625],
            [1.3125, 12.453125],
            [1.1406, 11.81640625],
            [1.17578, 12.27050781]
        ]

        # Test add_value function
        self.normalization.add_value([1, 1])

        normalized_data = []
        for data in test_data:
            response = self.normalization.get_normalized(data)
            normalized_data.append(response)

        for fail in normalized_data[:3]:
            # First 3 entries must be False
            self.assertFalse(fail)

        # Test expected results
        for response_indx in range(6):
            self.assertAlmostEqual(normalized_data[3+response_indx][0], result_data[response_indx][0], 4)
            self.assertAlmostEqual(normalized_data[3+response_indx][1], result_data[response_indx][1], 4)


class PeriodicAverageNormalizationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        configuration = {
            "N": 4,
            "period": 2
        }
        self.normalization = PeriodicLastNAverage()
        self.normalization.configure(conf=configuration)
        return super().setUp()


class PeriodicAverageNormalizationClassProperties(PeriodicAverageNormalizationTestCase):
    def test_properties(self):
        # Test setup from configuration
        self.assertEqual(self.normalization.N, 4)
        self.assertEqual(self.normalization.period, 2)
        self.assertEqual(self.normalization.memory_len, 7)


class PeriodicAverageNormalizationFunctionality(PeriodicAverageNormalizationTestCase):
    def test_normalization(self):
        # Testing data and expected results
        test_data = [[x, 11+2*x] for x in range(11)]
        result_data = [
            [2.5, 13],
            [2.125, 14.5],
            [2.78125, 16.375],
            [2.8203125, 15.84375],
            [3.236328125, 17.0859375]
        ]

        # Test add_value function
        self.normalization.add_value([1, 1])

        normalized_data = []
        for data in test_data:
            response = self.normalization.get_normalized(data)
            normalized_data.append(response)

        for fail in normalized_data[:6]:
            # First 3 entries must be False
            self.assertFalse(fail)

        # Test expected results
        for response_indx in range(5):
            self.assertAlmostEqual(normalized_data[6+response_indx][0], result_data[response_indx][0], 4)
            self.assertAlmostEqual(normalized_data[6+response_indx][1], result_data[response_indx][1], 4)


if __name__ == '__main__':
    unittest.main(verbosity=2)