import argparse
import json
import sys
import requests
import threading
import time
import logging
from datetime import datetime
import itertools  
from typing import Any, Dict, List
from multiprocessing import Process
from src.Test import Test
from src.consumer import ConsumerAbstract, ConsumerFile, ConsumerKafka

from src.AnomalyDetectorWrapper import AnomalyDetectorWrapper
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV
import numpy as np


def ping_watchdog(process: Process) -> None:
    """
    Function to ping the watchdog at regular intervals.

    Args:
        process (Process): The child process to monitor.

    Returns:
        None
    """
    interval = 30 # ping interval in seconds
    url = "localhost"
    port = 5001
    path = "/pingCheckIn/Data adapter"

    while(process.is_alive()):
        print("{}: Pinging.".format(datetime.now()))
        try:
            r = requests.get("http://{}:{}{}".format(url, port, path))
        except requests.exceptions.RequestException as e:
            logging.warning(e)
        else:
            logging.info('Successful ping at ' + time.ctime())
        time.sleep(interval)

def custom_scorer(estimator, X, y=None):
    return estimator.score(X, y)

def flatten_grid(conf_dict):
    """
    Converts a single dict (with lists as values) to a flat grid-style param dict
    that can be used in GridSearchCV.
    """
    from itertools import product

    keys, values = zip(*conf_dict.items())
    combos = [dict(zip(keys, v)) for v in product(*values)]
    return combos

def merge_param_dicts(dicts):
    """Merge list of dicts into a dict of lists for GridSearchCV."""
    from collections import defaultdict

    merged = defaultdict(set)
    for d in dicts:
        for k, v in d.items():
            merged[k].add(v)

    # Convert sets to sorted lists
    return {k: sorted(list(v)) for k, v in merged.items()}



def start_consumer(args: argparse.Namespace) -> None:
    """
    Function to start the consumer based on the command line arguments.

    Args:
        args (argparse.Namespace): Parsed command line arguments.

    Returns:
        None
    """
    if(args.data_file):
        consumer = ConsumerFile(configuration_location=args.config)
    
    elif args.param_tunning:

        start_time = time.time()

        #BorderCheck parameter combinations
        param_grid = {}

        fixed_params = {
            "input_vector_size": 1,
            "output": ["TerminalOutput()"],
            "output_conf": [{}],
        }

        param_options = {
                # " Best parameters: {'UL': 0.8, 'LL': -0.5} Best F1 score: 0.04919561669386804 === Program completed in 85.21 seconds ==="
                "border_check": {
                    "param_grid": {
                        "file_name": "data/ads-1.csv",
                        "anomaly_detection_alg": ["BorderCheck()"],
                        "anomaly_detection_conf": [{
                            "UL": [0.5, 0.7, 0.8], 
                            "LL": [-0.5, -0.7, -0.8],  
                        }]
                    },
                    "fixed_params": {"warning_stages": [0.0, 0.0]}
                },

                #Best parameters: {'treshold': 0.4, 'samples_for_retrain': 100, 'retrain_interval': 200, 'min_samples': 10, 'eps': 1.0}
                #Best F1 score: 0.043254034327372404
                #=== Program completed in 659.01 seconds ===
                "clustering": {  
                    "param_grid": {
                        "file_name": ["data/ads-1.csv"],
                        "anomaly_detection_alg": ["Clustering()"],
                        "anomaly_detection_conf": [
                            {
                                "eps": [0.5, 1.0, 2.0],  
                                "min_samples": [3, 5, 10],
                                "treshold": [0.2, 0.3, 0.4],  
                                "retrain_interval": [50, 100, 200],
                                "samples_for_retrain": [20, 50, 100],
                                
                            }
                        ]
                    },
                    "fixed_params": {
                        "retrain_file": "data/ads-1_train_unlabeled.csv",
                        "train_data": "data/ads-1_train_unlabeled.csv"                    
                        }
                },

                #Best parameters: {'decay': 0.3, 'averaging': 10, 'UL': 1.0, 'LL': -2.0}
                #Best F1 score: 0.02551418901327779
                #=== Program completed in 117.29 seconds ===
                "cumulative": {
                    "param_grid": {
                        "file_name": "data/ads-1.csv",
                        "anomaly_detection_alg": ["Cumulative()"],
                        "anomaly_detection_conf": [{
                            "decay": [0.1, 0.2, 0.3],      
                            "averaging": [5, 10],        
                            "UL": [1.0, 1.5, 2.0],              
                            "LL": [-1.0, -1.5, -2.0],             
                        }]
                    },
                    "fixed_params": {"warning_stages": [0.0, 0.0]}
                },

                #Best parameters: {'window': 20, 'start_on': 20, 'period': 30, 'percentile': 5}
                #Best F1 score: 0.01749068636736756
                #=== Program completed in 124.34 seconds ===
                "ema_percentile": {
                    "param_grid": {
                        "file_name": "data/ads-1.csv",
                        "anomaly_detection_alg": ["EMA_Percentile()"],
                        "anomaly_detection_conf": [{
                            "percentile": [5, 10 , 15],
                            "window": [10, 20, 50],
                            "start_on": [10, 20, 30],
                            "period": [10, 20, 30],
                            
                        }]
                    },
                    "fixed_params": {
                    }
                },

                #Best parameters: {'UL': 0.5, 'N': 5, 'LL': -0.5}
                #Best F1 score: 0.0295510513354802
                #=== Program completed in 138.36 seconds ===
                "ema": {
                    "param_grid": {
                        "file_name": "data/ads-1.csv", 
                        "anomaly_detection_alg": ["EMA()"],
                        "anomaly_detection_conf": [{
                            "N": [5, 10, 20],
                            "UL": [0.5, 1.0, 1.5],
                            "LL": [-0.5, -1.0, -1.5]
                        }]
                    },
                    "fixed_params": {"warning_stages": [0.0, 0.0]}
                },

                "fb_prophet": {},
                
                #Best parameters: {'mode': 1, 'filter_order': 6, 'cutoff_frequency': 0.2, 'UL': 0.5, 'LL': -1.5}
                #Best F1 score: 0.2741433021806854
                #=== Program completed in 111.79 seconds ===
                "filtering": {
                    "param_grid": {
                        "file_name": "data/ads-1.csv",
                        "anomaly_detection_alg": ["Filtering()"],
                        "anomaly_detection_conf": [{
                            "mode": [1, 2, 3],                  
                            "filter_order": [2, 4, 6],           
                            "cutoff_frequency": [0.05, 0.1, 0.2], 
                            "UL": [0.5, 1.0, 1.5],               
                            "LL": [-1.0, -0.5, -1.5],           
                        }]
                    },
                    "fixed_params": {
                        "warning_stages": [0.0, 0.0]  
                    }
                },

                #Best parameters: {'train_data': 'data/ads-1_train.csv', 'model_name': 'GAN_sensor_cleaning', 'len_window': 250, 'filtering': 'None', 'N_shifts': 0, 'N_latent': 4, 'K': 0.4}
                #Best F1 score: 0.04179557610566512
                #=== Program completed in 2506.18 seconds === RandomizedSearchCV
                "gan": { 
                    "param_grid": {
                        "file_name": "data/ads-1.csv",
                        "anomaly_detection_alg": ["GAN()"],
                        "anomaly_detection_conf": [{
                            "filtering": ["None"],
                            "train_data": ["data/ads-1_train.csv"],
                            "model_name": ["GAN_sensor_cleaning"],
                            "N_shifts": [0, 1, 2],
                            "N_latent": [2, 3, 4],
                            "K": [0.3, 0.4, 0.5],
                            "len_window": [250, 500, 750],
                        }]
                    },
                    "fixed_params": {
                        "warning_stages": [0.0, 0.0]
                    }
                },

                "hampel": { #TODO: check if parameters are correct 
                    "param_grid": {
                        "file_name": "data/ads-1.csv",
                        "anomaly_detection_alg": ["Hampel()"],
                        "anomaly_detection_conf": [{
                            "n_sigmas": [1.5, 2.0],
                            "W": [3, 5],
                            "K": [1.2, 1.5, 2.0]
                        }]
                    },
                    "fixed_params": {
                        "input_vector_size": 1,
                        "output": ["TerminalOutput()"],
                        "output_conf": [{}],
                    }
                },

                "isolation_forest": { #TODO: check if train data/retrain right & long to run
                    "param_grid": {
                        "file_name": "data/ads-1.csv",
                        "anomaly_detection_alg": ["IsolationForest()"],
                        "anomaly_detection_conf": [{
                        "retrain_interval": [25, 50, 100],
                        "samples_for_retrain": [100, 200, 400],
                        "max_samples": [64, 128, 256],
                        "max_features": [0.5, 0.75, 1.0],
                        }]
                    },
                    "fixed_params": {
                        "model_name": "isolation_forest_model.pkl",
                        "train_data": "data/ads-1_train.csv",
                        "retrain_file": "ads-1_train_unlabeled.csv",
                        "filtering": "None",
                        "warning_stages": [0.0, 0.0]
                    }
                },

                #Best parameters: {'confidence_norm': 0.2, 'UL': 1.2, 'N': 5, 'LL': -0.8}
                #Best F1 score: 0.041811846689895474
                #=== Program completed in 212.13 seconds ===
                "linear_fit": {
                    "param_grid": {
                        "file_name": "data/ads-1.csv",
                        "anomaly_detection_alg": ["LinearFit()"],
                        "anomaly_detection_conf": [{
                            "UL": [0.8, 1.0, 1.2],         
                            "LL": [-1.2, -1.0, -0.8],     
                            "confidence_norm": [0.1, 0.2, 0.3], 
                            "N": [5, 10, 20]    
                        }]
                    },
                    "fixed_params": {
                        "input_vector_size": 1,
                        "output": ["TerminalOutput()"],
                        "output_conf": [{}],
                        "warning_stages": [0.0, 0.0],
                    }
                },

                #Best parameters: {'period2': 26, 'period1': 8, 'UL': 0.8, 'LL': -0.8}
                #Best F1 score: 0.04912280701754385
                #=== Program completed in 129.50 seconds ===
                "macd": {
                    "param_grid": {
                        "file_name": "data/ads-1.csv",
                        "anomaly_detection_alg": ["MACD()"],
                        "anomaly_detection_conf": [{
                            "period1": [8, 12, 16],          
                            "period2": [20, 26, 32],         
                            "UL": [0.8, 1.0, 1.2],     
                            "LL": [-1.2, -1.0, -0.8],      
                        }]
                    },
                    "fixed_params": {
                        "warning_stages": [0.0, 0.1, 0.2],
                        "filtering": "None",
                    }
                },

                "pca": { # very slow
                    "param_grid": {
                        "file_name": "data/ads-2.csv",
                        "anomaly_detection_alg": ["PCA()"],
                        "anomaly_detection_conf": [{
                            "max_features": [0.5, 0.75, 1.0],
                            "max_samples": [50, 100, 200],
                            "N_components": [0.9, 0.95, 1],    
                        }]
                    },
                    "fixed_params": {
                        "input_vector_size": 1,
                        "retrain_interval": 50,
                        "model_name": "pca_model",
                        "retrain_file": "data/ads-1_train.csv",
                        "samples_for_retrain": 200,
                        "train_data": "data/ads-1_train.csv",
                        "output": ["TerminalOutput()"],
                        "output_conf": [{}]
                    }
                },

                "rrcf_trees": { # very slow
                    "param_grid": {
                        "file_name": "data/ads-1.csv",
                        "anomaly_detection_alg": ["RRCF_trees()"],
                        "anomaly_detection_conf": [{
                            "num_trees": [50, 100, 150],   
                            "tree_size": [128, 256, 512],
                            "UL": [0.3, 0.5, 0.7],   
                            "LL": [-0.7, -0.5, -0.3]      
                        }]
                    },
                    "fixed_params": {
                        "filtering": "None",
                        "input_vector_size": 1,
                        "warning_stages": [0.0, 0.0],
                        "output": ["TerminalOutput()"],
                        "output_conf": [{}]
                    }
                },

                "trend_classification": { # very slow
                    "param_grid": {
                        "file_name": "data/ads-1.csv",
                        "anomaly_detection_alg": ["Trend_Classification()"],
                        "anomaly_detection_conf": [{
                            "num_samples": [500, 1000, 1500],        
                            "N": [30, 50, 70],                       
                            "averaging": [3, 5, 7],        
                            "prediction_conv": [5, 10, 15],   
                            "train_noise": [0.01, 0.05, 0.1],  
                            "amp_scale": [1.0, 2.0, 3.0]        
                        }]
                    },
                    "fixed_params": {
                        "warning_stages": [0.1, 0.2],
                        "input_vector_size": 1,
                        "output": ["TerminalOutput()"],
                        "output_conf": [{}]
                    }
                },

                "welford": { # Best parameters: {'LL': -1.2, 'N': 30, 'UL': 0.8, 'X': 4} Best F1 score: 0.813627254509018
                    "param_grid": {
                        "file_name": "data/ads-1.csv", 
                        "anomaly_detection_alg": ["Welford()"],
                        "anomaly_detection_conf": [{
                            "UL": [0.8, 1.0, 1.2],           
                            "LL": [-1.2, -1.0, -0.8],       
                            "X": [2, 3, 4],                
                            "N": [30, 50, 70]              
                        }]
                    },
                    "fixed_params": {
                        "input_vector_size": 1,
                        "warning_stages": [0.0, 0.0],
                        "filtering": "None",
                        "output": ["TerminalOutput()"],
                        "output_conf": [{}]
                    }
                }
            }
        

        # Selected algorithm
        selected_algorithm = "border_check"  # Change this to the desired algorithm
        config = param_options[selected_algorithm]

        # Flatten config["param_grid"]["anomaly_detection_conf"] into list of dicts
        conf_list = config["param_grid"]["anomaly_detection_conf"]
        flat_dicts = flatten_grid(conf_list[0])  # handle one dict with multiple params
        param_grid = merge_param_dicts(flat_dicts)

        # Set fixed params
        fixed_params.update(config.get("fixed_params", {}))

        model = AnomalyDetectorWrapper(
            anomaly_class=config["param_grid"]["anomaly_detection_alg"][0],
            fixed_params=fixed_params
        )

        # Run GridSearchCV
        grid = RandomizedSearchCV(
            model,
            param_distributions=param_grid,
            n_iter=20,  # Adjust for speed/coverage tradeoff
            scoring=custom_scorer,
            cv=2,
            n_jobs=-1,
            random_state=42
    )


        X_dummy = np.zeros((100, 1))
        grid.fit(X_dummy)

        for mean_score, params in zip(grid.cv_results_['mean_test_score'], grid.cv_results_['params']):
            print(f"Params: {params}, Mean F1 Score: {mean_score}")

        print("Best parameters:", grid.best_params_)
        print("Best F1 score:", grid.best_score_)

        end_time = time.time() 
        elapsed_time = end_time - start_time
        print(f"=== Program completed in {elapsed_time:.2f} seconds ===")
        exit(0)

    elif args.test:
        test_instance = Test(configuration_location=args.config)
        test_instance.read()
        test_instance.confusion_matrix()
        exit(0)

    else:
        consumer = ConsumerKafka(configuration_location=args.config)
        
    print("=== Service starting ===", flush=True)
    consumer.read()

def main() -> None:
    """
    Main entry point of the program.

    Returns:
        None
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    parser = argparse.ArgumentParser(description="consumer")

    parser.add_argument(
        "-c",
        "--config",
        dest="config",
        default="config1.json",
        help=u"Config file located in ./config/ directory."
    )

    parser.add_argument(
        "-f",
        "--file",
        dest="data_file",
        action="store_true",
        help=u"Read data from a specified file on specified location."
    )

    parser.add_argument(
        "-fk",
        "--filekafka",
        dest="data_both",
        action="store_true",
        help=u"Read data from a specified file on specified location and then from kafka stream."
    )

    parser.add_argument(
        "-w",
        "--watchdog",
        dest="watchdog",
        action='store_true',
        help=u"Ping watchdog",
    )

    parser.add_argument(
        "-t", 
        "--test", 
        dest="test", 
        default="config1.json",
        help=u"Config file located in ./config/ directory."
    )

    parser.add_argument(
        "-p", 
        "--param_tunning", 
        dest="param_tunning", 
        action="store_true",
        help="Perform grid search for hyperparameter tuning."
    )

    # Display help if no arguments are defined
    if (len(sys.argv) == 1):
        parser.print_help()
        sys.exit(1)

    # Parse input arguments
    args = parser.parse_args()

    # Ping watchdog every 30 seconds if specified
    if (args.watchdog):
        process = Process(target=start_consumer, args=(args,))
        process.start()
        print("=== Watchdog started ==", flush=True)
        ping_watchdog(process)
    else:
        start_consumer(args)


if (__name__ == '__main__'):
    main()
