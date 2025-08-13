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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np

from itertools import product



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
    that can be used in RandomizedSearchCV.
    """

    keys, values = zip(*conf_dict.items())
    combos = [dict(zip(keys, v)) for v in product(*values)]
    return combos

def merge_param_dicts(dicts):
    """Merge list of dicts into a dict of lists for RandomizedSearchCV."""
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

        param_grid = {}

        fixed_params = {
            "input_vector_size": 1,
            "output": ["TerminalOutput()"],
            "output_conf": [{}],
        }

        param_options = {
                #Best parameters: {'UL': 3.0, 'LL': -0.4}
                #Best F1 score: 0.6341463414634146
                #=== Program completed in 109.75 seconds ===
                # can't go further 
                "border_check": {
                    "param_grid": {
                        "file_name": "data/ads-1.csv",
                        "anomaly_detection_alg": ["BorderCheck()"],
                        "anomaly_detection_conf": [{
                            "UL": [2.6, 2.8, 3.0, 3.2, 3.4],
                            "LL": [-0.6, -0.5, -0.4, -0.3, -0.2]
                        }]
                    },
                    "fixed_params": {"warning_stages": [0.0, 0.0]}
                },

                #Best parameters: {'treshold': 0.4, 'samples_for_retrain': 100, 'retrain_interval': 200, 'min_samples': 10, 'eps': 1.0}
                #Best F1 score: 0.043254034327372404
                #=== Program completed in 659.01 seconds ===
                "clustering": {  
                    "param_grid": {
                        "file_name": "data/ads-1.csv",
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

                #Best parameters: {'decay': 0.4, 'averaging': 35, 'UL': 0.7, 'LL': -1.0}
                #Best F1 score: 0.03713527851458886
                #=== Program completed in 92.64 seconds ===
                # can't go further 
                "cumulative": {
                    "param_grid": {
                        "file_name": "data/ads-1.csv",
                        "anomaly_detection_alg": ["Cumulative()"],
                        "anomaly_detection_conf": [{
                           "decay": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                            "averaging": [5, 10, 15, 20, 25, 30, 35],
                            "UL": [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0],
                            "LL": [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5]              
                        }]
                    },
                    "fixed_params": {"warning_stages": [0.0, 0.0]}
                },

                #Best parameters: {'window': 20, 'start_on': 20, 'period': 30, 'percentile': 5}
                #Best F1 score: 0.01749068636736756
                #=== Program completed in 124.34 seconds ===
                #Best parameters: {'percentile': 3, 'period': 15, 'start_on': 20, 'window': 20}
                #Best F1 score: 0.01749068636736756
                #=== Program completed in 899.95 seconds === with GridSearchCV
                # can't go further 
                "ema_percentile": {
                    "param_grid": {
                        "file_name": "data/ads-1.csv",
                        "anomaly_detection_alg": ["EMA_Percentile()"],
                        "anomaly_detection_conf": [{
                            "percentile": [3, 5, 7],     
                            "window": [10, 20, 30],      
                            "start_on": [10, 15, 20], 
                            "period": [15, 20, 25],        
                            
                        }]
                    },
                    "fixed_params": {
                    }
                },

                #Best parameters: {'UL': 0.55, 'N': 3, 'LL': -0.45}
                #Best F1 score: 0.0381791483113069
                #=== Program completed in 48.13 seconds ===
                "ema": {
                    "param_grid": {
                        "file_name": "data/ads-1.csv", 
                        "anomaly_detection_alg": ["EMA()"],
                        "anomaly_detection_conf": [{
                            "N": [3,4,5, 6],
                            "UL": [0.45, 0.5, 0.55],
                            "LL": [-0.55, -0.5, -0.45]
                        }]
                    },
                    "fixed_params": {"warning_stages": [0.0, 0.0]}
                },

                "fb_prophet": {},

                # Best parameters: {'LL': -1.2, 'UL': 0.25, 'cutoff_frequency': 0.22, 'filter_order': 5, 'mode': 1}
                # Best F1 score: 0.4034090909090909
                # === Program completed in 411.84 seconds ===
                #Can't go further
                "filtering": {
                    "param_grid": {
                        "file_name": "data/ads-1.csv",
                        "anomaly_detection_alg": ["Filtering()"],
                        "anomaly_detection_conf": [{
                            # Only change the following parameters to find better value
                            "mode": [1],  # keep original modes
                            "filter_order": [4, 5, 6],  
                            "cutoff_frequency": [0.20, 0.22, 0.24],  
                            "UL": [0.20, 0.25, 0.30],  
                            "LL": [-1.3, -1.2, -1.1] 
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
                            "train_data": ["data/ads-1.csv"],
                            "model_name": ["GAN_sensor_cleaning"],
                           "N_shifts": [0, 1],           # Keep 0 (best), test 1
                           "N_latent": [3, 4],           # 4 is best, try one below
                           "K": [0.4, 0.45],             # Best was 0.4, try one close
                           "len_window": [250, 300],     # 250 was best, small step
                        }]
                    },
                    "fixed_params": {
                        "warning_stages": [0.0, 0.0]
                    }
                },

                #Best parameters: {'K': 2.0, 'W': 12, 'n_sigmas': 2.0}
                #Best F1 score: 0.021897810218978103
                #=== Program completed in 168.53 seconds ===
                #Can  go further
                "hampel": {
                    "param_grid": {
                        "file_name": "data/ads-1.csv",
                        "anomaly_detection_alg": ["Hampel()"],
                        "anomaly_detection_conf": [{
                            "n_sigmas": [2.0, 3.0, 4.0],   # Center around your best
                            "W": [10, 12, 13],
                            "K": [2.0, 3.0, 4.0]  # Keep 1.0, test 0.5 and 1.2
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


                #Best parameters: {'confidence_norm': 0.2, 'UL': 0.8, 'N': 2, 'LL': -1.0}
                #Best F1 score: 0.5822021116138762
                #=== Program completed in 66.33 seconds ===
                "linear_fit": {
                    "param_grid": {
                        "file_name": "data/ads-1.csv",
                        "anomaly_detection_alg": ["LinearFit()"],
                        "anomaly_detection_conf": [{
                            "UL": [0.6, 0.8, 1.0, 1.2, 2.0, 3.0],
                            "LL": [-1.2, -1.0, -0.8, -0.5, -0.2],
                            "confidence_norm": [0.05, 0.1, 0.2, 0.3],
                            "N": [1, 2, 3, 4, 5, 6]
                        }]
                    },
                    "fixed_params": {
                        "input_vector_size": 1,
                        "output": ["TerminalOutput()"],
                        "output_conf": [{}],
                        "warning_stages": [0.0, 0.0],
                    }
                },

                #Best parameters: {'period2': 32, 'period1': 6, 'UL': 1.0, 'LL': -0.8}
                #Best F1 score: 0.1916932907348243
                #=== Program completed in 49.68 seconds ===
                #Cant go further
                "macd": {
                    "param_grid": {
                        "file_name": "data/ads-1.csv",
                        "anomaly_detection_alg": ["MACD()"],
                        "anomaly_detection_conf": [{
                            "period1": [4, 6, 8, 12, 16],          
                            "period2": [20, 26, 32, 38],         
                            "UL": [0.4, 0.6, 0.8, 1.0, 1.2],     
                            "LL": [-1.5, -1.0, -0.8, -0.6, -0.5],      
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

                #Best parameters: {'tree_size': 512, 'threshold': 38, 'num_trees': 50}
                #Best F1 score: 0.4774998033199591
                #=== Program completed in 1727.65 seconds ===
                "rrcf_trees": { # very slow
                    "param_grid": {
                        "file_name": "data/ads-1.csv",
                        "anomaly_detection_alg": ["RRCF_trees()"],
                        "anomaly_detection_conf": [{
                            "num_trees": [20, 25, 50],   
                            "tree_size": [512, 1024],
                            "threshold":  [36, 38],
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


                #Best parameters: {'train_noise': 0.01, 'prediction_conv': 5, 'num_samples': 1500, 'averaging': 3, 'amp_scale': 1.5, 'N': 20}
                #Best F1 score: 0.049450056353716576
                #=== Program completed in 2686.35 seconds ===

                "trend_classification": { # very slow
                    "param_grid": {
                        "file_name": "data/ads-1.csv",
                        "anomaly_detection_alg": ["Trend_Classification()"],
                        "anomaly_detection_conf": [{
                            "num_samples": [1000, 1500],           
                            "N": [20, 25],                          
                            "averaging": [2, 3],                     
                            "prediction_conv": [5, 10],              
                            "train_noise": [0.01, 0.02],        
                            "amp_scale": [1.0, 1.5]              
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
        selected_algorithm = "rrcf_trees"  # Change this to the desired algorithm
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

        grid = RandomizedSearchCV(
            model,
            param_distributions=param_grid,
            n_iter=5,
            scoring=custom_scorer,
            cv=2,
            n_jobs=-1,
            random_state=42
        )

        # grid = GridSearchCV(
        #     model,
        #     param_grid=param_grid,   # param_grid instead of param_distributions
        #     scoring=custom_scorer,
        #     cv=2,
        #     n_jobs=-1,
        # )


        X_dummy = np.zeros((100, 1))
        grid.fit(X_dummy)

        for mean_score, params in zip(grid.cv_results_['mean_test_score'], grid.cv_results_['params']):
            print(f"Params: {params}, Mean F1 Score: {mean_score}")

        print("#Best parameters:", grid.best_params_)
        print("#Best F1 score:", grid.best_score_)

        end_time = time.time() 
        elapsed_time = end_time - start_time
        print(f"#=== Program completed in {elapsed_time:.2f} seconds ===")
        exit(0)

    elif args.test:
        test_instance = Test(configuration_location=args.config)
        test_instance.read()
        test_instance.confusion_matrix()
        return test_instance

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
