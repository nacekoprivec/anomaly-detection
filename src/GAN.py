from typing import Any, Dict, List
import numpy as np
import sys
import json
from pandas.core.frame import DataFrame
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from ast import literal_eval

sys.path.insert(0,'./src')
sys.path.insert(1, 'C:/Users/Matic/SIHT/anomaly_det/anomalyDetection/')
from anomalyDetection import AnomalyDetectionAbstract
from isolationForest import IsolationForest
from output import OutputAbstract, TerminalOutput, FileOutput, KafkaOutput
from visualization import VisualizationAbstract, GraphVisualization,\
    HistogramVisualization, StatusPointsVisualization
from normalization import NormalizationAbstract, LastNAverage,\
    PeriodicLastNAverage

class GAN(AnomalyDetectionAbstract):
    name: str = "GAN"

    N_shifts: int
    N_latent: int
    GAN_error: List[float]

    isolation_forest: "IsolationForest"

    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(conf)

    def configure(self, conf: Dict[Any, Any] = None,
                  configuration_location: str = None,
                  algorithm_indx: int = None) -> None:
        super().configure(conf, configuration_location=configuration_location,
                          algorithm_indx=algorithm_indx)

        # Train configuration
        self.N_shifts = conf["train_conf"]["N_shifts"]
        self.N_latent = conf["train_conf"]["N_latent"]
        self.model_name = conf["train_conf"]["model_name"]
        self.K = conf["train_conf"]["K"]

        # Retrain configuration
        if("retrain_interval" in conf):
            self.retrain_counter = 0
            self.retrain_interval = conf["retrain_interval"]
            self.retrain_file = conf["retrain_file"]
            self.samples_from_retrain = 0
            if("samples_for_retrain" in conf):
                self.samples_for_retrain = conf["samples_for_retrain"]
            else:
                self.samples_for_retrain = None

            # Retrain memory initialization
            # Retrain memory is of shape [timestamp, ftr_vector]
            if("train_data" in conf):

                """df_ = pd.read_csv(conf["train_data"], skiprows=1, delimiter = ",", usecols = (0, 1,)).values
                vals = df_[:,1]
            
                #values = np.lib.stride_tricks.sliding_window_view(values, (self.input_vector_size))
                values = [vals[x:x+self.input_vector_size] for x in range(len(vals) - self.input_vector_size + 1)]
    
                timestamps = [df_[:,0][-len(values):]]
                df = np.concatenate((np.array(timestamps).T,values), axis=1)

                self.memory_dataframe = pd.DataFrame(df, index = None)"""

                self.memory_dataframe = pd.read_csv(conf["train_data"],
                                                    skiprows=0,
                                                    delimiter=",",
                                                    converters={"ftr_vector": literal_eval})

                if(self.samples_for_retrain is not None):
                    self.memory_dataframe = self.memory_dataframe.iloc[-self.samples_for_retrain:]
            else:
                columns = ["timestamp", "ftr_vector"]
                self.memory_dataframe = pd.DataFrame(columns=columns)
        else:
            self.retrain_interval = None
            self.samples_for_retrain = None
            self.memory_dataframe = None

        # Initialize model
        if("load_model_from" in conf):
            self.load_model(conf["load_model_from"])
        elif("train_data" in conf):
            self.train_model(train_file = conf["train_data"])
        else:
            raise Exception("Model or train dataset must be specified to\
                            initialize model.")

    def message_insert(self, message_value: Dict[Any, Any]) -> None:
        if(self.min != self.max):
            message_value['ftr_vector'] = list((np.array(message_value['ftr_vector'])- self.avg)/(self.max - self.min))

        super().message_insert(message_value)

        # Check feature vector
        if(not self.check_ftr_vector(message_value=message_value)):
            status = self.UNDEFINED
            status_code = self.UNDEFIEND_CODE
            self.normalization_output_visualization(status=status,
                                                status_code=status_code,
                                                value=message_value["ftr_vector"],
                                                timestamp=message_value["timestamp"])
            
            # Remenber status for unittests
            self.status = status
            self.status_code = status_code
            return

        value = message_value["ftr_vector"]
        value = value[0]

        if(self.use_cols is not None):
            value = []
            for el in range(len(message_value["ftr_vector"])):
                if(el in self.use_cols):
                    value.append(message_value["ftr_vector"][el])
        else:
            value = message_value["ftr_vector"]

        timestamp = message_value["timestamp"]

        feature_vector = list(value)

        if (feature_vector == False):
            # If this happens the memory does not contain enough samples to
            # create all additional features.

            # Send undefined message to output
            for output in self.outputs:
                output.send_out(timestamp=message_value['timestamp'],
                                value=None)
            
            # And to visualization
            if(self.visualization is not None):
                lines = [value[-1]]
                #self.visualization.update(value=[None], timestamp=timestamp,
                #                          status_code=2)
            return
        else:
            feature_vector = np.array(feature_vector)
            # print(feature_vector)
            #Model prediction
            prediction = self.GAN.predict(feature_vector.reshape(1, self.N_shifts+1))[0]
            self.GAN_error = self.mse(np.array(prediction),np.array(feature_vector))


            #print("GAN error: " + str(self.GAN_error))
            #IsolationForest_transformed =  self.IsolationForest.predict(self.GAN_error.reshape(-1, 1))
            
            if(self.GAN_error < self.threshold):
                status = self.OK
                status_code = self.OK_CODE
            elif(self.GAN_error >= self.threshold):
                status = "Error: outlier detected (GAN)"
                status_code = -1
            else:
                status = self.UNDEFINED
                status_code = self.UNDEFIEND_CODE

            self.normalization_output_visualization(status=status,
                                                    status_code=status_code,
                                                    value=value,
                                                    timestamp=timestamp)
        
        self.status = status
        self.status_code = status_code

        # Add to memory for retrain and execute retrain if needed 
        if (self.retrain_interval is not None):
            # Add to memory
            samples_in_memory = self.memory_dataframe.shape[0]

            self.memory_dataframe.at[samples_in_memory, "timestamp"] = timestamp
            self.memory_dataframe.at[samples_in_memory, "ftr_vector"] = value
            
            # Cut if needed
            if(self.samples_for_retrain is not None):
                self.memory_dataframe = self.memory_dataframe.iloc[-self.samples_for_retrain:]

            self.samples_from_retrain += 1

            # Retrain if needed (and possible)
            if(self.samples_from_retrain >= self.retrain_interval and
                self.samples_for_retrain == self.memory_dataframe.shape[0]):
                self.samples_from_retrain = 0
                self.train_model(train_dataframe=self.memory_dataframe)
                self.retrain_counter +=1
            return

    @staticmethod
    def mse(pre_GAN, post_GAN):
        #mean squared error - loss
        mse = np.sum((np.add(np.array(pre_GAN), -np.array(post_GAN))**2))/float(len(pre_GAN))
        return(mse)

    def save_model(self, filename):
        self.GAN.save("models/" + filename + "_GAN")
        #print("Saving GAN")

    def load_model(self, filename):
        self.GAN = keras.models.load_model(filename + "_GAN")

    def train_model(self, train_file: str = None, train_dataframe: DataFrame = None) -> None:
        if(train_dataframe is not None):
            # This is in case of retrain
            df = train_dataframe

            # Save train_dataframe to file and change the config file so the
            # next time the model will train from that file
            path = self.retrain_file
            df.to_csv(path,index=False)

            with open("configuration/" + self.configuration_location) as conf:
                whole_conf = json.load(conf)
                whole_conf["anomaly_detection_conf"][self.algorithm_indx]["train_data"] = path
            
            with open("configuration/" + self.configuration_location, "w") as conf:
                json.dump(whole_conf, conf)

            # Extract list of ftr_vectors and list of timestamps
            ftr_vector_list = df["ftr_vector"].tolist()
            timestamp_list = df["timestamp"].tolist()

            # Create a new  dataframe with features as columns
            df = pd.DataFrame.from_records(ftr_vector_list)
            df.insert(loc=0, column="timestamp", value=timestamp_list)
            # Transfer to numpy
            df = df.to_numpy()

        elif(train_file is not None):
            df_ = pd.read_csv(train_file, skiprows=0, delimiter = ",", usecols = (0, 1,), converters={'ftr_vector': literal_eval})
            vals = df_['ftr_vector'].values
            vals = np.array([np.array(xi) for xi in vals])
            self.min = min(min(vals, key=min))
            self.max = max(max(vals, key=max))
            self.avg = (self.min + self.max)/2

            if(self.min != self.max):
                values = (np.array(vals) - self.avg)/(self.max - self.min)
            else:
                values = np.array(vals)
            
            #values = np.lib.stride_tricks.sliding_window_view(values, (self.input_vector_size))
            #values = [vals[x:x+self.input_vector_size] for x in range(len(vals) - self.input_vector_size + 1)]
            timestamps = np.array(df_['timestamp'].values)
            timestamps = np.reshape(timestamps, (-1, 1))
            df = np.concatenate([timestamps,values], axis = 1)
        else:
            raise Exception("train_file or train_dataframe must be specified.")

        timestamps = np.array(df[:,0])
        data = np.array(df[:,1:(1 + self.input_vector_size)])

        # Requires special feature construction so it does not mess with the
        # feature-construction memory
        features = self.training_feature_construction(data=data,
                                                      timestamps=timestamps)


        # Fit IsolationForest model to data (if there was enoug samples to
        # construct at leat one feature)
        if(len(features) > 0):

            original_dim = np.prod(np.array(features).shape [1:]) # dimenzija vhodnih podatkov
            hidden_dim = 10 # skriti sloj z 64 node -i
            latent_dim = self.N_latent # 2D latentni prostor
            inputs = keras.Input(shape =(original_dim ,))
            h1 = keras.layers.Dense(hidden_dim, activation ='linear')(inputs)
            h2 = keras.layers.Dense(hidden_dim, activation ='tanh')(h1)
            h3 = keras.layers.Dense(hidden_dim, activation ='tanh')(h2)

            h4 = keras.layers.Dense(latent_dim, activation ='tanh')(h3)

            encoder = keras.Model(inputs, outputs = [h4, h4], name = 'encoder')
            latent_inputs = keras.Input(shape =latent_dim, name ='z_sampling')

            x1 = keras.layers.Dense(hidden_dim , activation ='tanh')(latent_inputs)
            x2 = keras.layers.Dense(hidden_dim , activation ='relu')(x1)
            x3 = keras.layers.Dense(hidden_dim , activation ='relu')(x2)

            outputs = keras.layers.Dense(original_dim , activation ='linear')(x3)
            decoder = keras.Model(latent_inputs, outputs, name ='decoder')

            outputs = decoder(encoder(inputs)[0])
            self.GAN = keras.Model(inputs, outputs, name ='vae')
            mse = tf.keras.losses.MeanSquaredError()

            GAN_loss = mse(inputs, outputs)
            self.GAN.add_loss(GAN_loss)
            self.GAN.compile(optimizer =tf.keras.optimizers.Adam(lr = 0.001, beta_1 = 0.95))
            features = np.array(features)
            self.GAN.fit(features,features, epochs =100, batch_size = 100, validation_data = None, verbose = 0)
            
            predictions = self.GAN.predict(features.reshape(len(features), self.N_shifts+1))
            
            GAN_transformed = [mse(np.array(features[i]), predictions[i]) for i in range(len(features))]
            self.threshold = self.K * max(GAN_transformed)

            self.save_model(self.model_name)
