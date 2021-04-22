# Takes in a trained model and returns quantized versions (tfLite and coreml)
import cnn_vanilla
import utils
import learning_data
import os
import random as rn
from tensorflow import keras
import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt


# macOS specific error workaround
import platform
if platform.system() == 'Darwin':
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# A path to re-sampled recordings which are organized into folders by user name.
data_path = r'../../data/processed_30hz_relabeled'

# Path to training results
results_path = r'../../training_save_path_accgyrmag'

# User whose model we want to load
user = '7'

# Get the data parameters used for loading
#with open(os.path.join(results_path, user, 'data_parameters.pkl'), 'rb') as f:
#    data_parameters = pickle.load(f)[0]

# Fetch the model
keras_model_path = os.path.join(results_path, 'model_best.h5')
keras_model = keras.models.load_model(keras_model_path)

# get input, output node names for the TF graph from the Keras model
input_name = keras_model.inputs[0].name.split(':')[0]
keras_output_node_name = keras_model.outputs[0].name.split(':')[0]
graph_output_node_name = keras_output_node_name.split('/')[-1]

converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_model_path)
tflite_model = converter.convert()
open(os.path.join(results_path, 'converted_model.tflite'), "wb").write(tflite_model)

# For coreml models (iOS)
#import tfcoreml
#coreml_model = tfcoreml.convert(os.path.join(results_path, user, 'model_best.h5'),
#                         input_name_shape_dict={input_name: (1, 180, 11, 1)},
#                         output_feature_names=[graph_output_node_name],
#                         minimum_ios_deployment_target='13')#

#coreml_model.save('./coreml_model.mlmodel')