import tensorflow as tf
import cnn_vanilla
import os
import utils
import learning_data

import random as rn
import numpy as np

# https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1_train.py


# A path to re-sampled recordings which are organized into folders by user name.
data_path = '../../data/processed_30Hz_relabeled'

# Path to where we want to save the training results
save_path = '../../quant_aware_path'

# A list of user names which are loaded.
users_ignore = []
users_all = utils.folders_in_path(data_path)
users = [u for u in users_all if u not in users_ignore]
users.sort()

# Keeping it simple. Comment this out and use the code above if you want to load everybody
users = ['1','2','3','4','5','6','7','8','9']

# List of users we want to train a model for
users_test = ['1']

# Hyper-parameters for loading data.
data_parameters = {'users': users,  # Users whose data is loaded
                   'labels': [0, 1, 2, 3, 4, 5],  # Labels we want to use
                   'combine_labels': {0: [0, 5]},  # Labels we want to combine. Here I am combining NULL and
                   # TURN into NULL
                   'data_columns': ['ACC_0', 'ACC_1', 'ACC_2', 'GYRO_0', 'GYRO_1', 'GYRO_2', 'MAG_0',
                                    'MAG_1', 'MAG_2', 'PRESS', 'LIGHT'],  # The sensor data we want to load
                   'time_scale_factors': [0.9, 1.1],  # time-scaling factors we want to use. A copy is made of each
                   # recording with these factors.
                   'win_len': 180,  # The length of the segmentation window in number of samples
                   'slide_len': 30,  # The slide length used for segmentation
                   'window_normalization': 'statistical',  # How we want to normalize the windows. Statistical means
                   # zero-mean and unit variance for each signal
                   'label_type': 'majority',  # How we label windows.
                   'majority_thresh': 0.75,  # Proportion of samples in a window that have to have the same label
                   'validation_set': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1},  # The number of users that represent each
                   # class in the validation set
                   }

# Data is loaded and stored in this object
swimming_data = learning_data.LearningData()

# Load recordings from data_path. Recordings are stored under
# swimming_data.data_dict['original][user_name][recording_name] which is a Pandas DataFrame
swimming_data.load_data(data_path=data_path,
                        data_columns=data_parameters['data_columns'],
                        users=data_parameters['users'],
                        labels=data_parameters['labels'])

# Combine labels
if data_parameters['combine_labels'] is not None:
    for label in data_parameters['combine_labels'].keys():
        swimming_data.combine_labels(labels=data_parameters['combine_labels'][label], new_label=label)

# Data augmentation for recordings. This is only for time-scaling. Other data augmentations happen during the learning
# Stored under swimming_data['time_scaled_1.1'][user_name]...
swimming_data.augment_recordings(time_scale_factors=data_parameters['time_scale_factors'])

# Compute the locations of the sliding windows in each recording
swimming_data.sliding_window_locs(win_len=data_parameters['win_len'], slide_len=data_parameters['slide_len'])

# Compile the windows. Stored under swimming_data.data_windows[group][label][user]['data' or 'label']
# Recordings are still stored under swimming_data.data_dict so a lot of memory might be needed
swimming_data.compile_windows(norm_type=data_parameters['window_normalization'],
                              label_type=data_parameters['label_type'],
                              majority_thresh=data_parameters['majority_thresh'])

# Parameters for the CNN model
model_parameters = {'filters': [64, 64, 64, 64],
                    'kernel_sizes': [3, 3, 3, 3],
                    'strides': [None, None, None, None],
                    'max_pooling': [3, 3, 3, 3],
                    'units': [128],
                    'activation': ['elu', 'elu', 'elu', 'elu', 'elu'],
                    'batch_norm': [False, False, False, False, False],
                    'drop_out': [0.5, 0.75, 0.25, 0.1, 0.25],
                    'max_norm': [0.1, 0.1, None, 4.0, 4.0],
                    'l2_reg': [None, None, None, None, None],
                    'labels': swimming_data.labels
                    }

# Parameters for training the CNN model
training_parameters = {'lr': 0.0005,
                       'beta_1': 0.9,
                       'beta_2': 0.999,
                       'batch_size': 64,
                       'max_epochs': 500,  # Keeping small for quick testing
                       'steps_per_epoch': 100,  # Keeping small for quick testing
                       'noise_std': 0.01,  # Noise standard deviation for data augmentation
                       'mirror_prob': 0.5,  # Probability of reversing a window for data augmentation
                       'random_rot_deg': 30,  # [-30, 30] is the range of rotation degrees we sample for each
                       # window in the mini-batch
                       'group_probs': {'original': 0.7, 'time_scaled_0.9': 0.15, 'time_scaled_1.1': 0.15},
                       'labels': swimming_data.labels
                       }

# The input shape of the CNN
input_shape = (data_parameters['win_len'], len(data_parameters['data_columns']), 1)

for (i, user_test) in enumerate(users_test):

    # Path for saving results
 #   print("Running experiment: %s" % user_test)
 #   experiment_save_path = os.path.join(save_path, user_test)
 #   if os.path.exists(experiment_save_path):
 #       print("Skipping: %s" % user_test)
 #       continue
 #   else:
 #       os.mkdir(experiment_save_path)

    model = cnn_vanilla.cnn_model(input_shape, model_parameters)

    # Quantize model
    sess = tf.keras.backend.get_session()
    tf.contrib.quantize.create_training_graph(input_graph=sess.graph, quant_delay=2000000)
    sess.run(tf.global_variables_initializer())

    # Users whose data we use for training
    users_train = [u for u in users if u != user_test]

    # Draw users for each class. train_dict and val_dict are dictionaries whose keys are labels and they contain
    # lists of names for each label
    train_dict, val_dict = swimming_data.draw_train_val_dicts(users_train,
                                                              users_per_class=data_parameters['validation_set'])

    print("Validation dictionary: %s" % val_dict)

    # The generator used to draw mini-batches
    gen = swimming_data.batch_generator_dicts(train_dict=train_dict,
                                              batch_size=training_parameters['batch_size'],
                                              noise_std=training_parameters['noise_std'],
                                              mirror_prob=training_parameters['mirror_prob'],
                                              random_rot_deg=training_parameters['random_rot_deg'])

    optimizer = tf.compat.v1.keras.optimizers.Adam(lr=training_parameters['lr'], beta_1=training_parameters['beta_1'],
                                                   beta_2=training_parameters['beta_2'])

    best_path = os.path.join(experiment_save_path, 'model_best.h5')

    # Which model is the best model and where we save it
    cp = tf.keras.callbacks.ModelCheckpoint(best_path, verbose=1, monitor='acc_1',
                                            save_best_only=True, mode='max')
    # Plot the evolution of the model
    logs_dir = os.path.join(save_path, 'logs')
    tb = tf.keras.callbacks.TensorBoard(log_dir=logs_dir, histogram_freq=0,
                                        batch_size=training_parameters['batch_size'],
                                        write_graph=True,
                                        write_grads=False, write_images=False, embeddings_freq=0,
                                        embeddings_layer_names=None, embeddings_metadata=None,
                                        embeddings_data=None, update_freq='epoch')

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'], weighted_metrics=['acc'])

    # Get the validation data
    x_val, y_val_cat, val_sample_weights = swimming_data.get_windows_dict(val_dict, return_weights=True)
    x_val = x_val.reshape((x_val.shape[0], x_val.shape[1], x_val.shape[2], 1))

    # Train the model
    history = model.fit_generator(generator=gen, validation_data=(x_val, y_val_cat, val_sample_weights),
                                  epochs=training_parameters['max_epochs'],
                                  steps_per_epoch=training_parameters['steps_per_epoch'])

    K.clear_session()
# TODO quantize, train and evaluate
