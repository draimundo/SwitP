import cnn_vanilla
import utils
import learning_data
import os
import random as rn
import tensorflow as tf
import numpy as np
import sklearn.model_selection

# A path to re-sampled recordings which are organized into folders by user name.
data_path = r'../../data/processed_30Hz_relabeled'

# Path to where we want to save the training results
save_path = r'../../training_save_path'
save_path = os.path.abspath(save_path)

# A list of user names which are loaded.
users_ignore = []
users_all = utils.folders_in_path(data_path)
users = [u for u in users_all if u not in users_ignore]
users.sort()

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
model_parameters = cnn_vanilla.get_default_model_parameters()
model_parameters['labels'] = swimming_data.labels

# Parameters for training the CNN model
training_parameters = training_parameters = {'lr': 0.0005,
                                             'beta_1': 0.9,
                                             'beta_2': 0.999,
                                             'batch_size': 64,
                                             'max_epochs': 60,
                                             'steps_per_epoch': 1000,
                                             'noise_std': 0.01,  # Noise standard deviation for data augmentation
                                             'mirror_prob': 0.5,
                                             # Probability of reversing a window for data augmentation
                                             'random_rot_deg': 30,
                                             # [-30, 30] is the range of rotation degrees we sample for each
                                             # window in the mini-batch
                                             'group_probs': {'original': 0.7, 'time_scaled_0.9': 0.15,
                                                             'time_scaled_1.1': 0.15},
                                             'labels': swimming_data.labels
                                             }

# The input shape of the CNN
input_shape = (data_parameters['win_len'], len(data_parameters['data_columns']), 1)

# All seeds used before training
os.environ['PYTHONHASHSEED'] = '0'
rn.seed(1337)
np.random.seed(1337)
tf.set_random_seed(1337)

# Preparing folds for cross-validation
kf = sklearn.model_selection.KFold(n_splits=10)

i = 0
for train_idx, test_idx in kf.split(users):
    # actual split
    i = i + 1

    # Reset session
    tf.keras.backend.clear_session()
    sess = tf.Session(graph=tf.get_default_graph())
    tf.keras.backend.set_session(sess)

    model = cnn_vanilla.cnn_model(input_shape, model_parameters)

    # Choose test and train splits
    users_train = [users[i] for i in train_idx]
    users_test = [users[i] for i in test_idx]

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

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(lr=training_parameters['lr'], beta_1=training_parameters['beta_1'],
                                         beta_2=training_parameters['beta_2'])

    # Plot the evolution of the model
    tb_dir = os.path.join(save_path, str(i))
    os.mkdir(tb_dir)
    tb = tf.keras.callbacks.TensorBoard(log_dir=tb_dir, histogram_freq=0, batch_size=training_parameters['batch_size'],
                                        write_graph=True,
                                        write_grads=False, write_images=False, embeddings_freq=0,
                                        embeddings_layer_names=None, embeddings_metadata=None,
                                        embeddings_data=None, update_freq='epoch')

    # Early stopping
    es = tf.keras.callbacks.EarlyStopping(monitor='val_weighted_acc', mode='max', verbose=1, patience=200)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'], weighted_metrics=['acc'])

    # Get the validation data
    x_val, y_val_cat, val_sample_weights = swimming_data.get_windows_dict(val_dict, return_weights=True)
    x_val = x_val.reshape((x_val.shape[0], x_val.shape[1], x_val.shape[2], 1))

    # Train the model
    history = model.fit(x=gen,
                        validation_data=(x_val, y_val_cat, val_sample_weights),
                        steps_per_epoch=training_parameters['steps_per_epoch'],
                        epochs=training_parameters['max_epochs'],
                        callbacks=[tb])
