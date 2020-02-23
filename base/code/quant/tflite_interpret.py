# Similar to tutorial_load, works with a tflite model

import numpy as np
import tensorflow as tf
import utils
import learning_data
import pickle
import matplotlib.pyplot as plt

from sklearn.metrics import log_loss

import platform
if platform.system() == 'Darwin':
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']


# A path to re-sampled recordings which are organized into folders by user name.
data_path = r'../../data/processed_30hz_relabeled'

# Path to training results
results_path = r'../../tutorial_save_path'

# User whose model we want to load
user = '7'

# Get the data parameters used for loading
with open(os.path.join(results_path, user, 'data_parameters.pkl'), 'rb') as f:
    data_parameters = pickle.load(f)[0]

# Load the recordings
swimming_data = learning_data.LearningData()
swimming_data.load_data(data_path=data_path,
                        data_columns=data_parameters['data_columns'],
                        users=[user],
                        labels=data_parameters['labels'])
if data_parameters['combine_labels'] is not None:
    for label in data_parameters['combine_labels'].keys():
        swimming_data.combine_labels(labels=data_parameters['combine_labels'][label], new_label=label)
swimming_data.sliding_window_locs(win_len=data_parameters['win_len'], slide_len=data_parameters['slide_len'])

recs = list(swimming_data.data_dict['original'][user].keys())
prediction_traces = {rec: None for rec in recs}
for (ii, rec) in enumerate(recs):
    print("Recording %d of %d" % (ii + 1, len(recs)))
    win_starts = swimming_data.window_locs['original'][user][rec][0]
    win_stops = swimming_data.window_locs['original'][user][rec][1]
    windows = np.zeros((len(win_starts), swimming_data.win_len, len(swimming_data.data_columns)))
    y_true_windows = np.zeros((len(windows), 5))
    y_true_windows_maj = np.zeros(len(windows))

    y_pred_windows = np.zeros((len(windows), 5))


    for iii in range(len(win_starts)):
        win_start = win_starts[iii]
        win_stop = win_stops[iii]
        window = swimming_data.data_dict['original'][user][rec][swimming_data.data_columns].values[
                 win_start:win_stop + 1, :]
        window_norm = swimming_data.normalize_window(window, norm_type=data_parameters['window_normalization'])
        windows[iii] = window_norm
        win_labels = swimming_data.data_dict['original'][user][rec]['label'].values[win_start: win_stop + 1]
        win_label_cat, majority_label = swimming_data.get_window_label(win_labels, label_type='proportional',
                                                                       majority_thresh=0.4)
        y_true_windows[iii, :] = win_label_cat
        y_true_windows_maj[iii] = majority_label

        windowint = np.array(window, dtype=np.float32).reshape((1, 180, 11, 1))

        interpreter.set_tensor(input_details[0]['index'], windowint)
        interpreter.invoke()
        y_pred_windows[iii] = interpreter.get_tensor(output_details[0]['index'])

    windows = windows.reshape((windows.shape[0], windows.shape[1], windows.shape[2], 1))

    # y_pred_windows = model.predict(windows)

    y_true_raw = swimming_data.data_dict['original'][user][rec]['label'].values
    win_mids = win_starts + (win_stops - win_starts) / 2
    x = win_mids
    y = y_pred_windows
    x_new = np.arange(0, len(y_true_raw))
    y_pred_raw = utils.resample(x, y.T, x_new, kind='nearest').T
    prediction_traces[rec] = {'window': {'true': y_true_windows, 'pred': y_pred_windows},
                              'raw': {'true': y_true_raw, 'pred': y_pred_raw}}

    y_pred_windowsk = np.zeros((y_pred_windows.shape[0], y_pred_windows.shape[1]+1))
    y_pred_windowsk[:,:-1] = np.nan_to_num(y_pred_windows)
    print("  Log-loss: %f" % log_loss(np.nan_to_num(y_true_windows_maj), y_pred_windowsk,labels=data_parameters['labels']))

for (i, rec) in enumerate(recs):
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=False)
    fig.suptitle("recording" + str(i), fontsize=16)
    ax[0].plot(swimming_data.data_dict['original'][user][rec]['ACC_0'].values)
    ax[0].plot(swimming_data.data_dict['original'][user][rec]['ACC_1'].values)
    ax[0].plot(swimming_data.data_dict['original'][user][rec]['ACC_2'].values)
    yt = np.argmax(prediction_traces[rec]['window']['true'], axis=1)
    ax[1].plot(yt)
    yp = np.argmax(prediction_traces[rec]['window']['pred'], axis=1)
    ax[1].plot(yp)
plt.show()