import numpy as np
import pandas as pd
import os
import csv
import scipy.interpolate
import scipy.stats
import constants
import scipy.signal as sp


def write_latex_confmat(cm, labels, is_integer=False):
    """
    Write confusion matrix into latex table
    :param cm: Two-dimensional confusion matrix
    :param labels: The labels in the confusion matrix
    :param is_integer: A boolean set to True if the values are integers
    :return: The confusion matrix in latex form
    """
    header = '\\begin{tabular}{' + 'c' * (len(labels) + 1) + '}\n'
    footer = '\\end{tabular}'
    text = header
    text = text + '&' + '&'.join(label for label in labels) + '\\\\\n'
    for (i, label) in enumerate(labels):
        if is_integer:
            new_line = label + '&' + '&'.join([str(int(v)) for v in cm[i, :]]) + '\\\\\n'
        else:
            new_line = label + '&' + '&'.join(['%.1f' % v for v in cm[i, :]]) + '\\\\\n'
        text = text + new_line
    text = text + footer
    return text


def downsample_dataframe(df, skipstep):
    """
    Down-sample a pandas dataframe. Mainly used for plots.
    :param df: The pandas dataframe
    :param skipstep: The down-sampling factor
    :return:
    """
    df_new = df.iloc[::skipstep]
    x = np.arange(len(df['label'].values))
    x_new = x[::skipstep]
    for (i, col) in enumerate(df.columns):
        y = df[col].values
        if col == 'timestamp':
            continue
        if col == 'label':
            kind = 'nearest'
        else:
            kind = 'cubic'
        y_new = resample(x, y, x_new, kind=kind)
        df_new[col] = y_new
    return df_new


def start_stop(x):
    """
    Start-stop samples where the values in x change
    :param x: An array of values
    :return: Two numpy arrays containing the start and stop sample locations.
    """
    ix = np.where(x == 1)[0]
    if len(ix) == 0:
        starts = np.array([])
        stops = np.array([])
        return starts, stops
    ix_diff = ix[1:] - ix[:-1]
    ix_diff_jump = np.where(ix_diff > 1)[0]
    starts = np.append(ix[0], ix[ix_diff_jump + 1])
    stops = np.append(ix[ix_diff_jump] + 1, ix[-1] + 1)
    return starts, stops


def unclose(x, open_size=100):
    """
    Opening operation
    :param x: An array of binary values
    :param open_size: Opening threshold
    :return: An array with the values in x after opening
    """
    y = np.copy(x)
    for i in range(len(x)):
        ix_1 = i
        ix_2 = np.min([len(x) + 1, i + open_size + 1])
        xwin = x[ix_1:ix_2]
        if x[ix_1] == 0:
            ix_uno = np.where(xwin == 1)[0]
            if len(ix_uno) > 0:
                if 0 in xwin[ix_uno[0]:]:
                    ix_end = ix_1 + ix_uno[0] + ix_uno[-1]
                    y[ix_1:ix_end] = 0
    return y


def close(x, close_size=100):
    """
    Opening operation
    :param x: An array of binary values
    :param close_size: Closing threshold
    :return: An array with the values in x after closing
    """
    y = np.copy(x)
    for i in range(len(x)):
        ix_1 = i
        ix_2 = np.min([len(x) + 1, i + close_size + 1])
        xwin = x[ix_1:ix_2]
        if x[ix_1] == 1:
            ix_null = np.where(xwin == 0)[0]
            if len(ix_null) == 0:
                y[i] = x[i]
            elif 1 in xwin[ix_null[0]:]:
                ix_uno = np.where(xwin[ix_null[0]:] == 1)[0]
                ix_end = ix_1 + ix_null[0] + ix_uno[-1]
                y[ix_1:ix_end] = 1
            else:
                y[i] = x[i]
    return y


def write_confusion_matrix(cm, labels):
    """
    Write confusion matrix to text. Use to get a cleaner output
    :param cm: Two-dimensional confusion matrix
    :param labels: A list of labels
    :return: The confusion matrix in text format
    """
    m = 15
    o = " ".ljust(m)
    for label in labels:
        o = o + constants.LABEL_NAMES[label].ljust(m)
    o = o + "\n"
    for (i, label) in enumerate(labels):
        o = o + constants.LABEL_NAMES[label].ljust(m)
        for j in range(len(labels)):
            val = cm[i, j]
            if val - int(val) == 0:
                val_str = str(int(val))
            else:
                val_str = "%.1f" % cm[i, j]
            o = o + val_str.ljust(m)
        o = o + "\n"
    return o


def normalize_confusion_matrix(cm):
    """
    Normalize confusion matrix w.r.t. the class size
    :param cm: Two-dimensional confusion matrix
    :return:
    """
    cmn = np.zeros(cm.shape)
    label_count = np.sum(cm, axis=1)
    for (i, c) in enumerate(label_count):
        if c != 0:
            cmn[i, :] = cm[i, :] / c
    return cmn


def normalize_range(x):
    """
    Normalize range, i.e. to  0, 1
    :param x: An array of values
    :return: x normalized
    """
    max_val = np.max(x)
    min_val = np.min(x)
    return (x - min_val) / (max_val - min_val)


def detrend(x, window_length=600, return_trend=False):
    """
    Remove trend form an array
    :param x: An array of values
    :param window_length: The length used to compute the moving average trend
    :param return_trend: Boolean indicating whether to return the trend or x detrended
    :return: x detrended or the trend
    """
    tail_length = np.floor(window_length / 2)
    nose_length = np.ceil(window_length / 2)
    trend = np.zeros(len(x))
    for i in range(len(x)):
        if i < tail_length:
            first_ix = 0
        else:
            first_ix = int(i - tail_length)
        if i > len(x) - nose_length:
            last_ix = len(x)
        else:
            last_ix = int(i + nose_length)
        trend[i] = np.mean(x[first_ix: last_ix])
    if not return_trend:
        return x - trend
    else:
        return trend


def diff(x):
    """
    The derivative on an array
    :param x:   An array of values
    :return:    An array for the derivative of x
    """
    return x[1:] - x[:-1]


def dirs_in_path(p):
    """
    List all directories in a path
    :param p: A full path
    :return: A list of strings of names of all directories in the path
    """
    return [d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]


def load_recording(file_path, category='raw'):
    """
    Load a recording into a Pandas DataFrame
    :param file_path:   Path to .csv file containing swimming data
    :param category:    = 'raw', 'labeled' or 'processed'
    :return: Recording data in a Pandas DataFrame. If category is 'raw' or 'labeled', the header and footer are also
             returned
    """
    if category == 'raw' or category == 'labeled':
        df = pd.read_csv(file_path, sep='; ', header=None, skiprows=[0], skipfooter=1)
        df.columns = constants.LABELED_COL_NAMES[0: len(df.columns)]
        df.sort_values(by=['timestamp'])
        header = list(pd.read_csv(file_path, sep='; ', nrows=1).columns)
        with open(file_path, 'r') as f:
            footer = list(csv.reader(f))[-1]
            footer = footer[0].split("; ")
        return df, header, footer
    if category == 'processed':
        df = pd.read_csv(file_path)
        return df


def butterfilter(y, fs, fc, order=100):
    """
    Function applying a zero-phase butterworth 2*order filter on y
    :param y: Input signal
    :param fs: Sampling frequency of the input (in Hz)
    :param fc: Cutoff frequency of the filter (in Hz)
    :param order: Order of the butterworth filter (doubled with filtfilt)
    :return: Filtered signal
    """
    corr_fac = ((np.sqrt(2.0)-1.0)**(1.0/(2.0*float(order))))
    fc_corr = float(fc)/corr_fac
    sos = sp.butter(order, 2*fc_corr / float(fs), output='sos')
    y_new = sp.sosfiltfilt(sos, y)
    return y_new


def resample(x, y, x_new, kind='cubic'):
    """
    A simple wrapper for interp1d
    :param x: Original timestamps
    :param y: Original values
    :param x_new: New timestamps
    :param kind: interpolation type
    :return: The values in y evaluated at x_new
    """
    f = scipy.interpolate.interp1d(x, y, kind=kind, bounds_error=False, fill_value=np.nan)
    y_new = f(x_new)
    return y_new


def filterresample(x, y, x_new, order=100, kind='cubic'):
    """
    Wrapper doing filtering and resampling
    :param x: Original timestamps (in ns)
    :param y: Original Values
    :param x_new: New timestampps (in ns)
    :param order: Order of the used filter (higher order -> more precise cutoff)
    :param kind: Type of interpolation used
    :return: Values in y evaluated at x_new
    """
    fs = 1/np.average(np.diff(x))*1E9
    fc = 1/np.average(np.diff(x_new))/2*1E9
    y_filt = butterfilter(y, fs, fc, order)
    y_res = resample(x, y_filt, x_new, kind)
    return y_res


def time_scale_dataframe(df, factor, time_col, label_col):
    """
    Time-scale a dataframe
    :param df: A pandas dataframe
    :param factor: Time-scaling factor
    :param time_col: The column name of timestamps
    :param label_col: The column name of labels
    :return: Time-scaled dataframe
    """
    data_cols = [col for col in df.columns if col not in [time_col, label_col]]
    df_new = pd.DataFrame(columns=df.columns)
    dt = df[time_col].values[1] - df[time_col].values[0]
    dts = dt / factor
    t = df[time_col].values
    ts = t[0] + np.arange(len(t)) * dts
    t_target = np.arange(t[0], ts[-1], dt)
    df_new['timestamp'] = t_target
    for col in data_cols:
        y = df[col].values
        df_new[col] = resample(ts, y, t_target, 'cubic')
    df_new[label_col] = resample(ts, df[label_col].values, t_target, 'nearest')
    return df_new


def get_sample_weights_new(y_cat):
    """
    Compute sample weights based on class size
    :param y_cat: Labels in categorical form
    :return: The sample weights
    """
    class_weights = np.zeros(y_cat.shape[1])
    for i in range(y_cat.shape[1]):
        if np.sum(y_cat[:, i]) == 0:
            class_weights[i] = 0
        else:
            class_weights[i] = 1 / np.sum(y_cat[:, i])
    y_sample_weights = np.sum(y_cat * class_weights, axis=1) / np.sum(y_cat, axis=1)
    y_sample_weights = y_sample_weights / np.sum(y_sample_weights) * len(y_sample_weights)
    return y_sample_weights


def from_categorical(y_cat):
    """
    From categorical to normal labeling
    :param y_cat: Two dimensional array of categorical labels
    :return: An array with normal labeling
    """
    y = np.argmax(y_cat, axis=1)
    return y.astype(int)


def folders_in_path(p):
    """
    Get folders in path
    :param p: Path
    :return: List of folder names in the path
    """
    return [d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]


def main():
    print("Main")


if __name__ == '__main__':
    main()
