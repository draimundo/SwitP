import os
import numpy as np
import utils
import pandas as pd

data_path = r'..\..\data\re_labeled'
save_path = r'..\..\data\processed'
data_path = os.path.abspath(data_path)
save_path = os.path.abspath(save_path)
fs = 30
skip_existing = True
kind = 'cubic'
sensor_expand = ['ACC', 'GYRO', 'MAG']

# Create folder structure
for dd in os.listdir(data_path):
    #if not os.path.exists(os.path.join(save_path, dd)):
    #    os.mkdir(os.path.join(save_path, dd))
    for ud in os.listdir(os.path.join(data_path, dd)):
        if not os.path.exists(os.path.join(save_path, ud)):
            os.mkdir(os.path.join(save_path, ud))

# Work
for dd in os.listdir(data_path):
    for ud in os.listdir(os.path.join(data_path, dd)):
        files_in_path = os.listdir(os.path.join(data_path, dd, ud))
        csv_in_path = [f for f in files_in_path if f.endswith('csv')]
        for f in csv_in_path:
            if os.path.exists(os.path.join(save_path, ud, f)) and skip_existing:
                print("Skipping: %s - %s - %s" % (dd, ud, f))
                continue
            print("Working on: %s - %s - %s" % (dd, ud, f))
            file_path = os.path.join(data_path, dd, ud, f)
            df, header, footer = utils.load_recording(file_path)
            if footer == 'Failure':
                print("Skipping: %s - %s - %s. Failure" % (dd, ud, f))
                continue
            df_acc = df[df['sensor'] == 'ACC']
            ts_new = np.arange(df_acc['timestamp'].values[0], df_acc['timestamp'].values[-1], 1 / fs * 10 ** 9)
            df_new = pd.DataFrame()
            df_new['timestamp'] = ts_new
            df_new['label'] = np.abs(np.rint(utils.filterresample(df_acc['timestamp'].values, df_acc['label'].values, ts_new)))
            # Adding new sensor columns
            for s in sensor_expand:
                df_sens = df[df['sensor'] == s]
                df_new[s + '_0'] = utils.filterresample(df_sens['timestamp'], df_sens['value_0'], ts_new)
                df_new[s + '_1'] = utils.filterresample(df_sens['timestamp'], df_sens['value_1'], ts_new)
                df_new[s + '_2'] = utils.filterresample(df_sens['timestamp'], df_sens['value_2'], ts_new)
                df_new[s + '_012'] = np.sqrt(df_new[s + '_0'] ** 2 + df_new[s + '_1'] ** 2 + df_new[s + '_2'] ** 2)
                df_new[s + '_01'] = np.sqrt(df_new[s + '_0'] ** 2 + df_new[s + '_1'] ** 2)
                df_new[s + '_02'] = np.sqrt(df_new[s + '_0'] ** 2 + df_new[s + '_2'] ** 2)
                df_new[s + '_12'] = np.sqrt(df_new[s + '_1'] ** 2 + df_new[s + '_2'] ** 2)
            df_new.to_csv(os.path.join(save_path, ud, f))

# Writing a description file
# file_name = 'description.txt'
# file = open(os.path.join(save_path, file_name), 'w')
# file.write("Sampling frequency: %d\n" % fs)
# file.write("Interpolation: %s\n" % kind)
# file.write("Expanded sensors: %s" % sensor_expand)
# file.close()
