# Basic tutorial in how to load data
import pandas as pd
import utils
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as nf
import scipy.signal as sp

p_res = '../../data/processed_30hz_relabeled/7/Butterfly_1532367411257.csv'
p_orig = '../../data/re_labeled/2018_07_23/darya/Butterfly_1532367411257.csv'

#interactive plots
mpl.use('Qt5Agg')

# Use this to load a re-sampled recording
df_res = pd.read_csv(p_res)
(df_orig,header,footer) = utils.load_recording(p_orig)

df_orig_acc = df_orig[df_orig['sensor'] == 'ACC']

res_avg_T = np.average(np.diff(df_res['timestamp'])*1E-9)
orig_avg_T = np.average(np.diff(df_orig_acc['timestamp'])*1E-9)

res_fft = nf.fft(df_res['ACC_0'].values)
orig_fft = nf.fft(df_orig_acc['value_0'])

# Normalize to keep amplitude in f domain
res_fft = res_fft/res_fft.size
orig_fft = orig_fft/orig_fft.size

res_f = nf.fftfreq(res_fft.size, d=res_avg_T)
orig_f = nf.fftfreq(orig_fft.size, d=orig_avg_T)

imp= np.zeros(100000)
imp[50000] = 1
imp_filt = utils.butterfilter(imp,fs=orig_avg_T,fc=15,order=100)
imp_filt_f = nf.fftfreq(imp_filt.size, d=orig_avg_T)
imp_filt_fft = nf.fft(imp_filt)

fig = plt.figure()
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [dB]')
plt.plot(imp_filt_f, 20*np.log10(abs(imp_filt_fft)),lw=0,marker='+')
xmin,xmax,ymin,ymax = plt.axis('tight')
plt.plot([xmin,xmax],[-3.01,-3.01])

filt_acc = utils.butterfilter(df_orig_acc['value_0'],fs=1/orig_avg_T,fc=15);
filt_res_acc_ts = np.arange(df_orig_acc['timestamp'].values[0], df_orig_acc['timestamp'].values[-1], 1 / 30 * 10 ** 9)
filt_res_acc = utils.resample(df_orig_acc['timestamp'].values,filt_acc, filt_res_acc_ts)

filt_res_avg_T = np.average(np.diff(filt_res_acc_ts)*1E-9)
filt_res_fft = nf.fft(filt_res_acc)
filt_res_fft = filt_res_fft/filt_res_fft.size
filt_res_f = nf.fftfreq(filt_res_fft.size, d=filt_res_avg_T)

fig1, (ax1,ax2) = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=False)

ax1.plot(df_orig_acc['timestamp'].values, df_orig_acc['value_0'], label='orig', alpha=0.8)
ax1.plot(df_res['timestamp'].values, df_res['ACC_0'].values, label='res', alpha=0.5)
ax1.plot(filt_res_acc_ts, filt_res_acc, label='filt_res', alpha=0.3)
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
ax1.set(xlabel='Time [s]', ylabel='Amplitude [1]')

ax2.plot(orig_f,20*np.log10(abs(orig_fft)), label='orig',lw=0,marker='+', alpha=0.8)
ax2.plot(res_f,20*np.log10(abs(res_fft)), label='res',lw=0,marker='+', alpha=0.5)
ax2.plot(filt_res_f,20*np.log10(abs(filt_res_fft)), label='filt_res',lw=0,marker='+', alpha=0.3)
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
ax2.set(xlabel='Frequency [Hz]', ylabel='Magnitude [dB]')

plt.grid(True, which='both')
plt.show()