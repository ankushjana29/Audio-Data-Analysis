import numpy as np
import matplotlib.pyplot as plt
import os
import librosa as lr
import pandas as pd
import IPython.display as ipd
from librosa.display import specshow
import sklearn
from scipy.signal import butter,filtfilt
import plotly.graph_objects as go

# Filter requirements.
T = 5.0         # Sample Period
fs = 30.0       # sample rate, Hz
cutoff = 2      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
nyq = 0.5 * fs  # Nyquist Frequency
order = 2       # sin wave can be approx represented as quadratic
n = int(T * fs) # total number of samples

def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y
    
audio1="Asian_Koel/Asian_Koel.000.wav"
ipd.Audio(audio1)

audio1_lr,sr=lr.load(audio1)

# Filter the data, and plot both the original and filtered signals.
y = butter_lowpass_filter(audio1_lr, cutoff, fs, order)
fig = go.Figure()
fig.add_trace(go.Scatter(
            y = audio1_lr,
            line =  dict(shape =  'spline' ),
            name = 'signal with noise'
            ))
fig.add_trace(go.Scatter(
            y = y,
            line =  dict(shape =  'spline' ),
            name = 'filtered signal'
            ))
fig.show()
