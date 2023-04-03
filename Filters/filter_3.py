import numpy as np
import matplotlib.pyplot as plt
import os
import librosa as lr
import pandas as pd
import IPython.display as ipd
from librosa.display import specshow
import sklearn
from scipy.signal import butter,filtfilt

# Filter requirements.
T = 5.0         # Sample Period
fs = 30.0       # sample rate, Hz
cutoff = 2      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
nyq = 0.5 * fs  # Nyquist Frequency
order = 2       # sin wave can be approx represented as quadratic
n = int(T * fs) # total number of samples

#The Nyquist rate or frequency is the minimum rate at which a finite bandwidth signal 
#needs to be sampled to retain all of the information

def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y
  
audio1="Asian_Koel/Asian_Koel.000.wav"
audio1_lr,sr=lr.load(audio1)
y = butter_lowpass_filter(audio1_lr, cutoff, fs, order)


#time domain graph
plt.figure(figsize=(14, 5))
plt.title("Original")
lr.display.waveshow(audio1_lr, sr=sr)

plt.figure(figsize=(14, 5))
plt.title("Filtered")
lr.display.waveshow(y, sr=sr)

plt.show()


#frequency domain graph
fft_spectrum = np.fft.rfft(audio1_lr)
fft_spectrum1 = np.fft.rfft(y)

freq = np.fft.rfftfreq(audio1_lr.size, d=1./sr)
freq1 = np.fft.rfftfreq(y.size, d=1./sr)

plt.plot(freq, np.abs(fft_spectrum))
plt.xlabel("frequency, Hz")
plt.ylabel("Amplitude, units")
plt.title("Original")
plt.show()

plt.plot(freq1, np.abs(fft_spectrum1))
plt.xlabel("frequency, Hz")
plt.ylabel("Amplitude, units")
plt.title("Filtered")
plt.show()

#comparison of values between original and filtered audio file
fft_spectrum_abs = np.abs(fft_spectrum)

print('Values for the original audio:') 
max=0
for i,f in enumerate(fft_spectrum_abs):
    if i >= 200 and i<200000: #frequency range 
      if(i > max and f>5):    
        max = i
        l=f
print('Max Frequency={}'.format(np.round(max)) + ' with Amplitude='+str(l))

max1=0
for i,f in enumerate(fft_spectrum_abs):
    if i >= 200 and i<200000: #frequency range 
        if(f > max1):    
            max1 = f
            k=i           
print('Max Amplitude={}'.format(np.round(max1)) + ' with Frequency='+str(k))

fft_spectrum_abs1 = np.abs(fft_spectrum1)

print('\nValues for the filtered audio:') 
max=0
for i,f in enumerate(fft_spectrum_abs1):
    if i >= 200 and i<200000: #frequency range 
      if(i > max and f>5):    
        max = i
        l=f
print('Max Frequency={}'.format(np.round(max)) + ' with Amplitude='+str(l))

max1=0
for i,f in enumerate(fft_spectrum_abs1):
    if i >= 200 and i<200000: #frequency range 
        if(f > max1):    
            max1 = f
            k=i
print('Max Amplitude={}'.format(np.round(max1)) + ' with Frequency='+str(k))
