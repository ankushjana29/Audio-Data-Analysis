import numpy as np
import matplotlib.pyplot as plt
import os
import librosa as lr
import pandas as pd
import IPython.display as ipd
from librosa.display import specshow
import sklearn
from sklearn import preprocessing
from unicodedata import normalize

audio1="/home/Asian_Koel.010.wav"
audio2="/home/Asian_Koel.011.wav"

audio1_lr, sr=lr.load(audio1)
audio2_lr, _=lr.load(audio2)

FRAME_SIZE=1024
HOP_LENGTH=512

sc_audio1_lr =lr.feature.spectral_centroid(y=audio1_lr, sr=sr, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0] #to make 1D
sc_audio2_lr =lr.feature.spectral_centroid(y=audio2_lr, sr=sr, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]

frames=range(len(sc_audio1_lr))
t=lr.frames_to_time(frames)

plt.figure(figsize=[25,10])

lr.display.waveshow(audio1_lr, sr = sr, alpha = 0.4)
plt.plot(t, sc_audio1_lr, color='c')
plt.title("Audio1")


lr.display.waveshow(audio1_lr, sr = sr, alpha = 0.4)
plt.plot(t, sc_audio2_lr, color='r')
plt.title("Audio2")

plt.show()

#spectral centroid
x,sr=lr.load(audio1)
z,sr=lr.load(audio2)

spectral_centroids = lr.feature.spectral_centroid(y=x, sr = sr)[0]

plt.figure(figsize = (16, 8))
frames = range(len(spectral_centroids))
t = lr.frames_to_time(frames)

def normalize(x, axis = 0):
  return sklearn.preprocessing.minmax_scale(x, axis = axis)


plt.title("Audio1")

lr.display.waveshow(x, sr = sr, alpha = 0.4)
#plt.plot(t, normalize(spectral_centroids), color = 'c')
plt.plot(t, lr.util.normalize(spectral_centroids), color = 'c')

spectral_centroids = lr.feature.spectral_centroid(y=z, sr = sr)[0]

plt.figure(figsize = (16, 8))
frames = range(len(spectral_centroids))
t = lr.frames_to_time(frames)

def normalize(z, axis = 0):
  return sklearn.preprocessing.minmax_scale(z, axis = axis)


plt.title("Audio2")
lr.display.waveshow(z, sr = sr, alpha = 0.4)
#plt.plot(t, normalize(spectral_centroids), color = 'r')
plt.plot(t, lr.util.normalize(spectral_centroids), color = 'r')

#spectral bandwidth
x,sr=lr.load(audio1)
z,sr=lr.load(audio2)

frames=range(len(sc_audio1_lr))
t=lr.frames_to_time(frames)

spectral_bandwidth_2 = lr.feature.spectral_bandwidth(y=x + 0.11, sr = sr)[0]
spectral_bandwidth_3 = lr.feature.spectral_bandwidth(y=x + 0.11, sr = sr, p = 3)[0]
spectral_bandwidth_4 = lr.feature.spectral_bandwidth(y=x + 0.11, sr = sr, p = 4)[0]

plt.figure(figsize = (14, 8))
lr.display.waveshow(x, sr = sr, alpha = 0.4)
plt.plot(t, lr.util.normalize(spectral_bandwidth_2), color = 'c')
plt.plot(t, lr.util.normalize(spectral_bandwidth_3), color = 'r')
plt.plot(t, lr.util.normalize(spectral_bandwidth_4), color = 'y')
plt.title("Audio1")

spectral_bandwidth_2 = lr.feature.spectral_bandwidth(y=z + 0.11, sr = sr)[0]
spectral_bandwidth_3 = lr.feature.spectral_bandwidth(y=z + 0.11, sr = sr, p = 3)[0]
spectral_bandwidth_4 = lr.feature.spectral_bandwidth(y=z + 0.11, sr = sr, p = 4)[0]

plt.figure(figsize = (14, 8))
lr.display.waveshow(z, sr = sr, alpha = 0.4)
plt.plot(t, lr.util.normalize(spectral_bandwidth_2), color = 'c')
plt.plot(t, lr.util.normalize(spectral_bandwidth_3), color = 'r')
plt.plot(t, lr.util.normalize(spectral_bandwidth_4), color = 'y')
plt.title("Audio2")



plt.show()

#spectral rolloff
spectral_rolloff = lr.feature.spectral_rolloff(y=x + 0.01, sr = sr)[0]
plt.figure(figsize = (16, 8))
lr.display.waveshow(x, sr = sr, alpha = 0.4)
#plt.plot(t, normalize(spectral_rolloff), color = 'r')
plt.plot(t, librosa.util.normalize(spectral_rolloff), color = 'r')
plt.title("Audio1")

spectral_rolloff = lr.feature.spectral_rolloff(y=z + 0.01, sr = sr)[0]
plt.figure(figsize = (16, 8))
lr.display.waveshow(z, sr = sr, alpha = 0.4)
#plt.plot(t, normalize(spectral_rolloff), color = 'r')
plt.plot(t, librosa.util.normalize(spectral_rolloff), color = 'r')
plt.title("Audio2")

plt.show()


#amplitude envelope
FRAME_SIZE=1024
HOP_LENGTH=512

def amplitude_envelope(signal,frame_size,hop_length):
  amplitude_envelope=[]

  #calculate AE for each frame
  for i in range(0,len(signal),hop_length):
    current_frame=max(signal[i:i+frame_size])
    amplitude_envelope.append(current_frame)

  return np.array(amplitude_envelope)  

def fancy_amplitude_envelope(signal,frame_size,hop_length):
  return np.array([max(signal[i:i+frame_size]) for i in range(0,signal.size,hop_length)])

ae_x=amplitude_envelope(x,FRAME_SIZE,HOP_LENGTH)
ae_z=amplitude_envelope(z,FRAME_SIZE,HOP_LENGTH)

frames=range(0,ae_x.size)
t=lr.frames_to_time(frames,hop_length=HOP_LENGTH)

plt.figure(figsize=(25,21))


lr.display.waveshow(x,alpha=0.5)
plt.plot(t,ae_x,color="r")
plt.title("Audio1")
plt.ylim=([-1,1])

plt.figure(figsize=(25,21))


lr.display.waveshow(z,alpha=0.5)
plt.plot(t,ae_z,color="r")
plt.title("Audio2")
plt.ylim=([-1,1])

plt.show()
