import numpy as np
import matplotlib.pyplot as plt
import os
import librosa as lr
import pandas as pd
import IPython.display as ipd
from librosa.display import specshow
import sklearn
import scipy
from scipy.signal import butter,filtfilt
from argparse import ArgumentParser

print("Enter the source path:")
path=input()
ipd.Audio(path)

audio,sr=lr.load(path)
FRAME_SIZE=1024
HOP_LENGTH=512

def normalize(z, axis = 0):
  return sklearn.preprocessing.minmax_scale(z, axis = axis)

def amplitude_envelope(signal,frame_size,hop_length):
  amplitude_envelope=[]

  #calculate AE for each frame
  for i in range(0,len(signal),hop_length):
    current_frame=max(signal[i:i+frame_size])
    amplitude_envelope.append(current_frame)

  return np.array(amplitude_envelope)

def spectral_centroid():
    sc_audio=lr.feature.spectral_centroid(y=audio, sr=sr)[0]
    frames=range(len(sc_audio))
    t=lr.frames_to_time(frames)

    plt.figure(figsize = (16, 8))

    plt.title("Spectral Centroid")

    lr.display.waveshow(audio, sr = sr, alpha = 0.4)
    plt.plot(t, lr.util.normalize(sc_audio), color = 'c')
    plt.show()
    print('Spectral centroid:', sc_audio.mean())
    
def spectral_rolloff():
    spectral_rolloff = lr.feature.spectral_rolloff(y=audio, sr = sr)[0]
    
    frames = range(len(spectral_rolloff))
    t = lr.frames_to_time(frames)
    
    plt.figure(figsize = (16, 8))
    lr.display.waveshow(audio, sr = sr, alpha = 0.4)

    plt.plot(t, lr.util.normalize(spectral_rolloff), color = 'r')
    plt.title("Spectral Rolloff")
    
    plt.show()
    print('Spectral Rolloff:', spectral_rolloff.mean())
    
def fft():
    fft_spectrum = np.fft.rfft(audio)

    freq = np.fft.rfftfreq(audio.size, d=1./sr)

    plt.plot(freq, np.abs(fft_spectrum))
    plt.xlabel("frequency, Hz")
    plt.ylabel("Amplitude, units")
    plt.title("FFT")
    plt.show()
    
    fft_spectrum_abs = np.abs(fft_spectrum)


    print('Values for the original audio:') 
    max=0
    for i,f in enumerate(fft_spectrum_abs):
        if(i > max and f>5):    
            max = i
            l=f
    print('Max Frequency={}'.format(np.round(max/10)) + ' with Amplitude='+str(l))

    max1=0
    for i,f in enumerate(fft_spectrum_abs): 
        if(f > max1):    
            max1 = f
            k=i           
    print('Max Amplitude={}'.format(np.round(max1)) + ' with Frequency='+str(k/10))
    

def amplitude():
    ae_x=amplitude_envelope(audio,FRAME_SIZE,HOP_LENGTH)
    frames=range(0,ae_x.size)
    t=lr.frames_to_time(frames,hop_length=HOP_LENGTH)
    
    plt.figure(figsize=(25,21))

    lr.display.waveshow(audio,alpha=0.5)
    plt.plot(t,ae_x,color="r")
    plt.title("Amplitude Envelope")
    plt.ylim=([-1,1])
    plt.show()
    print('Amplitude Envelope:', ae_x.mean())
    
def spectrogram():
    spec = np.abs(librosa.stft(audio))
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(librosa.amplitude_to_db(spec, ref=np.max),
                             y_axis='log', x_axis='time', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.xlabel('Time (s)')
    
    plt.show()
          
ans=True
print ("\n 1. Spectral Centroid \n 2. Spectral Rolloff \n 3. FFT \n 4. Amplitude Envelope\n 5. Spectrogram \n")
while ans:
    ans=input("What would you like to find?")
    if ans=="1": 
        spectral_centroid()
    elif ans=="2":
        spectral_rolloff()
    elif ans=="3":
        fft() 
    elif ans=="4":
        amplitude() 
    elif ans=="5":
        spectrogram() 
    elif ans !="":
        print("\n Not Valid Choice Try again")
        quit()
