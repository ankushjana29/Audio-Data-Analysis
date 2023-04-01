import matplotlib.pyplot as plt
import librosa as lr
import numpy as np
from scipy.io import wavfile
#%matplotlib inline

audio1="Asian_Koel/Asian_Koel.000.wav"
sound,sr=lr.load(audio1)

length_in_s = sound.shape[0] / sr
print(length_in_s)

time = np.arange(sound.shape[0]) / sound.shape[0] * length_in_s

signal=sound
plt.plot(time[6000:7000], signal[6000:7000])
plt.xlabel("time, s")
plt.ylabel("Signal, relative units")
plt.show()

fft_spectrum = np.fft.rfft(signal)
freq = np.fft.rfftfreq(signal.size, d=1./sr)
fft_spectrum_abs = np.abs(fft_spectrum)

plt.plot(freq[:110250], np.abs(fft_spectrum[:110250]))
plt.xlabel("frequency, Hz")
plt.ylabel("Amplitude, units")
plt.show()

for i,f in enumerate(fft_spectrum_abs):
    if i >= 200 and i<200000:
        print('frequency = {} Hz with amplitude {} '.format(np.round(freq[i],1),  np.round(f)))
        

max1=0
for i,f in enumerate(fft_spectrum_abs):
    if i >= 200 and i<200000: #frequency range 
        if(f > max1):    
            max1 = f
            k=i
print('Max Amplitude={}'.format(np.round(max1)) + ' with Frequency='+str(k))


for i,f in enumerate(fft_spectrum_abs):
    if i >= 200 and i<200000: #frequency range 
      if(i > max and f>5):    
        max = i
        l=f
print('Max Frequency={}'.format(np.round(max)) + ' with Amplitude='+str(l))

plt.plot(freq[:30000], np.abs(fft_spectrum[:30000]))
plt.xlabel("frequency, Hz")
plt.ylabel("Amplitude, units")
plt.show()

