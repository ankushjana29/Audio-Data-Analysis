import librosa as lr
import IPython.display as ipd
import matplotlib.pyplot as plt
from librosa.display import specshow

#mention the path
audio=("/home/audio/Asian_Koel.wav")
x,sr=lr.load(audio)

X = lr.stft(x)
Xdb = lr.amplitude_to_db(abs(X))
plt.figure(figsize = (10, 5))
lr.display.specshow(Xdb, sr = sr, x_axis = 'time', y_axis = 'hz')
plt.colorbar()

