import matplotlib
matplotlib.use('TkAgg')   # Use TkAgg backend to avoid conflicts with Librosa
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

print("Enter the path of first audio file:")
path1 = input()

print("Enter the path of second audio file:")
path2 = input()

y1, sr = librosa.load(path1)
y2, sr = librosa.load(path2)

n_fft = 2048   
hop_length = 512   
stft1 = librosa.stft(y1, n_fft=n_fft, hop_length=hop_length)
stft2 = librosa.stft(y2, n_fft=n_fft, hop_length=hop_length)

spec1 = librosa.amplitude_to_db(abs(stft1), ref=np.max)
spec2 = librosa.amplitude_to_db(abs(stft2), ref=np.max)

print("STFT shape of audio 1: ", stft1.shape)
print("STFT shape of audio 2: ", stft2.shape)

plt.figure(figsize=(10, 4))
librosa.display.specshow(spec1, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Magnitude spectrogram of Audio 1')
plt.xlabel('Time')
plt.ylabel('Frequency (Hz)')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
librosa.display.specshow(spec2, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Magnitude spectrogram of Audio 2')
plt.xlabel('Time')
plt.ylabel('Frequency (Hz)')
plt.tight_layout()
plt.show()
