import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

audio1 = r"C:\Users\admin\Desktop\internship\Dysphagia\Normal\mic.wav"
audio1_lr, sr = librosa.load(audio1, sr=None)

audio2=r"C:\Users\admin\Desktop\internship\Dysphagia\Patient\mic_02_14.wav"
audio2_lr, sr = librosa.load(audio2, sr=None)

print(sr)


# Plot the time-domain waveform
plt.figure(figsize=(10, 4))
librosa.display.waveshow(y=audio1_lr, sr=sr)
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("Normal")
plt.tight_layout()
plt.show()

# Plot the time-domain waveform
plt.figure(figsize=(10, 4))
librosa.display.waveshow(y=audio2_lr, sr=sr)
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("Patient")
plt.tight_layout()
plt.show()

#fft spectrum
fft1 = np.fft.fft(audio1_lr)
magnitude1 = np.abs(fft1)
frequency1 = np.linspace(0, sr, len(magnitude1))

fft2 = np.fft.fft(audio2_lr)
magnitude2 = np.abs(fft2)
frequency2 = np.linspace(0, sr, len(magnitude2))

# Plot the FFT spectrum
plt.figure(figsize=(10, 4))
plt.plot(frequency1[1:len(frequency1)//2], magnitude1[1:len(magnitude1)//2])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('FFT Spectrum of Normal Person')
plt.tight_layout()
plt.show()

# Plot the FFT spectrum for patient
plt.figure(figsize=(10, 4))
plt.plot(frequency1[1:len(frequency2)//2], magnitude2[1:len(magnitude2)//2])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('FFT Spectrum of Normal Person')
plt.tight_layout()
plt.show()



#for normal person
spectrogram1 = np.abs(librosa.stft(audio1_lr))
spectral_centroids1 = librosa.feature.spectral_centroid(y=audio1_lr, sr=sr)[0]
spectral_centroids_norm1 = librosa.util.normalize(spectral_centroids1)

# Plot the spectrogram and spectral centroid values
plt.figure(figsize=(10, 6))

# Plot the spectrogram
librosa.display.specshow(librosa.amplitude_to_db(spectrogram1, ref=np.max),
                             y_axis='log', x_axis='time', sr=sr)


# Plot the spectral centroid values
plt.plot(librosa.times_like(spectral_centroids1), spectral_centroids_norm1, color='w', linewidth=2)

# Add title and axis labels
plt.title('Spectrogram for Normal Person')
plt.xlabel('Time (s)')
plt.ylabel('Normalized Spectral Centroid')
plt.show()



#for patient
spectrogram2 = np.abs(librosa.stft(audio2_lr))
spectral_centroids2 = librosa.feature.spectral_centroid(y=audio2_lr, sr=sr)[0]
spectral_centroids_norm2 = librosa.util.normalize(spectral_centroids2)

# Plot the spectrogram and spectral centroid values
plt.figure(figsize=(10, 6))

# Plot the spectrogram
librosa.display.specshow(librosa.amplitude_to_db(spectrogram2, ref=np.max),
                             y_axis='log', x_axis='time', sr=sr)


# Plot the spectral centroid values
plt.plot(librosa.times_like(spectral_centroids2), spectral_centroids_norm2, color='w', linewidth=2)

# Add title and axis labels
plt.title('Spectrogram for Patient')
plt.xlabel('Time (s)')
plt.ylabel('Normalized Spectral Centroid')
plt.show()

