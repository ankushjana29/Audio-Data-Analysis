import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Load the audio file
audio_file = "Birds/Asian_Koel/Asian_Koel.002.wav"
audio, sr = librosa.load(audio_file, sr=None)

# Compute the spectrogram
spectrogram = np.abs(librosa.stft(audio))

# Compute the spectral centroid
spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]

# Normalize the spectral centroid values between 0 and 1
spectral_centroids_norm = librosa.util.normalize(spectral_centroids)

# Plot the spectrogram and spectral centroid values
plt.figure(figsize=(10, 6))

# Plot the spectrogram
librosa.display.specshow(librosa.amplitude_to_db(spectrogram, ref=np.max),
                         y_axis='log', x_axis='time', sr=sr)
plt.colorbar(format='%+2.0f dB')

# Plot the spectral centroid values
plt.plot(librosa.times_like(spectral_centroids), spectral_centroids_norm, color='w', linewidth=2)

# Add title and axis labels
plt.title('Spectrogram with Spectral Centroid')
plt.xlabel('Time (s)')
plt.ylabel('Normalized Spectral Centroid')

plt.show()
