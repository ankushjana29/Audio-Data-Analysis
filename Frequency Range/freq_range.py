import os
import librosa
import numpy as np

# User inputs
freq_min = int(input("Enter minimum frequency range (in Hz): "))
freq_max = int(input("Enter maximum frequency range (in Hz): "))
frame_size = int(input("Enter frame size (in samples): "))
hop_length = int(input("Enter hop length (in samples): "))

# Folder path containing audio files
folder_path = input("Enter the path: ")

# Initialize arrays to store feature values
centroids = []
bandwidths = []
rolloffs = []
ffts = []

# Round up `n_fft` to next power of 2 greater than or equal to `frame_size`
n_fft = 2**int(np.ceil(np.log2(frame_size)))

# Loop over all files in folder
for filename in os.listdir(folder_path):
    # Check if file is a WAV file
    if filename.endswith('.wav'):
        # Load audio file
        file_path = os.path.join(folder_path, filename)
        y, sr = librosa.load(file_path, sr=None, mono=True)

        # Compute features
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, fmin=freq_min, fmax=freq_max)
        spectral_centroid_mel = librosa.feature.spectral_centroid(S=mel_spec, freq=None)
        bandwidth_mel = librosa.feature.spectral_bandwidth(S=mel_spec, freq=None)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, roll_percent=0.85)

        # Convert spectral centroid and bandwidth from Mel to Hz
        spectral_centroid_hz = librosa.core.mel_to_hz(spectral_centroid_mel)
        bandwidth_hz = librosa.core.mel_to_hz(bandwidth_mel)

        # Compute FFT
        fft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hann', center=True))

        # Append feature values to arrays
        centroids.append(np.mean(spectral_centroid_hz))
        bandwidths.append(np.mean(bandwidth_hz))
        rolloffs.append(np.mean(rolloff))
        ffts.append(np.mean(fft))

    else:
        print('Skipping file:', filename, '- Not a WAV file')

# Compute mean feature values
mean_centroid = np.mean(centroids)
mean_bandwidth = np.mean(bandwidths)
mean_rolloff = np.mean(rolloffs)
mean_fft = np.mean(ffts)

# Display mean feature values
print('Mean Spectral Centroid (Hz):', mean_centroid)
print('Mean Spectral Bandwidth (Hz):', mean_bandwidth)
print('Mean Rolloff:', mean_rolloff)
print('Mean FFT:', mean_fft)
