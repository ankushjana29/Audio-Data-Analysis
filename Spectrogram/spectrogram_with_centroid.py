import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from PIL import Image

# Define the path to the folder containing audio files
print("Enter the folder path:")
path=input()
audio_folder = (path)

# Define the path to the output folder for the plots
print("Enter the output folder path:")
out_path=input()
plot_folder = (out_path)

# Create the output folder if it doesn't already exist
if not os.path.exists(plot_folder):
    os.makedirs(plot_folder)

# List all the audio files in the folder
audio_files = [os.path.join(audio_folder, file) for file in os.listdir(audio_folder)
               if file.endswith('.wav')]

# Loop through each audio file and generate the plot
for audio_file in audio_files:
    # Load the audio file
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

    # Save the plot as a PNG file in the output folder with the same name as the audio file
    plot_filename = os.path.join(plot_folder, os.path.basename(audio_file)[:-4] + '_plot.png')
    plt.savefig(plot_filename, transparent=True)
    
    # Open the PNG file and remove the alpha channel
    img = Image.open(plot_filename).convert('RGB')
    img.save(plot_filename)

    plt.close()

# Combine all the plots into a single PDF file

# List all the plot files in the folder
plot_files = [os.path.join(plot_folder, file) for file in os.listdir(plot_folder)
              if file.endswith('_plot.png')]

# Create a list to store the images
images = []

# Loop through each plot file and add it to the list of images
for plot_file in plot_files:
    images.append(Image.open(plot_file))

# Combine all the images into a single PDF file
images[0].save(os.path.join(plot_folder, 'combined_plots.pdf'), save_all=True, append_images=images[1:])
