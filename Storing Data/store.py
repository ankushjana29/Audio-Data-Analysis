import os
import librosa
import pandas as pd
from openpyxl import Workbook
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from PIL import Image

# set path to folder containing audio files
print("Enter the path:")
path = input()

frame_size = int(input("Enter frame size (in samples): "))
hop_length = int(input("Enter hop length (in samples): "))

# create empty lists to store results
spectral_centroids = []
rolloffs = []
spectral_bandwidths = []
fft_a=[]
mean_spec_cent=[]
filenames = []

count=0
# accessing audio files in folder
for filename in os.listdir(path):
    if filename.endswith('.wav'): # only process .wav files
        audio_path = os.path.join(path, filename)
        y, sr = librosa.load(audio_path) # load audio file
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_size, hop_length=hop_length)[0]
        mean_spec_cent = centroid.mean() 
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=frame_size, hop_length=hop_length)[0]
        fft_data = np.fft.fft(y)
        fft_abs = np.abs(fft_data)
        fft_a.append(fft_abs.mean())
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=frame_size, hop_length=hop_length)[0]
        spectral_centroids.append(mean_spec_cent) # store mean spectral centroid
        rolloffs.append(rolloff.mean()) # store mean spectral rolloff
        spectral_bandwidths.append(bandwidth.mean())
        filenames.append(filename) # store filename
        count+=1    
        
# create a pandas DataFrame to store results
df = pd.DataFrame({'filename': filenames,
                   'spectral_centroid': spectral_centroids,
                   'spectral_bandwidth': spectral_bandwidths,
                   'rolloff': rolloffs,
                   'FFT': fft_a})

# specify the path to the folder where you want to save the Excel file
print("Enter the path where you want to save:")
excel_folder_path = input()

# create the full path to the Excel file
excel_file_path = os.path.join(excel_folder_path, 'audio_testing.xlsx')

# write results to an Excel file in the specified folder
wb = Workbook()
ws = wb.active
ws.title = "Audio Features"
ws.append(['filename', 'spectral_centroid', 'rolloff','spectral_bandwidth','FFT'])
for index, row in df.iterrows():
    ws.append([row['filename'], row['spectral_centroid'], row['rolloff'],row['spectral_bandwidth'],row['FFT']])
wb.save(excel_file_path)

print("Total number of files processed: ", count)


# List all the audio files in the folder
audio_files = [os.path.join(path, file) for file in os.listdir(path)
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
    plot_filename = os.path.join(path, os.path.basename(audio_file)[:-4] + '_plot.png')
    plt.savefig(plot_filename, transparent=True)
    
    # Open the PNG file and remove the alpha channel
    img = Image.open(plot_filename).convert('RGB')
    img.save(plot_filename)

    plt.close()

# Combine all the plots into a single PDF file

# List all the plot files in the folder
plot_files = [os.path.join(excel_folder_path, file) for file in os.listdir(excel_folder_path)
              if file.endswith('_plot.png')]

# Create a list to store the images
images = []

# Loop through each plot file and add it to the list of images
for plot_file in plot_files:
    images.append(Image.open(plot_file))

# Combine all the images into a single PDF file
images[0].save(os.path.join(excel_folder_path, 'combined_plots.pdf'), save_all=True, append_images=images[1:])
