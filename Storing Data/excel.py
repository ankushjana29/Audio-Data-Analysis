import os
import librosa
import pandas as pd
from openpyxl import Workbook
import numpy as np

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
excel_file_path = os.path.join(excel_folder_path, 'audio_features.xlsx')

# write results to an Excel file in the specified folder
wb = Workbook()
ws = wb.active
ws.title = "Audio Features"
ws.append(['filename', 'spectral_centroid', 'rolloff','spectral_bandwidth','FFT'])
for index, row in df.iterrows():
    ws.append([row['filename'], row['spectral_centroid'], row['rolloff'],row['spectral_bandwidth'],row['FFT']])
wb.save(excel_file_path)

print("Total number of files processed: ", count)

