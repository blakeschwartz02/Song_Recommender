'''
This script uses the VGGish model to extract features for each 30 sec 
audio in the GTZAN dataset and stores them in a csv file.
'''
import os
import numpy as np
import pandas as pd
import csv

import vggish_feature_extractor as extractor

os.chdir('C:\\Users\\blake\OneDrive - The University of Texas at Austin')
base_dir = 'metal_subgenres'
genres = ['alternative', 'death', 'folk', 'glam', 'industrial', 'metalcore', 'nu', 'NWOBHM', 'progressive', 'symphonic', 'thrash']
csv_path = 'Project\\vggish_feature_extraction/vggish_metal_features.csv'

feature_columns = ['Feature ' + str(col) for col in range(128 * 31)]
csv_columns = ['song', 'genre'] + feature_columns

with open(csv_path, 'w', newline='', encoding="utf-8") as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow(csv_columns)

    for genre in genres:
        genre_dir = os.path.join(base_dir, genre)
        
        for audio in os.listdir(genre_dir):
            print('Extracting features from: ', audio)
            audio_path = os.path.join(genre_dir, audio)
            
            audio_features = extractor.extract_features_from_wav_file(audio_path, postprocess=False)
            audio_features = audio_features.flatten()

            audio_features = [str(feature) for feature in audio_features]

            audio_columns = [audio, genre] + audio_features
            csv_writer.writerow(audio_columns)

