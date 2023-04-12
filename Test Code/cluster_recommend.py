import sklearn
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from sklearn import metrics 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.optimizers import SGD
import os
from sklearn.neighbors import NearestNeighbors
import youtube_dl
import feature_extractor
from pydub import AudioSegment

os.chdir('C:\\Users\\blake\\OneDrive - The University of Texas at Austin\\Project')
music_data_with_titles = pd.read_csv('Data\\new_features.csv')
music_data = music_data_with_titles.drop(['Unnamed: 0', 'title'], axis=1)

neighbors = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(music_data)

print(os.listdir('Data'))
input_path = 'Data\\call_me_little_sunshine.wav'

input_features = feature_extractor.extract_features_from_wav_file(input_path)
print(input_features)
input_features = input_features.drop(['title'],axis=1)
recommend = neighbors.kneighbors(input_features, n_neighbors=4)
recommend_titles = music_data_with_titles.iloc[recommend[1][0]]
print(recommend_titles)


