'''
This file contains methods for training and evaluating
multiple classification models. It additionally contains
methods for extracting features from audios using the VGGish
pre-trained model as well as methods to classify the genre
of an audio using a model.
'''


import numpy as np
import pandas as pd
import sys
import os
import tensorflow as tf
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
sys.path.append(os.getcwd() + '/vggish_imports')
import vggish_feature_extractor as feature_extractor
import csv
import os
import pickle
from sklearn.preprocessing import LabelEncoder

GTZAN_vggish_features_csv_path = 'Data/vggish_features.csv'


def train_CNN_model(X_train, Y_train, X_test, Y_test):
    '''
    Trains a CNN model on the given data and returns the trained model
    and saves it on the disk.

    params:
        X_train: training data
        Y_train: training labels
        X_test: testing data
        Y_test: testing labels

    returns:
        model: trained CNN model
    '''

    # Define the architecture of the CNN
    model=tf.keras.models.Sequential([
        layers.Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
        layers.MaxPooling1D(2),
        layers.Conv1D(32, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')   
    ])
    # Compile the model
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train, Y_train, epochs=100, batch_size=16, validation_data=(X_test, Y_test))

    # save the model to disk
    filename = 'cnn_model.sav'
    pickle.dump(model, open(filename, 'wb'))

    return model



def train_mlp_model(X_train, Y_train, X_test, Y_test):
    '''
    Trains a MLP model on the given data and returns the trained model
    and saves it on the disk.

    params:
        X_train: training data
        Y_train: training labels
        X_test: testing data
        Y_test: testing labels

    returns:
        mlp: trained MLP model
    '''

    #train model
    mlp = MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=300, alpha=0.001,solver='sgd', verbose=1,  random_state=21,tol=0.000000001, activation='relu', learning_rate='adaptive')
    mlp.fit(X_train, Y_train)

    # save the model to disk
    filename = 'mlp_model.sav'
    pickle.dump(mlp, open(filename, 'wb'))

    return mlp

def train_logistic_model(X_train, Y_train, X_test, Y_test):
    '''
    Trains a Logistic Regression model on the given data and returns the trained model
    and saves it on the disk.

    params:
        X_train: training data
        Y_train: training labels
        X_test: testing data
        Y_test: testing labels

    returns:    
        logreg: trained logistic regression model
    '''

    #train model
    logreg = LogisticRegression()
    logreg.fit(X_train, Y_train)

    # save the model to disk
    filename = 'logreg_model.sav'
    pickle.dump(logreg, open(filename, 'wb'))

    return logreg

def calculate_accuracy(model, X, Y):
    '''
    Calculates the accuracy of the given model on the given data.

    params:
        model: trained model
        X: data
        Y: labels

    returns:
        accuracy: accuracy of the model on the given data
    '''

    accuracy = accuracy_score(Y, model.predict(X))
    return accuracy


def classify_song(audio_path, model):
    '''
    Classifies the given audio file into one of the 10 genres.

    params:
        audio_path: path to the 30 second wav audio file of song

    returns:
        genre: predicted genre of the song
    '''

    # load the data
    genres = pd.read_csv('Data/vggish_features.csv')['genre']

    # label encoding
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    Y = le.fit_transform(genres)

    # load the model from disk

    # load the audio file
    features = feature_extractor.extract_features_from_wav_file(audio_path, postprocess=False, shorten=True)


    features = features.flatten()

    pred = model.predict([features])

    return pred[0]
    

def classify_songs(data_path, plot=True):
    '''
    Classifies the songs in the given csv file into one of the 10 genres
    and plots the predictions in a histogram, if plot=True.

    params:
        data_path: path to the csv file containing the features of the songs
        plot: boolean value indicating whether to plot the predictions or not

    returns:
        predictions: predicted genres of the songs
        probabilities: probabilities of the predictions
    '''

    # load genres
    genres = pd.read_csv('vggish_features.csv')['genre']
    
    # label encoding
    le = LabelEncoder()
    genres_int = le.fit_transform(genres)

    # load data
    data = pd.read_csv(data_path, encoding='cp1252')
    features = data.drop(['genre', 'song'], axis=1)

    # load mlp model
    filename = 'mlp_model.sav'
    mlp = pickle.load(open(filename, 'rb'))

    predictions = mlp.predict(features)
    probabilities = mlp.predict_proba(features)

    # plot predictions histogram
    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(15, 10))
        plt.bar(le.classes_, np.bincount(predictions))

        plt.xlabel('Genre')
        plt.ylabel('Count')
        plt.title('Predicted Genre Distribution')

        # mean probability of each genre
        mean_probabilities = {}
        for i in range(10):
            genres = le.inverse_transform([i])
            mean_probabilities[genres[0]] = np.mean(probabilities[:, i])

        # make figure bigger
        plt.figure(figsize=(15, 10))
        plt.bar(mean_probabilities.keys(), mean_probabilities.values())
        plt.xlabel('Genre')
        plt.ylabel('Mean Probability')
        plt.title('Mean Probability of Each Genre')
        plt.show()

    return predictions, probabilities

def extract_features(audios_path, csv_path):
    '''
    Extracts the features from the given audio files using 
    the pre-trained VGGish model and saves them in a csv file.

    params:
        audios_path: path to the folder containing the audio files
        csv_path: path to the csv file to save the features
    '''

    # load audio files
    files = os.listdir('../Audios/' + audios_path)

    feature_columns = ['Feature ' + str(col) for col in range(128 * 31)]
    csv_columns = ['song', 'genre'] + feature_columns

    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(csv_columns)

        for file in files:
            print(file)
            audio_path = f'../Audios/{audios_path}/{file}'
            print(audio_path)
            features = feature_extractor.extract_features_from_wav_file(audio_path, postprocess=False, shorten=True)
            features = features.flatten()
            row = [file, audios_path] + list(features)
            try:
                writer.writerow(row)
            except:
                print("Error writing row")
                continue

