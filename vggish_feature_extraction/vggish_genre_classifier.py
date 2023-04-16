
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


def train_CNN_model(X_train, Y_train, X_test, Y_test):
    
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

    return model



def train_mlp_model(X_train, Y_train, X_test, Y_test):
    mlp = MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=500, alpha=0.001,solver='sgd', verbose=0,  random_state=21,tol=0.000000001, activation='relu', learning_rate='adaptive')
    mlp.fit(X_train, Y_train)
    return mlp

def train_logistic_model(X_train, Y_train, X_test, Y_test):
    logreg = LogisticRegression()
    logreg.fit(X_train, Y_train)
    return logreg

def calculate_accuracy(model, X, Y):
    accuracy = accuracy_score(Y, model.predict(X))
    return accuracy