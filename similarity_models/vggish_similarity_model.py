'''
This file contains methods for finding top k 
most acoustically similar songs for a given song.

Similarity scores:
    - cosine similarity
    - euclidean distance
    - manhattan distance
'''

# standard imports
import sys
import pandas as pd
import numpy as np
import os

# similarity score imports
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
from scipy.spatial.distance import correlation
from sklearn.model_selection import train_test_split

# vggish imports
sys.path.append(os.getcwd() + '/vggish_imports')
import vggish_imports.vggish_feature_extractor as extractor
sys.path.append(os.getcwd() + '/classification_models')
import classification_models.vggish_genre_classifier as classifier

NUM_ARGS = 3
TOTAL_NUM_SONGS = 100*10

# Similarity score functions
similarity_scores_measures = ["cosine_similarity", "euclidean_distance", "manhattan_distance", "correlation_distance"]

def findTopKSimilarSongsFromFeatures(features, similarity_score, k, print_results=True):
    '''
    finds the top k most similar songs to a song given its extracted vggish features and similarity score.
    Optionally, you can print the results on the screen by setting print_results to True.

    Parameters:
        features (list): list of vggish features for a song
        similarity_score (str): similarity score to use
        k (int): number of similar songs to return
        print_results (bool): whether to print the results on the screen or not

    Returns:
        list: list of top k most similar songs
    '''

    songs = pd.read_csv('vggish_features.csv')
    song_features = songs.drop(['song', 'genre'], axis=1)
    similarity_scores = {}

    for index, row in song_features.iterrows():
        score = None
        if similarity_score == "cosine_similarity":
            score = cosine_similarity([features], [row])[0][0]

        elif similarity_score == "euclidean_distance":
            score = euclidean_distances([features], [row])[0][0]

        elif similarity_score == "manhattan_distance":
            score = manhattan_distances([features], [row])[0][0]

        elif similarity_score == "correlation_distance":
            score = correlation(features, row)

        else:
            print('Similarity score not implemented yet', file=sys.stderr)
            sys.exit()

        similarity_scores[songs['song'][index]] = score

    # sort the scores
    if(similarity_score == "cosine_similarity"):
        similarity_scores = {k: v for k, v in sorted(similarity_scores.items(), key=lambda item: item[1], reverse=True)}

    elif(similarity_score == "euclidean_distance" or similarity_score == "manhattan_distance" or similarity_score == "correlation_distance"):
        similarity_scores = {k: v for k, v in sorted(similarity_scores.items(), key=lambda item: item[1])}

    # print the top songs
    if(print_results):
        print(f'\nTop {k} songs: ', list(similarity_scores.keys())[:k])
    
    return list(similarity_scores.keys())[:k]

def findTopKSimilarSongs(audio_path, similarity_score, k, print_results=True):
    '''
    finds the top k most similar songs to a song given the path
    to an audio file, a similarity score, the number of top similar
    songs to find.

    Parameters:
        audio_path (str): path to audio file
        similarity_score (str): similarity score to use
        k (int): number of similar songs to return
        print_results (bool): whether to print the results on the screen or not

    Returns:
        list: list of top k most similar songs
    '''

    # see if audio file exists
    try:
        file = open(audio_path, 'r')
        file.close()
    except FileNotFoundError:
        print('File not found: ', audio_path, file=sys.stderr)
        sys.exit()

    # see if similarity score is valid
    if(similarity_score not in similarity_scores_measures):
        print('Invalid similarity score. Valid similarity scores are: ', similarity_scores_measures, file=sys.stderr)
        sys.exit()

    # see if number of similar songs is valid
    try:
        num_songs = int(sys.argv[3])
    except ValueError:
        print('Invalid number of songs. Must be an integer', file=sys.stderr)
        sys.exit()

    if(num_songs < 1 or num_songs > 100*10):
        print(f'Invalid number of songs. Must be between 1 and {TOTAL_NUM_SONGS}', file=sys.stderr)
        sys.exit()

    # extract features from audio file
    audio_features = extractor.extract_features_from_wav_file(audio_path, postprocess=False)
    audio_features = audio_features.flatten()

    findTopKSimilarSongsFromFeatures(audio_features, similarity_score, k, print_results)

def findTopKSimilarSongsWithinGenre(path_to_audio, similarity_score, k, print_results=True):
    '''
    This methods finds the top k most similar songs to 
    a song given the path to a 30 second wav file of it by
    first identifying its genre and then finding the top k most
    similar songs within that genre.

    parameters:
        path_to_audio (str): path to 30 second wav file of song
        similarity_score (str): similarity score to use
        k (int): number of similar songs to return
        print_results (bool): whether to print the results on the screen or not

    returns:
        list: list of top k most similar songs
    '''

    songs = pd.read_csv('Data/vggish_features.csv')
    song_features = songs.drop(['song', 'genre'], axis=1)
    labels = songs['genre']
    X_train, X_test, y_train, y_test = train_test_split(song_features, labels, test_size=0.2, random_state=42)

    model = classifier.train_mlp_model(X_train, y_train, X_test, y_test)

    # get the genre of the song
    genre = classifier.classify_song(path_to_audio, model)

    print(f'Genre of song: {genre}')

    print(songs['genre'])

    # only get songs from the same genre
    genre_songs_features = songs[songs['genre'] == genre].drop(['song', 'genre'], axis=1)

    audio_features = extractor.extract_features_from_wav_file(path_to_audio, postprocess=False)
    audio_features = audio_features.flatten()

    similarity_scores = {}
    for index, row in genre_songs_features.iterrows():
        score = None
        if similarity_score == "cosine_similarity":
            score = cosine_similarity([audio_features], [row])[0][0]

        elif similarity_score == "euclidean_distance":
            score = euclidean_distances([audio_features], [row])[0][0]

        elif similarity_score == "manhattan_distance":
            score = manhattan_distances([audio_features], [row])[0][0]

        elif similarity_score == "correlation_distance":
            score = correlation(audio_features, row)

        else:
            print('Similarity score not implemented yet', file=sys.stderr)
            sys.exit()

        similarity_scores[songs['song'][index]] = score

     # sort the scores
    if(similarity_score == "cosine_similarity"):
        similarity_scores = {k: v for k, v in sorted(similarity_scores.items(), key=lambda item: item[1], reverse=True)}

    elif(similarity_score == "euclidean_distance" or similarity_score == "manhattan_distance" or similarity_score == "correlation_distance"):
        similarity_scores = {k: v for k, v in sorted(similarity_scores.items(), key=lambda item: item[1])}

    # print the top songs
    if(print_results):
        print(f'\nTop {k} songs: ', list(similarity_scores.keys())[:k])
    
    return list(similarity_scores.keys())[:k]