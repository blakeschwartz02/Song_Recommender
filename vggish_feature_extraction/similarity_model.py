'''
Similarity scores:
    - cosine similarity
    - euclidean distance
    - manhattan distance
'''

# standard imports
import sys
import pandas as pd
import numpy as np

# similarity score imports
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
from scipy.spatial.distance import correlation

# vggish imports
import imports.vggish_feature_extractor as extractor


NUM_ARGS = 3
TOTAL_NUM_SONGS = 100*10

# Similarity score functions
similarity_scores_measures = ["cosine_similarity", "euclidean_distance", "manhattan_distance", "correlation_distance"]

if(len(sys.argv) != NUM_ARGS+1):
    #print to stderr
    print('Invalid arguments. Usage: python similarity_model.py <audio_path> <similarity_score>', file=sys.stderr)
    sys.exit()

audio_path = sys.argv[1]
similarity_score = sys.argv[2]

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


songs = pd.read_csv('vggish_features.csv')
song_features = songs.drop(['song', 'genre'], axis=1)
similarity_scores = {}

# extract features from audio file
audio_features = extractor.extract_features_from_wav_file(audio_path, postprocess=False)
audio_features = audio_features.flatten()

for index, row in song_features.iterrows():
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

print(f'\nTop {num_songs} songs: ', list(similarity_scores.keys())[:num_songs])