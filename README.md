# Song_Recommender
## Abstract
We propose methods to classify and recommend songs to a user based on an input song. We used two datasets, including GTZAN and a self-generated dataset. Features were extracted directly from the WAV files using two different methods to be used in various models. After feature engineering, multiple models were evaluated to analyze the tradeoff between effectiveness of the model and interpretability. 
## Background
Dataset: This project used two separate datasets. First, the GTZAN dataset which consists of 1000 songs from 10 different genres was used to train classifiers and analyze audio similarity scores. Then, a dataset consisting of 2002 metal songs from 11 metal subgenres was created to analyze the models’ effectiveness with subgenre classification. 
<br><br>
Metrics: There are multiple issues in defining an adequate metric in this problem. For unsupervised methods, there is no way to quantify the accuracy of the recommended songs without user input or extensive metadata. While traditional classification metrics can be used for the supervised models, a lot of genre definitions are subjective and overlapping - making the classification of a song controversial, even among humans. Despite this, the team chose accuracy as the metric of choice for classification.
<br><br>
Audio: The audio used consisted of WAV files. The Waveform Audio File Format is an uncompressed format for storing audio bitstreams. As a result, the WAV format is considered “lossless” and provides high quality audio, but the files are also very large. Therefore, features must be extracted from the large files to make the data manageable.

## Instructions to use code
### GTZAN models
The Data folder contains all necessary files for models run on the GTZAN dataset. These files are GTZAN_features_supervised, GTZAN_features_unsupervised, k_means_clustering, and PCA. Simply change the directory location to the feature files located in this folder, and the code will successfully run. 
### Librosa and VGGish models
The librosa and VGGish models used separate datasets from GTZAN, but these are too large to be posted here. Instructions for downloading these datasets are as follows: <br>
#1
