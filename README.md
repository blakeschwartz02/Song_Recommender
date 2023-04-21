# Song_Recommender
## Abstract
We propose methods to classify and recommend songs to a user based on an input song. We used two datasets, including GTZAN and a self-generated dataset. Features were extracted directly from the WAV files using two different methods to be used in various models. After feature engineering, multiple models were evaluated to analyze the tradeoff between effectiveness of the model and interpretability. 
## Background
Dataset: This project used two separate datasets. First, the GTZAN dataset which consists of 1000 songs from 10 different genres was used to train classifiers and analyze audio similarity scores. Then, a dataset consisting of 2002 metal songs from 11 metal subgenres was created to analyze the models’ effectiveness with subgenre classification. 
The GTZAN dataset is small enough to have in the repository, however the metal subgenre dataset is far too large (about 100 GB). There are instruction on how to download songs to construct this dataset at the end of the file. Furthermore, the file containing the VGGish features of the metal songs was too large for the repository (about 350 MB). There are also instruction on how to run the VGGish model to extract the features from the metal subgenre dataset at the end of this file. The librosa extracted features of the metal songs is in the Data folder.
<br><br>
Metrics: There are multiple issues in defining an adequate metric in this problem. For unsupervised methods, there is no way to quantify the accuracy of the recommended songs without user input or extensive metadata. While traditional classification metrics can be used for the supervised models, a lot of genre definitions are subjective and overlapping - making the classification of a song controversial, even among humans. Despite this, the team chose accuracy as the metric of choice for classification.
<br><br>
Audio: The audio used consisted of WAV files. The Waveform Audio File Format is an uncompressed format for storing audio bitstreams. As a result, the WAV format is considered “lossless” and provides high quality audio, but the files are also very large. Therefore, features must be extracted from the large files to make the data manageable.
<br><br>
## VGGish
VGGish is a pretrained Convolutional Neural Network created by Google that we used as a method of feature extraction. For more information please see the README.md file located in the vggish_feature_extraction folder.

## Instructions to use code
### Feature Extraction
To run the librosa feature extraction on the GTZAN/metal dataset:
Run: librosa_feature_extraction/extract_features_from_dataset.ipynb
To run the VGGish feature extraction on the GTZAN dataset: 
Run: vggish_feature_extraction/extract_features_from_gtzan.py
To run the VGGish feature extraction on the metal dataset: 
Run: vggish_feature_extraction/extract_features_from_metal.py
### Feature Analysis
To analyze the features with PCA and T-SNE, run: feature_analysis/PCA.ipynb
### Classification models
The classification_models folder contains all necessary files for models run on the GTZAN and metal dataset. These files are GTZAN_features_supervised, librosa_metal_classification, and vggish_metal_classification.
Also, to run some unsupervised clustering algorithms, the folder 'Test Code' has two unsupervised files. 
Run k_means_clustering.py and GTZAN_features_unsupervised.ipynb for k-means clustering algorithms and further PCA analysis.
### Downloading a Metal Dataset
First, install youtube-dl. 
Create a folder structure with multiple subgenres. These subgenres used in the project were:
Alternative, Death, Folk, Glam, Industrial, Metalcore, Nu, New Wave of British Heavy Metal, Progressive, Symphonic, Thrash
On youtube, find a playlist for a subgenre with at least 200 songs and copy the link of the playlist.
In a terminal, change your current directory to one a subgenre's folder and run the command:
youtube-dl -x --audio-format wav --yes-playlist --playlist-end 200 'link to the youtube playlist'
This will download the first 200 songs of the playlist. Find a playlist and run the command for each subgenre.

