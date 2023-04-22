## what are all these files?
Most of these files are from the [tensorflow model repository](https://github.com/tensorflow/models/tree/master/research/audioset/vggish) and are library files that contains methods useful for using the VGGish model in applications. We included these files here to 1. be able to use vggish model to extract features form audio and 2.to document all outside resources we used for vggish feature extraction. These files are:

    1. mel_features.py
    2. vggish_export_tfhub.py
    3. vggish_input.py
    4. vggish_params.py
    5. vggish_postprocess.py
    6. vggish_slim.py

extract_features_from_gtzan.py, extract_features_from_metal.py, vggish_feature_extractor.py are files created by us. We included them here 
it made sense to add them to this folder.

## What is Vggish?
This model uses a pre-trained deep neural network model called Vggish. This model was developed by researchers at Google for processing and extracting data from audios. This model uses multiple convolutional and max-pooling layers and was trained on an earlier version large dataset of audio recordings from Youtube, which later came to be known as [Youtube-8M](https://research.google.com/youtube8m/). It takes in audio samples and outputs a 128 dimension embedding vector for every second of the audio.

## How did we use Vggish?
We used Vggish to extract 3840 features from each song in the [GTZAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) (128 features for each second of a song). We then stored those features in a csv file (vggish_features). We additionally use it to extract the features from any audio and use thos features to compare the audio with the audios from the GTZAN dataset and determine its similarity with other songs in the dataset.