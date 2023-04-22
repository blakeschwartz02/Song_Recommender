# Song Similarity Finder Based on Extracted Features Using Vggish Pre-trained Model

## What is Vggish?
This model uses a pre-trained deep neural network model called Vggish. This model was developed by researchers at Google for processing and extracting data from audios. This model uses multiple convolutional and max-pooling layers and was trained on an earlier version large dataset of audio recordings from Youtube, which later came to be known as [Youtube-8M](https://research.google.com/youtube8m/). It takes in audio samples and outputs a 128 dimension embedding vector for every second of the audio.

## How did we use Vggish?

We used Vggish to extract 3840 features from each song in the [GTZAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) (128 features for each second of a song). We then stored those features in a csv file (vggish_features). We additionally use it to extract the features from any audio and use thos features to compare the audio with the audios from the GTZAN dataset and determine its similarity with other songs in the dataset.

## Similarity Score Model

This model extracts the features of a given audio using the Vggish model and determines how similar that audio is to every other audio in the GTZAN dataset by using a particular similarity score. It then prints the top k similar songs to the given audio.


### Usage

you can get the top k similar songs by doing the following:

1. make this directory (vggish_feature_extraction) your working directory.

2. Download [VGGish model checkpoint](https://storage.googleapis.com/audioset/vggish_model.ckpt) and [Embedding PCA parameters](https://storage.googleapis.com/audioset/vggish_pca_params.npz) and put them in the `imports` directory.

3. Save a .wav file of the audio you want to use (e.g. Numb_LinkinPark.wav) somewhere in your computer.

4. run the python script:
    ```
    python similarity_model.py <path/to/audio/wav> <similarity score> <number of similar songs>
    ```

5. The script will then print to the console the top "number of similar songs" songs for the given audio.

#### Available Similarity Scores
1. "cosine_similarity"
2. "euclidean_distance"
3. "manhattan_distance"
4. "correlation_distance"
