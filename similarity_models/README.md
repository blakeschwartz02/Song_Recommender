# Song Similarity Finder Based on Extracted Features Using Vggish Pre-trained Model

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
