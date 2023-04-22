from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf

import os
import sys
sys.path.append(os.getcwd() + '/vggish_imports')

import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim

import librosa



# Paths to downloaded VGGish files.
checkpoint_path = 'vggish_imports/vggish_model.ckpt'
pca_params_path = 'vggish_imports/vggish_pca_params.npz'

def extract_features_from_wav_file(audio_path, postprocess=True, shorten=True):
    # Create input batch for Vggish model from audio file
    input_batch = None
    if shorten:
        waveform, sr = librosa.load(audio_path, duration=30, sr=vggish_params.SAMPLE_RATE)
        input_batch = vggish_input.waveform_to_examples(waveform, sr)

    else:
        input_batch = vggish_input.wavfile_to_examples(audio_path)

    print(input_batch.shape)

    with tf.Graph().as_default(), tf.Session() as sess:
        vggish_slim.define_vggish_slim()                                # define the vggish model
        vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)  # load the checkpoint of the model

        features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)    # get the input tensor
        embedding_tensor = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)  # get embedding tensor
        [embedding_batch] = sess.run([embedding_tensor],feed_dict={features_tensor: input_batch})  # run the batch through the model to produce embeddings

    if(postprocess):
        pproc = vggish_postprocess.Postprocessor(pca_params_path)
        postprocessed_batch = pproc.postprocess(embedding_batch)
        embedding_batch = postprocessed_batch

    return embedding_batch
