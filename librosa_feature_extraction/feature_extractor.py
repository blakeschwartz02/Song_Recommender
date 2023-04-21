'''
python file to extract features using the librosa package.
To use this file, just change the path to your Project path.
The input to the extractor is the path of your audio file.
'''
import pandas as pd
import os
import librosa 
import numpy as np
import pathlib

os.chdir('C:\\Users\\blake\\OneDrive - The University of Texas at Austin\\Project')

def extract_features_from_wav_file(audio_path):
    features = pd.DataFrame(columns=['title','tempo','zcr_mean','zcr_var','rmse_mean','rmse_var','centroid_mean','centroid_var','bandwidth_mean','bandwidth_var','contrast_mean','contrast_var','rolloff_mean','rolloff_var','flatness_mean','flatness_var','tempogram_ratio_mean','tempogram_ratio_var','harmonic_mean','harmonic_var','percussive_mean','percussive_var','mfcc1_mean','mfcc1_var','mfcc2_mean','mfcc2_var','mfcc3_mean','mfcc3_var','mfcc4_mean','mfcc4_var','mfcc5_mean','mfcc5_var','mfcc6_mean','mfcc6_var','mfcc7_mean','mfcc7_var','mfcc8_mean','mfcc8_var','mfcc9_mean','mfcc9_var','mfcc10_mean','mfcc10_var','mfcc11_mean','mfcc11_var','mfcc12_mean','mfcc12_var','mfcc13_mean','mfcc13_var','mfcc14_mean','mfcc14_var','mfcc15_mean','mfcc15_var','mfcc16_mean','mfcc16_var','mfcc17_mean','mfcc17_var','mfcc18_mean','mfcc18_var','mfcc19_mean','mfcc19_var','mfcc20_mean','mfcc20_var'])
    audio_file = librosa.load(audio_path)
    y, sr = audio_file
    S = librosa.magphase(librosa.stft(y))[0]
    path = pathlib.PurePath(audio_path)
    tempo = librosa.beat.beat_track(y=y, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y)
    rmse = librosa.feature.rms(S=S)
    cent = librosa.feature.spectral_centroid(S=S)
    band = librosa.feature.spectral_bandwidth(S=S)
    contr = librosa.feature.spectral_contrast(S=S)
    roll = librosa.feature.spectral_rolloff(S=S)
    flat = librosa.feature.spectral_flatness(S=S)
    tempogram = librosa.feature.tempogram(y=y, sr=sr)
    tgr = librosa.feature.tempogram_ratio(tg=tempogram, sr=sr)
    harmonic, percussive = librosa.effects.hpss(y=y)
    mfccs = librosa.feature.mfcc(y=y, sr=sr)

    features.title = [os.path.basename(audio_path)]
    features.tempo = [tempo]
    features.zcr_mean = [zcr.mean()]
    features.zcr_var = [zcr.var()]
    features.rmse_mean = [rmse.mean()]
    features.rmse_var = [rmse.var()]
    features.centroid_mean = [cent.mean()]
    features.centroid_var = [cent.var()]
    features.bandwidth_mean = [band.mean()]
    features.bandwidth_var = [band.var()]
    features.contrast_mean = [contr.mean()]
    features.contrast_var = [contr.var()]
    features.rolloff_mean = [roll.mean()]
    features.rolloff_var = [roll.var()]
    features.flatness_mean = [flat.mean()]
    features.flatness_var = [flat.var()]
    features.tempogram_ratio_mean = [tgr.mean()]
    features.tempogram_ratio_var = [tgr.var()]
    features.harmonic_mean = [harmonic.mean()]
    features.harmonic_var = [harmonic.var()]
    features.percussive_mean = [percussive.mean()]
    features.percussive_var = [percussive.var()]

    i = 0
    for c in mfccs:
        features['mfcc'+str(i+1)+'_mean'] = [np.mean(c)]
        features['mfcc'+str(i+1)+'_var'] = [np.var(c)]
        i += 1
    
    return features
    