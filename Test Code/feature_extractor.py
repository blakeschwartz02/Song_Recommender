import pandas as pd
import os
import librosa 
import numpy as np
import pathlib

os.chdir('C:\\Users\\blake\\OneDrive - The University of Texas at Austin\\Project')

def extract_features_from_wav_file(audio_path):
    features = []
    audio_file = librosa.load(audio_path)
    y, sr = audio_file
    S = librosa.magphase(librosa.stft(y))[0]
    path = pathlib.PurePath(audio_path)
    title = path.name
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

    features.extend(title, tempo, zcr.mean(), zcr.var(), rmse.mean(), rmse.var(), cent.mean(), cent.var(), band.mean(), band.var(), 
                    contr.mean(), contr.var(), roll.mean(), roll.var(), flat.mean(), flat.var(), tgr.mean(), tgr.var(), 
                    harmonic.mean(), harmonic.var(), percussive.mean(), percussive.var())

    i = 0
    for c in mfccs:
        features.append(np.mean(c))
        features.append(np.var(c))
        i += 1
    
    return features
    