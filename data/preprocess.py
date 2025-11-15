import librosa
import  numpy as np
import importlib

# importlib.reload(librosa)

def preprocess_audio(clip, sr_target=16000, duration=4.0):
    y, sr = clip.audio
    y = librosa.to_mono(y)
    y = librosa.resample(y, orig_sr=sr, target_sr=sr_target)
    y = librosa.util.fix_length(y, size=int(sr_target * duration))
    return y, sr_target, clip.class_label, clip.fold

def to_logmel(y, sr, n_mels=64):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    logmel = librosa.power_to_db(mel, ref=np.max)
    # normalize to [0,1]
    logmel = (logmel - logmel.min()) / (logmel.max() - logmel.min() + 1e-6)
    return logmel