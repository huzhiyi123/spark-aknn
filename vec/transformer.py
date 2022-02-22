import librosa
import numpy as np
import os

win_length=400
hop_length=80
n_fft=256

def wav2spec(wav, n_fft, win_length, hop_length, time_first=True):
    stft = librosa.stft(y=wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mag = np.abs(stft)
    phase = np.angle(stft)

    if time_first:
        mag = mag.T
        phase = phase.T
    return mag, phase


input_path = "english_small/test"
def wav2vec(path):
    mylist=[]
    namelist=[]
    for file in os.listdir(input_path):
        filename = os.fsdecode(file)
        curpath = os.path.join(input_path, filename)
        wav, _ = librosa.load(curpath, sr=16000 )
        vec,b = wav2spec(wav,n_fft,hop_length,win_length)
        mylist.append(vec[0].tolist())
        namelist.append(filename)
    return np.array(mylist),namelist
