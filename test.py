import matplotlib.pyplot as plt
import librosa 
from pydub import AudioSegment
from tempfile import mktemp
import sklearn
import librosa.display
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
import os
from os import path
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
import pylab


src ="LetMeLoveYou.mp3"
dsk= "generated.wav"
sound=AudioSegment.from_mp3(src)
sound.export(dsk,format="wav")
wavsong,samplingFrequency =librosa.load(dsk)
print(samplingFrequency)
print(wavsong)

#display Spectrogram
# X = librosa.stft(wavsong)
# Xdb = librosa.amplitude_to_db(abs(wavsong))
# plt.figure()
# D = librosa.amplitude_to_db(np.abs(librosa.stft(wavsong)), ref=np.max)
#librosa.display.specshow(D, y_axis='linear', x_axis='time',sr=samplingFrequency)
# #If to pring log of frequencies  
# #librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
# plt.colorbar()