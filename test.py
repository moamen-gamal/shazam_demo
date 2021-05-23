import matplotlib.pyplot as plt
import librosa
from numpy.core.fromnumeric import ptp 
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


src ="Ghaliaa.mp3"
dsk= "generated.wav"
sound=AudioSegment.from_mp3(src)
sound.export(dsk,format="wav")
wavsong,samplingFrequency =librosa.load(dsk)
#print(samplingFrequency)
#print(wavsong)

#display Spectrogram

X = librosa.stft(wavsong)
Xdb = librosa.amplitude_to_db(abs(wavsong))
plt.figure()
Data = librosa.amplitude_to_db(np.abs(librosa.stft(wavsong)), ref=np.max)
librosa.display.specshow(Data, y_axis='linear', x_axis='time',sr=samplingFrequency)
# #If to pring log of frequencies  
# #librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()
plt.show()


idx = slice(*librosa.time_to_frames([0, 60], sr=samplingFrequency))
plt.figure(figsize=(12, 4))
librosa.display.specshow(librosa.amplitude_to_db(X[:, idx], ref=np.max),
                         y_axis='log', x_axis='time', sr=samplingFrequency)
plt.colorbar()
plt.tight_layout()
plt.show()

S_full, phase = librosa.magphase(librosa.stft(wavsong))
idx = slice(*librosa.time_to_frames([0, 60], sr=samplingFrequency))
plt.figure(figsize=(12, 4))
librosa.display.specshow(librosa.amplitude_to_db(S_full[:, idx], ref=np.max),
                         y_axis='log', x_axis='time', sr=samplingFrequency)
plt.colorbar()
plt.tight_layout()
plt.show()

# and works well to discard vocal elements.

S_filter = librosa.decompose.nn_filter(S_full,
                                       aggregate=np.median,
                                       metric='cosine',
                                       width=int(librosa.time_to_frames(2, sr=samplingFrequency)))

# The output of the filter shouldn't be greater than the input
# if we assume signals are additive.  Taking the pointwise minimium
# with the input spectrum forces this.
S_filter = np.minimum(S_full, S_filter)

margin_i, margin_v = 2, 10
power = 2

mask_i = librosa.util.softmask(S_filter,
                               margin_i * (S_full - S_filter),
                               power=power)

mask_v = librosa.util.softmask(S_full - S_filter,
                               margin_v * S_filter,
                               power=power)

# Once we have the masks, simply multiply them with the input spectrum
# to separate the components

S_foreground = mask_v * S_full
S_background = mask_i * S_full
print(type(S_foreground))

plt.figure(figsize=(12, 8))
#plt.subplot(3, 1, 1)
# librosa.display.specshow(librosa.amplitude_to_db(S_full[:, idx], ref=np.max),
#                           y_axis='log', sr=samplingFrequency)
# plt.title('Full spectrum')
# plt.colorbar()
# plt.show()
# plt.subplot(3, 1, 2)
# librosa.display.specshow(librosa.amplitude_to_db(S_background[:, idx], ref=np.max),
#                           y_axis='log', sr=samplingFrequency)
# plt.title('Background')
# plt.colorbar()
# plt.subplot(3, 1, 3)
librosa.display.specshow(librosa.amplitude_to_db(S_foreground[:, idx], ref=np.max),
                         y_axis='log', x_axis='time', sr=samplingFrequency)
plt.title('Foreground')
plt.colorbar()
plt.tight_layout()
plt.show()