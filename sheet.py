import pyaudio
import sys
import wave
import librosa
import glob
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import librosa.display
# import IPython.display as ipd

# audio = pyaudio.PyAudio()

# FORMAT = pyaudio.paInt16
# CHANNELS =1
# RATE = 44100
# CHUNKS = 1024

# stream = audio.open(format = FORMAT, 
#                     channels = CHANNELS,
#                     rate = RATE, 
#                     input = True,
#                     frames_per_buffer=CHUNKS)

# frames = []
# seconds =10


# print("..start recording")
# for j in range(0, seconds):
#     for i in range(0, int((RATE/CHUNKS))):
#         data = stream.read(CHUNKS)
#         frames.append(data)

#     print(j+1," seconds...")

# stream.stop_stream()
# stream.close()
# audio.terminate()

# wf = wave.open("chaoscanine.wav","wb")
# wf.setnchannels(CHANNELS)
# wf.setsampwidth(audio.get_sample_size(FORMAT))
# wf.setframerate(RATE)
# wf.writeframes(b''.join(frames))
# wf.close()

y, sr = librosa.load("chaoscanine.wav")

# pd.Series(y).plot(figsize=(10,5),lw = 1)
# plt.show()

D = librosa.cqt(y, sr =sr)
harmonic, percussive = librosa.decompose.hpss(D)
SDB = librosa.amplitude_to_db(np.abs(harmonic), ref = np.max)

df = pd.DataFrame(SDB)
df.to_csv('AmplitudeChaos.csv', index=False)

onset_env = librosa.onset.onset_strength(y=y, sr=sr)
onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
onset_times = librosa.frames_to_time(onset_frames, sr=sr)

fig, ax = plt.subplots(figsize=(10,5))
img = librosa.display.specshow(SDB,x_axis="time", y_axis="cqt_note",ax = ax,cmap='magma', bins_per_octave=12)

x=0
arr=[]
for onset in onset_times:

    ax.axvline(onset, color='cyan', linestyle='--', linewidth=1)

notes = librosa.cqt_frequencies(n_bins=SDB.shape[0], fmin=librosa.note_to_hz('C1'))
for frame in onset_frames:
    x+=1
    maxj = -81
    maxo = -1
    note = ""
    for i in range(len(SDB[:,frame])):
        if SDB[i,frame]>=maxj:
            maxj = SDB[i,frame]
            maxo = i
    freq = notes[maxo]
    note = librosa.hz_to_note(freq)

    print("Onset: ", x," :", note," ",maxj)

print(onset_frames)
plt.show()
