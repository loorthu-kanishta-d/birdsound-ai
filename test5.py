import os
import base64
import numpy as np 
import matplotlib.pyplot as plt 
from glob import glob
import librosa as lr

audio1='b.wav'
y, sr = lr.load('./{}'.format(audio1))
y_fast = lr.effects.time_stretch(y, 2.0)
time = np.arange(0,len(y_fast))/sr
fig, ax = plt.subplots()
ax.plot(time,y_fast)
ax.set(xlabel='Time(s)',ylabel='sound amplitude')

plt.show()

nn=len(time)
print("nn="+str(nn))
i=0
j=1

###########
path_main = 'static/dataset'
for fname in os.listdir(path_main):
    path=path_main+"/"+fname
    audio2=fname
    y2, sr2 = lr.load('./{}'.format(path))
    y_fast2 = lr.effects.time_stretch(y2, 2.0)
    time2 = np.arange(0,len(y_fast2))/sr2
    nn2=len(time2)
    #print(nn2)

    if nn==nn2:
        print(fname)
        print("yes")
        break

    
