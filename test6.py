import soundfile
import numpy as np

signal_1, samplerate_1 = soundfile.read("a.wav")
signal_2, samplerate_2 = soundfile.read("b.wav")

mismatch = (signal_1 != signal_2).astype(np.int64)

print(mismatch)
