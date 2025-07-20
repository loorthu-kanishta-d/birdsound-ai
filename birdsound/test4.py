from scipy.io import wavfile
import numpy as np
from skimage.restoration import denoise_wavelet
import matplotlib.pyplot as plt

Fs, x = wavfile.read('h.wav') # Reading audio wave file

x = x/max(x)   # Normalizing amplitude


sigma = 0.05  # Noise variance
x_noisy = x + sigma * np.random.randn(x.size)   # Adding noise to signal

# Wavelet denoising
x_denoise = denoise_wavelet(x_noisy, method='VisuShrink', mode='soft', wavelet_levels=3, wavelet='sym8', rescale_sigma='True')


plt.figure(figsize=(20, 10), dpi=100)
plt.plot(x_noisy)
plt.plot(x_denoise)
fn="p3.png"
#plt.savefig('static/graph/'+fn)
plt.show()
