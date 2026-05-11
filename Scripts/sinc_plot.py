import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# x vector created for plotting purposes
# Range chosen -4pi to 4pi to show 4 zero crossing either side of x = 0
x = np.linspace(-4 * np.pi, 4 * np.pi, 2000)

# Numpy's in-built sinc function is naturally the normalised sinc function,
# however, for clarity, the definition is sin(x)/(pi*x)
sinc_norm = np.sinc(x)

# Conventional definition of unnnormalised sinc function
#sinc_unnorm = np.where(x == 0, 1.0, np.sin(x)/x)
# Alternatively, the above can be handled as:
sinc_unnorm = np.sinc(x/np.pi)
# The key difference between normalised and unnormalised sinc(x) functions
# is that unnormalised has zeros at integer multiples of pi, whereas the
# normalised sinc function horizontally scales the zeros to non-zero integers

fig1, ax1 = plt.subplots(figsize=(10, 5))

ax1.plot(x, sinc_norm,   label = r'Normalised: $\mathrm{sinc}(x) = \sin(\pi x)/(\pi x)$', 
        color = 'blue', linewidth = 1.8)
ax1.plot(x, sinc_unnorm, label = r'Unnormalised: $\mathrm{sinc}(x) = \sin(x)/x$', 
        color = 'red', linewidth = 1.8, linestyle = '--')

# Mark zero crossings of the normalised sinc, i.e., integers excluding 0
norm_zeros = [n for n in range(-4, 5) if n != 0]
for z in norm_zeros:
    ax1.axvline(x = z, color='blue', linewidth = 1, linestyle = ':', alpha = 0.5)

# Mark zero crossings of the unnormalised sinc, i.e., multiples of π excluding 0
unnorm_zeros = [n * np.pi for n in range(-4, 5) if n != 0]
for z in unnorm_zeros:
    ax1.axvline(x = z, color = 'red', linewidth = 1, linestyle = ':', alpha = 0.5)

# Annotate one zero crossing on each curve to state the general rule clearly
ax1.annotate('Zeros at $x = n$\n(non-zero integers)',
            xy = (1, 0), xytext = (1.4, 0.28),
            arrowprops = dict(arrowstyle = '->', color = 'blue'),
            color = 'blue', fontsize = 9)

ax1.annotate(r'Zeros at $x = n\pi$' + '\n' + r'(non-zero multiples of $\pi$)',
            xy = (np.pi, 0), xytext = (4.5, -0.28),
            arrowprops = dict(arrowstyle = '->', color = 'red'),
            color = 'red', fontsize = 9)

# Setting x-axis reference line
ax1.axhline(0, color = 'black', linewidth = 0.7)
# Setting y-axis reference line
ax1.axvline(0, color = 'black', linewidth = 0.7)
# Setting x-axis label
ax1.set_xlabel('x', fontsize = 11)
# Setting arbitrary y-axis amplitude label
ax1.set_ylabel('Amplitude', fontsize = 11)
# Setting plot title relevant to depicted plots
ax1.set_title('Normalised vs Unnormalised Sinc Function', fontsize = 12)
ax1.legend(fontsize = 10)
# Setting bounds on y-axis
ax1.set_ylim(-0.3, 1.1)
# Activating a background grid for visual clarity
ax1.grid(True, linewidth = 0.4, alpha = 0.5)

# Setting a sampling rate, in units of Hz
fs = 100
# Setting the sample spacing
dt = 1/fs
# Setting the total number of points across the FFT
N = 2000
# Creating a symmetric time vector centred at 0
t = np.arange(-N//2, N//2) * dt
# Setting the cut-off frequency of the sinc signal, in units of Hz
fc = 10
# Creating a second instance of sinc_norm, but applied to t above
sinc = np.sinc(2*fc*t)
# Computing complex FFT of the sinc function
sinc_FFT = fft(sinc)
# Creating the frequency axis
freqs = fftfreq(N, d = 1/fs)
pos_mask = freqs >= 0

# Converting the FFT from linear to decibel scale
sinc_FFT_mag = np.abs(sinc_FFT[pos_mask]) / np.max(np.abs(sinc_FFT))

# Creating a second plot with new axes
fig2, ax2 = plt.subplots(figsize=(10, 5))

ax2.plot(freqs[pos_mask], sinc_FFT_mag , label = r'FFT of normalised sinc', color = 'blue', linewidth = 1.8)
# Setting x-axis label
ax2.set_xlabel('Frequency (Hz)', fontsize = 11)
# Setting y-axis magnitude label
ax2.set_ylabel('Normalised Magnitude (linear)', fontsize = 11)
# Setting plot title relevant to depicted plot
ax2.set_title('FFT of Normalised Sinc Function', fontsize = 12)

plt.tight_layout()
plt.show()
