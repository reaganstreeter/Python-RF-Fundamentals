import numpy as np
import matplotlib.pyplot as plt

# Defining the raised cosine pulse as a function, with arguments of t, T and alpha
# The use of the function assists in handling the special cases of the RC pulse in 
# its time domain form (td)
def raised_cosine_td(t, T, alpha):

    # When alpha = 0, the RC pulse returns to sinc
    if alpha == 0.0:
        return np.sinc(t / T)
    
    # General case: RC = sinc(t/T) * cos(pi*alpha*t/T) / (1 - (2*alpha*t/T)^2)
    # Compute denominator argument of the RC function
    denom_arg = (2 * alpha * t / T) ** 2        # (2*alpha*t/T)^2
    # Compute numerator of the RC function
    numerator  = np.cos(np.pi * alpha * t / T)
    # Compute denominator of the RC function using the denominator argument
    denominator = 1.0 - denom_arg
    
    # There are two special cases for the RC function:
    # Special case 1 is when t = 0 whereby sinc(0) = 1, second term goes to 1
    # This is handled implicitly by np.sinc and for this purpose can be ignored in the code
    # Special case 2 is when t = +/-T/(2*alpha) whereby both numerator and denominator
    # tend to 0, which would lead to a computational error when running the code if not
    # appropriately handled. 
    # L'Hopital rule can be used to determine that the value of the RC function at the special
    # values of t is: (pi/4) * sin(pi / (2*alpha))
    limit_value = (np.pi / 4.0) * np.sin(np.pi / (2.0 * alpha))

    # Identify the problematic points, i.e., where denominator tends to zero
    tend_to_zero = np.abs(denominator) < 1e-10

    # The below substitutes 1.0 into the denominator at the points in t where the RC
    # function would tend to zero
    safe_denom = np.where(tend_to_zero, 1.0, denominator)

    # Compute the full expression using safe_denom so there are no computational errors
    raisedcosine_td = np.sinc(t / T) * numerator / safe_denom

    # Replace the values at t = tend_to_zero points with the correct limiting value
    raisedcosine_td = np.where(tend_to_zero, limit_value, raisedcosine_td)

    return raisedcosine_td

# Defining the raised cosine frequency domain representation as a function, 
# with arguments of f, T and alpha
def raised_cosine_fd(f, T, alpha):
    # The RC function's frequency domain representation can be considered in terms of
    # a piecewise definition, consisting of three regions:
    # A passband region exists for |f| <= (1-alpha)/2T whereby RC_fd = T
    # A cosine rolloff transition region for (1-alpha)/2T < |f| <= (1+alpha)/2T
    # Where RC_fd = (T/2)*(1 + cos((pi*T/alpha)(|f| - (1-alpha)/2*T)))  -> cosine rolloff
    # A stopband region exists for |f| > (1+alpha)/2T whereby RC_fd = 0
    
    # f_abs is computed so that the signage of the elements in f = np.linspace() dont
    # influence the calculation of RC_fd as specified above
    f_abs = np.abs(f)
    
    # The passband edge is located at:
    f1 = (1.0 - alpha) / (2.0 * T)
    # The stopband edge is located at:
    f2 = (1.0 + alpha) / (2.0 * T)

    # For alpha = 0, there is no transition region, so f_cutoff is located at:
    f_c = 1/(2*T)

    if alpha == 0.0:
        # The spectrum becomes rectangular, so:
        RC_FD = np.where(np.abs(f) <= f_c, T, 0.0)
        return RC_FD

    # Transition band values for alpha > 0
    transition_band = (T / 2.0) * (1.0 + np.cos((np.pi * T / alpha) * (f_abs - f1)))

    # Implementing RC_FD as per the 3 part piecewise definition described previously
    RC_FD = np.where(f_abs <= f1, T, np.where(f_abs <= f2, transition_band, 0.0))
    return RC_FD

# Defining an arbitrary symbol period of 1 second
T = 1.0

# Defining a set of rolloff values, with alpha = 0 equating to the sinc case
# alpha = 1 gives maximum mainlobe bandwidth and maximum timing-error tolerance
alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

# Creating a time vector with +/- 8 symbol periods to fully visualise
# lobes to show decay behaviour across different values of alpha
t = np.linspace(-8*T, 8*T, 2000)

# Creating a frequency vector for ploting the frequency-domain RC pulse
# RC_fd has its bandwidth edge at f = (1+alpha)/2T, so where the frequency units are in terms
# of ft, this becomes fT = (1+alpha)/2, for alpha = 1, fT = 1. To give a visual margin for
# plotting, +/- 0.2 is added, so the bounds are -1.2 to 1.2
f = np.linspace(-1.2, 1.2, 2000)

# Different colours required to differentiate the varying rolloff values
colours = ['black', 'blue', 'red', 'green', 'orange']

# Below, the time-domain form of the RC pulse is plotted with the varying rolloff values
fig1, ax1 = plt.subplots(figsize = (10,5))


# Plotting the RC pulse in the time domain for varying rolloff values
for alpha, colour in zip(alphas, colours):
    RC_td = raised_cosine_td(t, T, alpha)
    # Plotted against t / T to normalise x-axis to the symbol period
    ax1.plot(t / T, RC_td, label = '$\\alpha$ = ' + str(alpha), color = colour, linewidth = 1.5)

# Marking the zero crossings at non-zero integer multiples of T
# these are shared across all values of alpha, i.e., it is the fundamental 
# zero ISI property of the raised cosine function
for n in range(-8, 9):
    if n != 0:
        ax1.axvline(x=n, color='grey', linewidth=0.5, linestyle=':', alpha=0.4)

# Annotating the zero-ISI property once, with labels pointing to t = T
ax1.annotate('Zero crossings at $t = nT$\n(ISI-free for all $\\alpha$)',
             xy = (1, 0), xytext = (2.2, 0.35),
             arrowprops = dict(arrowstyle = '->', color = 'black'),
             fontsize = 9, color = 'black')

# Axes for visual purposes
ax1.axhline(0, color = 'black', linewidth = 0.7)
ax1.axvline(0, color = 'black', linewidth = 0.7)

# Axes labelling, plot title and including a legend
ax1.set_xlabel('Normalised Time (t/T)', fontsize = 11)
ax1.set_ylabel('Amplitude', fontsize = 11)
ax1.set_title('Raised Cosine Function Time Domain Representation', fontsize = 12)
ax1.legend(fontsize = 10)

# Visual clean up
ax1.set_xlim(-8, 8)
ax1.set_ylim(-0.3, 1.15)
ax1.grid(True, linewidth = 0.4, alpha = 0.5)

# Below, the frequency-domain form of the RC pulse is plotted with the varying rolloff values
fig2, ax2 = plt.subplots(figsize = (10, 5))

# Plotting the RC pulse in the time domain for varying rolloff values 
for alpha, colour in zip(alphas, colours):
    RC_fd = raised_cosine_fd(f, T, alpha)
    # RC_fd / T normalises the magnitude of the passband
    ax2.plot(f, RC_fd / T, label = '$\\alpha$ = ' + str(alpha), color = colour, linewidth = 1.8)

    # Annotating the stopband edge, i.e., the bandwidth, for each alpha
    f_stop = (1.0 + alpha) / (2.0 * T)
    ax2.axvline(x = f_stop, color = colour, linewidth = 0.6, linestyle = '--', alpha = 0.5)
    ax2.axvline(x = -f_stop, color = colour, linewidth = 0.6, linestyle = '--', alpha = 0.5)

# Annotating the bandwidth of a pulse
ax2.annotate('Bandwidth $= \\frac{1+\\alpha}{2T}$' + '\n(increases with $\\alpha$)',
             xy=(0.625, 0.5), xytext=(0.6, 0.75),
             arrowprops=dict(arrowstyle='->', color='black'),
             fontsize=9)

# Axes for visual purposes 
ax2.axhline(0, color='black', linewidth=0.7)
ax2.axvline(0, color='black', linewidth=0.7)

# Setting axes labels, plot title and including a legend
ax2.set_xlabel('Normalised Frequency (fT)', fontsize=11)
ax2.set_ylabel('Normalised Magnitude', fontsize=11)
ax2.set_title('Raised Cosine Frequency Domain Representation', fontsize=12)
ax2.legend(fontsize=10)

# Visual clean up
ax2.set_xlim(-1.2, 1.2)
ax2.set_ylim(-0.05, 1.15)
ax2.grid(True, linewidth=0.4, alpha=0.5)

plt.tight_layout()
plt.show()