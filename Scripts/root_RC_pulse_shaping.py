import numpy as np
import matplotlib.pyplot as plt

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

def rrc_td(t, T, alpha):
    # In the case of alpha = 0, the root RC function reduces to sinc
    # and there is no cosine rolloff term
    if alpha == 0.0:
        return np.sinc(t / T) / np.sqrt(T)

    # The root raised cosine pulse in the time domain has the following form:
    # h_RRC(t) = (1/sqrt(T)) * (4*alpha/pi) * 
    #            (cos((1+alpha)*pi*t/T) + sin((1-alpha)*pi*t/T) / (4*alpha*t/T)) / 
    #            (1 - (4*alpha*t/T)^2)
    # There are three special cases that need explicit handling for the RRC function:
    # Special case 1 is when t = 0 and the RRC function tends to:
    # (1/sqrt(T)) * (1 + alpha*(4/pi - 1))
    # Special case 2 is when t = +/-T/4a and the RRC function tends to:
    # (alpha/sqrt(2T)) * [(1+2/pi)*sin(pi/4a) + (1-2/pi)*cos(pi/4a)]

    # The below represents time normalised to the symbol period
    t_norm = t / T
    # The below represents the term 4*alpha*t/T which appears frequently
    # in the RRC function
    a4t    = 4.0 * alpha * t_norm

    # Terms that appear in the numerator of the RRC function
    cos_term = np.cos((1.0 + alpha) * np.pi * t_norm)
    sin_term = np.sin((1.0 - alpha) * np.pi * t_norm)

    # Term that appears in the denominator of the RRC function
    denominator = 1.0 - a4t ** 2

    # The below handles special case 1 mentioned previously by setting the 
    # correct limiting value at t = 0
    limit_t0 = (1.0 / np.sqrt(T)) * (1.0 + alpha * (4.0 / np.pi - 1.0))

    # The below handles special case 2, i.e., when t = +/-T/(4*alpha)
    # Since both numerator and denominator tend to 0 simultaneously giving 
    # # an indeterminate (0/0) form, the denominator needs to be guarded to
    # prevent division by zero. To this end, the correct limiting value is applied
    # using np.where after "safe" division is completed
    limit_t1 = (alpha / np.sqrt(2.0 * T)) * (
        (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * alpha)) +
        (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * alpha))
                )

    # Identify singular points for both special cases
    # 1e-10 is chosen as it is approx. equal to 0
    singular_t0 = np.abs(t) < 1e-10
    singular_t1 = np.abs(denominator) < 1e-10

    # Using np.where, a value of 1.0 is substituted at the "singular points" to 
    # prevent inf/nan during division. The final np.where below will replace these with
    # the proper limit values 
    safe_denom = np.where(singular_t0 | singular_t1, 1.0, denominator)

    # Also guard the sin_term division by 4*alpha*t/T at t = 0
    safe_a4t = np.where(singular_t0, 1.0, a4t)

    # Compute full expression "safely" using the "safe" terms that do not contain
    # division by 0
    h = (1.0 / np.sqrt(T)) * (4.0 * alpha / np.pi) * (cos_term + sin_term / safe_a4t) / safe_denom

    # As previously mentioned, the correct limiting values are now substituted
    h = np.where(singular_t0, limit_t0, h)
    h = np.where(singular_t1, limit_t1, h)

    return h

def rrc_fd(f, T, alpha):
    
    # Root raised cosine filter — frequency domain.
    # Defined as the square root of the RC frequency response.
    # Three regions identical to RC but are now square rooted, making this a 
    # simply extension of the previously defined RC_fd function
    if alpha == 0.0:
        return np.where(np.abs(f) <= 0.5 / T, np.sqrt(T), 0.0)

    f_abs = np.abs(f)
    # The passband edge is located at:
    f1 = (1.0 - alpha) / (2.0 * T)
    # The stopband edge is located at:
    f2 = (1.0 + alpha) / (2.0 * T)     

    # The magnitude of the transition region between f1 and f2 is:
    transition = np.sqrt(
        (T / 2.0) * (1.0 + np.cos((np.pi * T / alpha) * (f_abs - f1)))
    )

    H = np.where(f_abs <= f1, np.sqrt(T), np.where(f_abs <= f2, transition, 0.0))
    return H

# Defining an arbitrary symbol period of 1 second
T = 1.0

# Defining a set of rolloff values from 0 to 1
alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

# Different colours required to differentiate the varying rolloff values
colours = ['black', 'blue', 'red', 'green', 'orange']

f = np.linspace(-1.2, 1.2, 2000)

# Creating a time vector with +/- 8 symbol periods
# It is noted that for the alpha = 0 case, because the sinc function is infinite
# in the time-domain, when two sincs are truncated and convolved with each other
# it is expected that the magnitude of the convolved result will differ from the 
# natural sinc function towards the edges as the tails cut off by truncation
# accumulate energy near the boundaries of the plot. These truncation artefacts
# can be mitigated by extended the t vector below to larger multiples of the symbol
# period, e.g., 20 or 30 multiples of T
t = np.linspace(-8*T, 8*T, 2000)

# Plotting the time domain representation of the root RC pulse to visualise how it
# differs from the RC pulse, and that this form does not contain zero values 
# at integer multiples of the symbol period, as was the case for the RC pulse

fig1, ax1 = plt.subplots(figsize=(10, 5))

for alpha, colour in zip(alphas, colours):
    td_rrc = rrc_td(t, T, alpha)
    ax1.plot(t / T, td_rrc, label = '$\\alpha$ = ' + str(alpha), color = colour, linewidth = 1.8)

# Marking integer multiples of T and noting these are NOT zero crossings for RRC
for n in range(-8, 9):
    if n != 0:
        ax1.axvline(x = n, color = 'grey', linewidth = 0.5, linestyle = ':', alpha = 0.4)

# Annotating that RRC does not satisfy the zero-ISI property on its own
ax1.annotate('Sampling instants $t = nT$\nRRC does NOT have zeros here\n'
             '(zero-ISI only recovered after\nTx * Rx convolution)',
             xy = (1, 0),
             xytext = (2.5, 0.4),
             arrowprops = dict(arrowstyle = '->', color = 'black'),
             fontsize = 9, color = 'black')

ax1.axhline(0, color = 'black', linewidth = 0.7)
ax1.axvline(0, color = 'black', linewidth = 0.7)

ax1.set_xlabel('Normalised time (t/T)', fontsize = 11)
ax1.set_ylabel('Amplitude', fontsize = 11)
ax1.set_title('Root Raised Cosine Time Domain Representation', fontsize = 12)
ax1.legend(fontsize = 10)
ax1.set_xlim(-8, 8)
ax1.grid(True, linewidth = 0.4, alpha = 0.5)

# Plotting the frequency domain representation of the root RC pulse to visualise
# how it differs from the typical RC pulse frequency domain plot
fig2, ax2 = plt.subplots(figsize = (10, 5))

for alpha, colour in zip(alphas, colours):
    H_rrc = rrc_fd(f, T, alpha)
    ax2.plot(f, H_rrc / np.sqrt(T), label = '$\\alpha$ = ' + str(alpha), color = colour, linewidth = 1.8)

ax2.axhline(0, color = 'black', linewidth = 0.7)
ax2.axvline(0, color = 'black', linewidth = 0.7)

ax2.set_xlabel('Normalised Frequency (fT)', fontsize = 11)
ax2.set_ylabel('Normalised Magnitude', fontsize = 11)
ax2.set_title('Root Raised Cosine Frequency Domain Representation', fontsize = 12)
ax2.legend(fontsize = 10)
ax2.set_xlim(-1.2, 1.2)
ax2.set_ylim(-0.05, 1.15)
ax2.grid(True, linewidth = 0.4, alpha = 0.5)

# Plotting the square of the root RC pulse in the frequency domain to show
# that the original RC pulse is recovered in the frequency domain using
# only multiplication
fig2, ax2 = plt.subplots(figsize = (10, 5))

for alpha, colour in zip(alphas, colours):
    H_rrc = rrc_fd(f, T, alpha)
    ax2.plot(f, (H_rrc ** 2) / np.sqrt(T), label = '$\\alpha$ = ' + str(alpha), color = colour, linewidth = 1.8)

ax2.axhline(0, color = 'black', linewidth = 0.7)
ax2.axvline(0, color = 'black', linewidth = 0.7)

ax2.set_xlabel('Normalised Frequency (fT)', fontsize = 11)
ax2.set_ylabel('Normalised Magnitude', fontsize = 11)
ax2.set_title('Root Raised Cosine Frequency Domain After Matched Filtering', fontsize = 12)
ax2.legend(fontsize = 10)
ax2.set_xlim(-1.2, 1.2)
ax2.set_ylim(-0.05, 1.15)
ax2.grid(True, linewidth = 0.4, alpha = 0.5)

# Plotting the convolution of the time domain representation of the root RC
# pulse to visually demonstrate that the original RC pulse is recovered

# Only plotting three values of alpha so that subplots are not too compressed
alphas_conv = [0.0, 0.5, 1.0]
colours_conv = ['blue', 'orange', 'red']

fig7, axes = plt.subplots(len(alphas_conv), 1, figsize=(10, 14))

for i, (alpha, colour) in enumerate(zip(alphas_conv, colours_conv)):
    ax = axes[i]

    # Computing RC and RRC pulses on the same time vector
    rc_td  = raised_cosine_td(t, T, alpha)
    td_rrc = rrc_td(t, T, alpha)

    # Convolving RRC with itself (Tx * Rx) and normalise by dt to
    # account for the discrete approximation of continuous convolution
    dt_val  = t[1] - t[0]
    rrc_conv = np.convolve(td_rrc, td_rrc, mode='same') * dt_val

    # Normalising the convolution result to unit peak for direct comparison
    rrc_conv_norm = rrc_conv / np.max(np.abs(rrc_conv))
    rc_norm = rc_td / np.max(np.abs(rc_td))

    ax.plot(t / T, rc_norm, color = colour,  linewidth = 1.8, label = 'RC ($\\alpha$ = ' + str(alpha))
    ax.plot(t / T, rrc_conv_norm, color = 'black', linewidth = 1.2, linestyle = '--', label = 'RRC * RRC')

    ax.axhline(0, color = 'black', linewidth = 0.5)
    ax.set_xlim(-8, 8)
    ax.set_ylim(-0.4, 1.15)
    ax.legend(fontsize = 9, loc = 'upper right')
    ax.grid(True, linewidth = 0.4, alpha = 0.5)
    ax.set_ylabel('Amplitude', fontsize = 9)

axes[-1].set_xlabel('Normalised time (t/T)', fontsize=11)
fig7.suptitle('Tx/Rx Split Matched Filter Verification (RRC $*$ RRC = RC)', fontsize=12)

plt.tight_layout()
plt.show()