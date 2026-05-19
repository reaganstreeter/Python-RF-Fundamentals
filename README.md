# Python-RF-Fundamentals

This repository contains Python scripts demonstrating core mathematical and signal processing concepts fundamental to RF, communications, and DSP engineering. Developed as part of a structured self-directed learning program targeting junior RF, comms, and systems engineering roles in defence, space, and aerospace.

## Background and motivation

The sinc function, discrete Fourier transform, pulse shaping filters, and spectral windowing are fundamental tools used in RF and communications engineering. The sinc function is the impulse response of an ideal low-pass filter and is used as the basis for the Nyquist pulse shaping criterion to minimise intersymbol interference. The raised cosine and root raised cosine filters are derived from the sinc function to address its signal timing limitations, governing how symbols are transmitted in nearly every modern digital communications standard.

Spectral leakage arises from the DFT implicitly assuming sampled signals periodically repeat outside the sampling window. If the signal is not periodic within the observation interval, discontinuities appear at the window boundaries, causing energy to spread across multiple frequency bins. The resulting sidelobes can mask weaker nearby signals and introduce amplitude measurement errors, which are consequential in spectrum monitoring, receiver sensitivity analysis, and other high-dynamic-range signal analysis tasks. Windowing functions such as Hanning, Hamming, and Blackman reduce sidelobe levels by smoothing the signal near the window boundaries, at the cost of a wider mainlobe and therefore reduced frequency resolution. This tradeoff between frequency resolution and sidelobe suppression is a routine design consideration in spectrum analysers, SDRs, and digital filter banks used throughout RF systems.

These scripts work through each of these concepts to produce plots allowing visualisation of the underlying theory.

## Repository structure

python-rf-fundamentals/  
scripts/
- sinc_fundamentals.py — Sinc function properties and Fourier pair
- raised_cosine_pulse_shaping.py — Raised cosine family in both time and frequency domains 
- root_RC_pulse_shaping.py      — Root raised cosine + Rx/Tx split & matched filtering
- windowing.py                — Spectral leakage and window functions
- fft_synthetic.py            — FFT of a synthetic multi-tone signal
- csv_readwrite.py            — Engineering data I/O with pandas  
outputs/
- committed plot outputs for each script

## Script summaries

### sinc_fundamentals.py

The sinc function is used in filter theory, sampling theory, and pulse shaping. This script visualises the two common definitions of the sinc function, i.e., normalised and unnormalised, and demonstrates the fundamental Fourier relationship between the sinc and the rectangular function. The normalised sinc, `sin(πx)/(πx)`, has zeros at every non-zero integer while the unnormalised form, `sin(x)/x`, has zeros at non-zero multiples of π. This distinction matters because the two conventions appear inconsistently across textbooks and standards, and confusing them produces errors in applications such as filter design and link budget calculations.

The second plot computes the FFT of a normalised sinc signal with a defined cutoff frequency and confirms that the magnitude spectrum approximates a rectangular (brick-wall) shape. The artefacts visible at the cutoff edge, Gibbs overshoot and passband droop, result from the finite number of samples used to approximate the signal, and are thus expected consequences of truncating an infinite-length function. Together, this illustrates why an ideal brick-wall low-pass filter can never be physically realised.

**Key concepts:** normalised vs. unnormalised sinc conventions, sinc/rect Fourier pair, ideal low-pass filter impulse response, finite-length approximation artefacts.

**Outputs:**

<p>
  <img src="outputs/sinc_norm_vs_unnorm.png" width="48%"/>
  <img src="outputs/sinc_fft.png" width="48%"/>
</p>

### `raised_cosine_pulse_shaping.py`

In a digital communications link, transmitting perfect rectangular pulses require infinite channel bandwidth due to their sharp discontinuities and produce inter-symbol interference (ISI) at the receiver, increasing bit error rate (BER). The solution is Nyquist pulse shaping, where the sinc function is the theoretical ideal pulse shape because it has zero crossings at integer multiples of the symbol period T which guarantee ISI-free sampling, and it has a well-defined bandwidth. However, due to signal-timing limitations, it is difficult to implement. The raised cosine (RC) filter family is the practical realisation of Nyquist pulse shaping that addresses its shortfalls. It features a rolloff parameter α ∈ [0, 1] that tapers the frequency-domain transition from the ideal rectangular shape, trading bandwidth efficiency for tolerance to timing errors. At α = 0 the RC reduces to the ideal sinc; at α = 1 it occupies twice the minimum bandwidth but is maximally forgiving of timing errors.

This script plots the raised cosine pulse in both the time and frequency domains, with values of α varying from 0 to 1, to visualise the aforementioned properties of the pulse. 

**Key concepts:** Nyquist ISI criterion, raised cosine filter family, rolloff parameter α, bandwidth vs. timing margin tradeoff.

**Outputs:**

<p>
  <img src="outputs/raised_cosine_TD.png" width="48%"/>
  <img src="outputs/raised_cosine_FD.png" width="48%"/>
</p>

### `root_RC_pulse_shaping.py`

In a real system, the full RC filter is not placed at either the transmitter or receiver alone. Instead, it is split equally between the two ends as a root raised cosine (RRC) filter. Each RRC filter has a frequency response equal to the square root of the RC. When the transmit and receive RRC filters are cascaded, their combined response recovers the full RC, satisfying the Nyquist ISI criterion at the decision point while simultaneously implementing a matched filter at the receiver, which maximises SNR in a channel containing added white Gaussian noise (AWGN). This Tx/Rx split is specified in virtually every modern digital communications standard, such as DVB-S2, LTE, APCO P25, and military SATCOM waveforms, which all define an RRC rolloff parameter as part of the specification.

This script plots the time domain and frequency domain representations of the root raised cosine function, followed by demonstrating that multiplication of the RRC in the frequency domain recovers the previously seen RC frequency domain waveform, while convolution of the RRC in the time domain yields the RC time domain waveform, collectively demonstrating the Fourier property of convolution in the time domain being equivalent to multiplication in the frequency domain, and vice versa.

**Key concepts:** root raised cosine family, matched filtering, Tx/Rx filter splitting.

**Outputs:**

<p>
  <img src="outputs/rrc_td_plot.png" width="48%"/>
  <img src="outputs/rrc_fd_plot.png" width="48%"/>
  <img src="outputs/rrc_rxtx_multiply_fd.png" width="48%"/>
  <img src="outputs/rrc_rxtx_convolved_td.png" width="48%"/>
</p>
