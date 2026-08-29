"""Filter and FFT design / application helpers.

Split verbatim out of utils.py (see that module's compatibility shim).
"""

import math
import sys
from math import tau

import numba
import numpy as np
import scipy.signal as sps
from numba import njit

from .dsp import nb_round


# Essential (or at least useful) standalone routines

# https://stackoverflow.com/questions/20924085/python-conversion-between-coordinates
def polar2z(r, θ):
    return r * np.exp(1j * θ)


def emphasis_iir(t1, t2, fs):
    """Generate an IIR filter for 6dB/octave pre-emphasis (t1 > t2) or
    de-emphasis (t1 < t2), given time constants for the two corners."""

    # Convert time constants to frequencies, and pre-warp for bilinear transform
    w1 = 2 * fs * np.tan((1 / t1) / (2 * fs))
    w2 = 2 * fs * np.tan((1 / t2) / (2 * fs))

    # Zero at t1, pole at t2
    tf_b, tf_a = sps.zpk2tf([-w1], [-w2], w2 / w1)
    rv = sps.bilinear(tf_b, tf_a, fs)

    return rv


# This converts a regular B, A filter to an FFT of our selected block length
def filtfft(filt, blocklen):
    return sps.freqz(filt[0], filt[1], blocklen, whole=1)[1]


@njit(cache=True)
def inrange(a, mi, ma):
    return (a >= mi) & (a <= ma)


def sqsum(cmplx):
    return np.abs(cmplx)


@njit(cache=True, nogil=True)
def calczc_findfirst(data, target, rising):
    if rising:
        for i in range(1, len(data)):
            if data[i - 1] < target and data[i] >= target:
                return i

        return None
    else:
        for i in range(1, len(data)):
            if data[i - 1] > target and data[i] <= target:
                return i

        return None


@njit(cache=True, nogil=True)
def calczc_do(data, _start_offset, target, edge=0, count=16):
    start_offset = int(_start_offset)
    icount = int(count + 1)

    if edge == 0:  # capture rising or falling edge
        if data[start_offset] < target:
            edge = 1
        else:
            edge = -1

    loc = calczc_findfirst(
        data[start_offset : start_offset + icount], target, edge == 1
    )

    if loc is None:
        return None

    x = start_offset + loc
    a = data[x - 1] - target
    b = data[x] - target

    if b - a != 0:
        y = -a / (-a + b)
    else:
        print(
            "RuntimeWarning: Div by zero prevented at lddecode/utils.calczc_do()", a, b
        )
        y = 0

    return x - 1 + y


def calczc(data, _start_offset, target, edge=0, count=16, reverse=False):
    """ edge:  -1 falling, 0 either, 1 rising """
    if reverse:
        # Instead of actually implementing this in reverse, use numpy to flip data
        rev_zc = calczc_do(data[_start_offset::-1], 0, target, edge, count)
        if rev_zc is None:
            return None

        return _start_offset - rev_zc

    return calczc_do(data, _start_offset, target, edge, count)


@njit(cache=True, nogil=True)
def refine_pilot_zcs(demod_pilot, linelocs, n, length_px, freq, linelen, pilot_mhz):
    """ Hot inner loop of FieldPAL.refine_linelocs_pilot, lifted to numba.

    Reproduces the per-line work exactly (slice -> argmax|abs| -> calczc ->
    normalised zero-crossing phase) for lineoffset == 0, which is always the
    case when pilot refinement runs.  Returns (zcs, plen) as float64 arrays.
    """
    zcs = np.empty(n, dtype=np.float64)
    plen = np.empty(n, dtype=np.float64)
    prev = 0.0
    for l in range(n):
        adjfreq = freq
        if l > 1:
            # a player skip can collapse adjacent linelocs to the same value;
            # keep the nominal frequency for such degenerate (or reversed)
            # spacings instead of dividing by ~zero (the line is garbage
            # either way - downstream confidence handles it)
            spacing = (linelocs[l] - linelocs[l - 1]) / linelen
            if spacing > 0.1:
                adjfreq = freq / spacing
        pl = (adjfreq / pilot_mhz) / 2
        plen[l] = pl

        begin = linelocs[l]
        start = int(begin)
        stop = int(begin + length_px + 1)
        lsoffset = begin - start

        pilots = demod_pilot[start:stop]

        # np.argmax(np.abs(pilots)) -- first index of the largest magnitude
        peakloc = 0
        mx = -1.0
        for i in range(pilots.shape[0]):
            v = pilots[i]
            if v < 0:
                v = -v
            if v > mx:
                mx = v
                peakloc = i

        zc_base = calczc_do(pilots, peakloc, 0.0, 0, 16)
        if zc_base is not None:
            zcs[l] = (zc_base - lsoffset) / pl
        else:
            zcs[l] = prev
        prev = zcs[l]

    return zcs, plen


@njit(cache=True, nogil=True)
def refine_hsync_zcs(
    demod_05, linelocs1, linebad, n, is_pal, freq,
    vsync_target, neg55, pos30,
):
    """ Hot inner loop of Field.refine_linelocs_hsync, lifted to numba.

    Refines each line's hsync start by zero-crossing on the 0.5 MHz demod, with
    the same level/sanity gating as the Python original.  linebad is updated in
    place; returns the refined linelocs2.
    """
    linelocs2 = linelocs1.copy()
    for i in range(n):
        # skip VSYNC lines, since they handle the pulses differently
        if (3 <= i <= 6) or (is_pal and (1 <= i <= 2)):
            linebad[i] = True
            continue

        ll1 = linelocs1[i] - freq
        zc = calczc_do(demod_05, ll1, vsync_target, 0, freq * 2)

        if zc is not None and not linebad[i]:
            linelocs2[i] = zc

            # the hsync area, burst, and porches should stay within -50..30 IRE
            hsync_area = demod_05[int(zc - (freq * 0.75)) : int(zc + (freq * 8))]
            if np.min(hsync_area) < neg55 or np.max(hsync_area) > pos30:
                linebad[i] = True
                linelocs2[i] = linelocs1[i]
            else:
                porch_level = np.median(
                    demod_05[int(zc + (freq * 8)) : int(zc + (freq * 9))]
                )
                sync_level = np.median(
                    demod_05[int(zc + (freq * 1)) : int(zc + (freq * 2.5))]
                )

                zc2 = calczc_do(demod_05, ll1, (porch_level + sync_level) / 2, 0, 400)

                if zc2 is not None and abs(zc2 - zc) < (freq / 2):
                    linelocs2[i] = zc2
                else:
                    linebad[i] = True
        else:
            linebad[i] = True

        if linebad[i]:
            linelocs2[i] = linelocs1[i]

    return linelocs2


# copied from vhs-decode

def gen_bpf_supergauss(freq_low, freq_high, order, nyquist_hz, block_len):
    sg = supergauss(
        np.linspace(0, nyquist_hz, block_len // 2 + 1),
        freq_high - freq_low,
        order,
        (freq_high + freq_low) / 2.0,
    )[:-1]

    return np.concatenate([sg, np.flip(sg)])


def supergauss(x, freq, order=1, centerfreq=0):
    return np.exp(
        -2
        * np.power(
            (2 * (x - centerfreq) * (math.log(2.0) / 2.0) ** (1 / (2 * order))) / freq,
            2 * order,
        )
    )


# Shamelessly based on https://github.com/scipy/scipy/blob/v1.6.0/scipy/signal/signaltools.py#L2264-2267
# ... and intended for real FFT, but seems fine with complex as well ;)
def build_hilbert(fft_size):
    if (fft_size // 2) - (fft_size / 2) != 0:
        raise Exception("build_hilbert: must have even fft_size")

    output = np.zeros(fft_size)
    output[0] = output[fft_size // 2] = 1
    output[1 : fft_size // 2] = 2

    return output


if not numba.version_info.major and numba.version_info.minor < 59:
    print("DEPRECATION WARNING: Please upgrade numba to 0.59 or later.", file=sys.stderr)
    print("(follow instructions on the ld-decode wiki to set up a virtualenv)", file=sys.stderr)


@njit(cache=True, nogil=True)
def unwrap_hilbert(hilbert, freq_hz):
    """Recover the instantaneous frequency (Hz, in the range [0, freq_hz)) of an
    analytic (complex) signal.

    Conjugate-product FM discriminator: the per-sample phase increment is taken
    as the argument of hilbert[n] * conj(hilbert[n-1]).  arctan2 returns that
    increment already wrapped into (-pi, pi], so there is no need for the
    np.unwrap() pass nor the subsequent re-wrap "fixangles" clamp that the
    previous implementation depended on.

    On well-behaved data this is numerically equivalent to the old
    angle-difference + unwrap method, but it is local: a single corrupted sample
    only disturbs its own increment instead of being propagated forward by
    np.unwrap().  It is also fully numba-jittable (the old deprecation path ran
    np.unwrap outside numba) and uses a single arctan2 pass instead of an
    arctan2 pass plus an unwrap pass plus the clamp loops.
    """
    out = np.empty(len(hilbert), dtype=np.float64)
    out[0] = 0.0

    # phase increment between consecutive samples = arg(z[n] * conj(z[n-1]))
    prod = hilbert[1:] * np.conj(hilbert[:-1])
    d = np.arctan2(prod.imag, prod.real)

    # preserve the historical [0, tau) convention (positive frequencies only)
    out[1:] = np.where(d < 0.0, d + tau, d)

    return out * (freq_hz / tau)


@njit(cache=True, nogil=True)
def unwrap_hilbert_pll(hilbert, freq_hz, center_hz, loop_fn_hz, zeta):
    """PLL FM discriminator: recover instantaneous frequency (Hz) of an analytic
    signal with a second-order phase-locked loop instead of the conjugate-product
    differentiator in unwrap_hilbert().

    A digitally-controlled oscillator tracks the input phase; the loop's
    frequency-integrator state IS the demodulated FM.

    NOTE (negative result, kept for reference): PLL threshold extension only
    helps *high* modulation-index FM.  LaserDisc video is low index - IEC 60856
    9.2.3 gives only 800 kHz blanking-to-white deviation against a ~5 MHz message
    bandwidth, so beta ~= 0.1 - and the loop must be wide enough to follow that
    ~5.8 MHz message (loop_fn_hz >= ~5 MHz).  At the 40 MHz sample rate that loop
    is near-unstable and amplifies noise rather than rejecting it; measured ~3.5 dB
    *worse* wSNR than the conjugate-product unwrap_hilbert() on real captures.  The
    conjugate-product discriminator is the right tool for this modulation; this is
    retained behind --fm_pll only as an experiment / building block.

    loop_fn_hz sets the loop natural frequency (must exceed the video modulation
    rate to follow the picture); zeta is the damping (~0.707 critically damped).
    """
    n = len(hilbert)
    out = np.empty(n, np.float64)
    twopi = 2.0 * np.pi

    # second-order loop gains from the normalised natural frequency
    theta = twopi * loop_fn_hz / freq_hz
    kp = 2.0 * zeta * theta
    ki = theta * theta

    omega_c = twopi * center_hz / freq_hz
    phase = 0.0
    integ = omega_c  # frequency integrator (rad/sample), seeded at the carrier

    for i in range(n):
        zr = hilbert[i].real
        zi = hilbert[i].imag
        c = np.cos(phase)
        s = np.sin(phase)
        # phase error = arg(z * exp(-j*phase)), already wrapped into (-pi, pi]
        er = zr * c + zi * s
        ei = zi * c - zr * s
        err = np.arctan2(ei, er)

        freq = integ + kp * err  # proportional + integral = instantaneous freq
        out[i] = freq
        integ += ki * err
        phase += freq
        if phase >= twopi:
            phase -= twopi
        elif phase < 0.0:
            phase += twopi

    return out * (freq_hz / twopi)


def fft_determine_slices(center, min_bandwidth, freq_hz, bins_in):
    """ returns the # of sub-bins needed to get center+/-min_bandwidth.
        The returned lowbin is the first bin (symmetrically) needed to be saved.

        This will need to be 'flipped' using fft_slice to get the trimmed set
    """

    # compute the width of each bin
    binwidth = freq_hz / bins_in

    cbin = nb_round(center / binwidth)

    # compute the needed number of fft bins...
    bbins = nb_round(min_bandwidth / binwidth)
    # ... and round that up to the next power of two
    nbins = 2 * (2 ** math.ceil(math.log2(bbins * 2)))

    lowbin = cbin - (nbins // 4)

    cut_freq = binwidth * nbins

    return lowbin, nbins, cut_freq


def fft_do_slice(fdomain, lowbin, nbins, blocklen):
    """ Uses lowbin and nbins as returned from fft_determine_slices to
        cut the fft """
    nbins_half = nbins // 2
    return np.concatenate(
        [
            fdomain[lowbin : lowbin + nbins_half],
            fdomain[blocklen - lowbin - nbins_half : blocklen - lowbin],
        ]
    )


def overlap_save_fft(data, blocklen=32768, blockcut_begin=1024, blockcut_end=512):
    '''
    overlap/save [i]fft functions for testing.  use in a jupyter notebook or similar
    like this:

    f = overlap_save_fft(fields[0].dspicture)
    invf = overlap_save_ifft(f, round=True).astype(np.uint16)
    sum(invf != fields[0].dspicture) # should be 0
    '''
    blockstride = blocklen - blockcut_begin - blockcut_end

    numblocks = (len(data) // blockstride) + 1

    # return the length as the first element so we can reconstruct the original data
    # with the correct length in overlap_save_ifft
    fft_out = [len(data)]
    for i in range(numblocks):
        blockstart = i * blockstride
        dcut = data[blockstart : blockstart + blocklen]
        if len(dcut) < blocklen:
            pad = np.full(blocklen - len(dcut), 0, dtype=dcut.dtype)
            dcut = np.concatenate((dcut, pad))

        fft_out.append(np.fft.fft(dcut))

    return fft_out


def overlap_save_ifft(ffts, blockcut_begin=1024, blockcut_end=512, round=False):
    # blocklen is inferred from fft size
    data_out = []

    for i, f in enumerate(ffts[1:]):
        invf = np.fft.ifft(f).real
        if round:
            invf = np.round(invf)
        if not i:
            data_out.append(invf[:-blockcut_end])
        else:
            data_out.append(invf[blockcut_begin:-blockcut_end])

    return np.concatenate(data_out)[:ffts[0]]
