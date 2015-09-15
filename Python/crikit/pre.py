# -*- coding: utf-8 -*-
"""
Coherent Raman Imaging (CRI) ToolKit (CRIKit)\n
Spectral Pre-Processing Tools (CRIKit.pre)
=======================================================

    kkrelation : Retrieve real and imaginary components from a
    spectrum that is the modulus of a function\n

    hilbertfft : Fourier-domain Hilbert transform\n

Citation Reference
------------------
    C. H. Camp Jr, Y. J. Lee, and M. T. Cicerone, "Quantitative,
    Comparable Coherent Anti-Stokes Raman Scattering (CARS)
    Spectroscopy: Correcting Errors in Phase Retrieval" (2015).
    arXiv:1507.06543.


===================================
Original Python branch: Feb 16 2015

@author: ("Charles H Camp Jr")
@email: ("charles.camp@nist.gov")
@date: ("Jun 28 2015")
@version: ("0.1.1")
"""

__all__ = ['kkrelation','hilbertfft']

_DEFAULT_THREADS = 1

import numpy as _np

## Conditional modules
# Check for and load pyFFTW if available (kkrelation, hilbertfft)
try:
    import pyfftw as _pyfftw
    _pyfftw_available = True
except ImportError:
    print("No pyFFTW found. Using Scipy instead. \n\
    You may want to install pyFFTW and FFTW for [potentially]\n\
    significant performance enhancement")
    from scipy import fftpack as _fftpack
    _pyfftw_available = False

#from crikit.utils.als_methods import als_baseline

# Check for and load multiprocessing to determine number
# of CPUs
try:
    #from multiprocessing import cpu_count
    import multiprocessing as _multiprocessing
    _thread_num = _multiprocessing.cpu_count()
except ImportError:
    print("No multiprocessing module found. \n\
    Default thread number set to 1. This can be\n\
    changed within the .py file")
    _thread_num = _DEFAULT_THREADS

def kkrelation(ref_spectral_data, cri_spectral_data, phase_offset=0.0,
               norm_by_ref_flag=True):
    """
    Retrieve the real and imaginary components of a CRI spectra(um) via
    the Kramers-Kronig (KK) relation.

    Parameters
    ----------
    ref_spectral_data : ndarray
        NRB reference spectra(um) array that can be one-, two-,
        or three-dimensional
    cri_spectral_data : ndarray
        CRI spectra(um) array that can be one-,two-,or three-dimensional
    (phase_offset) : int, float, or ndarray, optional
        Global phase offset applied to the KK, which effecively controls
        the real-to-imaginary components relationship
    (norm_by_ref_flag) : bool
        Should the output be normalized by the square-root of the
        reference NRB spectrum(a)

    Returns
    -------
    out : complex ndarray
        The real and imaginary components of KK.

    Note
    ----
    (1) The imaginary components provides the sponatenous Raman-like
    spectra(um).

    (2) This module assumes the spectra are oriented as such that the
    frequency (wavenumber) increases with increasing index.  If this is
    not the case for your spectra(um), apply a phase_offset of _np.pi

    (3) This is the first attempt at converting MATLAB (Mathworks, Inc)
    scripts into Python code; thus, there will be bugs, the efficiency
    will be low(-ish), and I appreciate any useful suggestions or
    bug-finds.

    References
    ----------
    Y. Liu, Y. J. Lee, and M. T. Cicerone, "Broadband CARS spectral
    phase retrieval using a time-domain Kramers-Kronig transform,"
    Opt. Lett. 34, 1363-1365 (2009).

    C. H. Camp Jr, Y. J. Lee, and M. T. Cicerone, "Quantitative,
    Comparable Coherent Anti-Stokes Raman Scattering (CARS)
    Spectroscopy: Correcting Errors in Phase Retrieval"

    ===================================
    Original Python branch: Feb 16 2015

    @author: ("Charles H Camp Jr")\n
    @email: ("charles.camp@nist.gov")\n
    @date: ("Jun 28 2015")\n
    @version: ("0.1.1")\n
    """
    #import numpy as np

    # Ensure the shape of phase_offset is compatible
    # with cri_spectral_data
    if _np.size(phase_offset) == 1 or \
    phase_offset.shape == cri_spectral_data.shape:
        pass
    else:
        phase_offset = _matchsize(cri_spectral_data, phase_offset)

    # Ensure the shape of ref_spectral_data is compatible
    # with cri_spectral_data
    if ref_spectral_data.shape == cri_spectral_data.shape:
        pass
    else:
        ref_spectral_data = _matchsize(cri_spectral_data, ref_spectral_data)

    # Return the complex KK relation using the Hilbert transform.
    if norm_by_ref_flag is True: # Norm the Amp by ref)spectral_data
        return _np.sqrt(cri_spectral_data/ref_spectral_data)*\
        _np.exp(1j*phase_offset+1j*\
        _np.imag(hilbertfft(0.5*_np.log(cri_spectral_data/\
        ref_spectral_data))))
    else: # Do NOT norm the Amp by ref)spectral_data
        return _np.sqrt(cri_spectral_data)*_np.exp(1j*phase_offset+1j*\
        _np.imag(hilbertfft(0.5*_np.log(cri_spectral_data/\
        ref_spectral_data))))

def hilbertfft(spectral_data, pad_factor=1):
    """
    Compute the one-dimensional Hilbert Transform.

    This function computes the one-dimentional Hilbert transform
    using the Fourier-domain implementation, and outputs the analytic
    function (i.e., real input and imaginary Hilbert transformed signal)

    Parameters
    ----------
    spectral_data : ndarray
        Input array that can be one-,two-,or three-dimensional
    (pad_factor) : int, optional
        The multiple number of spectral_data-length pads that will be
        applied before and after the original spectra

    Returns
    -------
    out : complex ndarray
        The analytic expression with the original spectral_data in the
        real component, and the Hilbert transformed data within the
        imaginary component

    Note
    ----
    This is the first attempt at converting MATLAB (Mathworks, Inc)
    scripts into Python code; thus, there will be bugs, the efficiency
    will be low(-ish), and I appreciate any useful suggestions or
    bug-finds.

    References
    ----------
    C. H. Camp Jr, Y. J. Lee, and M. T. Cicerone, "Quantitative,
    Comparable Coherent Anti-Stokes Raman Scattering (CARS)
    Spectroscopy: Correcting Errors in Phase Retrieval"

    A. D. Poularikas, "The Hilbert Transform," in The Handbook of
    Formulas and Tables for Signal Processing (ed., A. D. Poularikas),
    Boca Raton, CRC Press LLC (1999).

    ===================================
    Original Python branch: Feb 16 2015

    @author: ("Charles H Camp Jr")\n
    @email: ("charles.camp@nist.gov")\n
    @date: ("Jun 28 2015")\n
    @version: ("0.1.1")\n
    """



    # Find dimensionality. Computations always performed in 3D.
    # Return is of same size of input.
    dims = spectral_data.ndim
    if dims == 1:
        spectral_data = spectral_data[:, _np.newaxis, _np.newaxis]
    elif dims == 2:
        spectral_data = spectral_data[:, :, _np.newaxis]
        y_len_orig = spectral_data[0, :, 0].size
    spectrum_len = spectral_data[:, 0, 0].size
    y_len = spectral_data[0, :, 0].size
    x_len = spectral_data[0, 0, :].size

    # Will hold the spectrally-padded input data
    spectral_data_pad = _np.zeros([spectrum_len+2*spectrum_len*\
    pad_factor, y_len, x_len], dtype=complex)

    # time_vec keeps track of positive and negative time in the
    # Fourier-domain
    time_vec = _np.fft.fftfreq(spectral_data_pad.shape[0])
    time_vec = time_vec[:, _np.newaxis, _np.newaxis]

    # Hilbert transformed data (padded)
    hilbert_spectral_data_pad = _np.zeros([spectrum_len+2*\
    spectrum_len*pad_factor, y_len, x_len], dtype=complex)
    if pad_factor > 0:
        spectral_data_pad[0:spectrum_len*pad_factor, :, :] =\
        spectral_data[0, :, :]*_np.ones([spectrum_len*pad_factor\
        , 1, 1])
        spectral_data_pad[spectrum_len*pad_factor+spectrum_len:\
        , :, :] = spectral_data[-1, :, :]*\
        _np.ones([spectrum_len*pad_factor, 1, 1])
        spectral_data_pad[spectrum_len*pad_factor:spectrum_len*\
        pad_factor+spectrum_len, :, :] = spectral_data
    else:
        spectral_data_pad = spectral_data

    # Perform Hilbert Transform with FFTW if available
    # Hilbert{f(w)} = FFT{i*sgn(t) * iFFT{f(w)}}

    if _pyfftw_available == True:
        _pyfftw.interfaces.cache.enable()
        hilbert_spectral_data_pad = 1j*\
        _np.real(_pyfftw.interfaces.scipy_fftpack.fft(1j*\
        _np.sign(time_vec)*_pyfftw.interfaces.scipy_fftpack.ifft(\
        spectral_data_pad, axis=0, overwrite_x=True, \
        threads=_thread_num, auto_align_input=True, \
        planner_effort='FFTW_MEASURE'), axis=0, \
        overwrite_x=True, threads=_thread_num, \
        auto_align_input=True, planner_effort='FFTW_MEASURE'))
    else: # Perform Hilbert Transform with Scipy FFTPACK
        hilbert_spectral_data_pad = 1j*\
        _np.real(_fftpack.fft(1j*_np.sign(time_vec)*\
        _fftpack.ifft(spectral_data_pad, axis=0, \
        overwrite_x=True), axis=0, overwrite_x=True))

    # Return data with the same dimensionality as the original
    # input
    if dims == 1:
        return _np.squeeze(spectral_data +\
        hilbert_spectral_data_pad[spectrum_len*\
        pad_factor:spectrum_len*pad_factor+spectrum_len, :, :])
    elif dims == 2:
        return _np.reshape(spectral_data +\
        hilbert_spectral_data_pad[spectrum_len*\
        pad_factor:spectrum_len*pad_factor+spectrum_len, :, :],\
        [spectrum_len, y_len_orig])
    else:
        return spectral_data +\
        hilbert_spectral_data_pad[spectrum_len*\
        pad_factor:spectrum_len*pad_factor+spectrum_len, :, :]

def _matchsize(reference, signal):
    """
    _matchsize(reference,signal)     PRIVATE MODULE

    Matches the size of a signal to a reference as long as the first
    dimension lengths agree

    Parameters
    ----------
    reference : ndarray (real or complex)
        Input array that can be one-,two-,or three-dimensional
    signal : ndarray (real or complex)
        Input array that can be one-,two-,or three-dimensional

    Returns
    -------
    out : ndarray (real or complex)
        signal resized to be equivalent in shape to reference

    @author: ("Charles H Camp Jr")\n
    @email: ("charles.camp@nist.gov")\n
    @date: ("Jun 28 2015")\n
    @version: ("0.1.1")\n
    """

    dims_ref = reference.ndim
    dims_sig = signal.ndim
    assert(dims_ref <= 3), 'Reference is %d-dimensional. \
    This module can only support up to three-dimensions' \
    % dims_ref
    matched_signal = _np.zeros(reference.shape)
    #print('Size of matched signal: %s' % str(_np.shape(matched_signal)))
    assert(dims_ref > 1), 'Cannot match a reference with shape\
    %s to a signal with shape %s' % (str(reference.shape),\
    str(signal.shape))
    if dims_ref == 2:
        #print('DIM = 2')
        len_sig_0 = signal.shape[0]
        len_ref_0 = reference.shape[0]
        len_ref_1 = reference.shape[1]
        assert(len_sig_0 == len_ref_0), 'Cannot match a \
        reference with first dimension length %d and a signal \
        with length %d' % (reference.shape[0], signal.shape[0])
        assert(dims_sig <= 3), 'The signal has more than three \
        dimensions'
        if dims_sig == 1:
            pass
        elif dims_sig == 2:
            signal = signal[:, 0]
        elif dims_sig == 3:
            signal = signal[:, 0, 0]

        matched_signal = signal[:, _np.newaxis]*_np.ones([1, len_ref_1])
    elif dims_ref == 3:
        #print('DIM = 3')
        len_sig_0 = signal.shape[0]
        len_ref_0 = reference.shape[0]
        len_ref_1 = reference.shape[1]
        len_ref_2 = reference.shape[2]
        assert(len_sig_0 == len_ref_0), 'Cannot match a \
        reference with first dimension length %d and a signal \
        with length %d' % (reference.shape[0], signal.shape[0])
        assert(dims_sig <= 3), 'The signal has more than three \
        dimensions'
        if dims_sig == 1:
            pass
        elif dims_sig == 2:
            signal = signal[:, 0]
        elif dims_sig == 3:
            signal = signal[:, 0, 0]
        for count in range(len_ref_2):
            matched_signal[:, :, count] = signal[:, _np.newaxis]*\
            _np.ones([1, len_ref_1])
    #print('Size of matched signal at end: %s' % str(_np.shape(matched_signal)))
    return matched_signal
