"""Directed spectrum class and estimator function.

Class
-----
DirectedSpectrum : Represents directed spectrum and relevant labels.

Public Function
---------------
ds : Return a DirectedSpectrum object for multi-channel timeseries data.
"""
from itertools import combinations
import numpy as np
from scipy.signal import csd
from scipy.fft import fft, ifft
from numpy.linalg import cholesky, solve

# for debugging purposes !!!
import matplotlib.pyplot as plt


class DirectedSpectrum(object):
    """Represents directed spectrum and relevant labels.

    Attributes
    ----------
    ds_array : ndarray, shape (n_frequencies, n_groups, n_groups)
        Directed spectrum values between each pair of channel
        groups for each frequency.
    f : ndarray, shape (n_frequencies)
        Frequencies associated with the last dimension of ds_array.
    cgroups : ndarray of strings, shape (n_groups)
        Names the channel groups associated with ds_array.
    """

    def __init__(self, ds_array, f, cgroups):
        self.ds_array = ds_array
        self.f = f
        self.cgroups = cgroups

def ds(X, fs, groups, pairwise=False, fres=None, window='hann', nperseg=None,
       noverlap=None):
    """Returns a DirectedSpectrum object calculated from data X.

    Calculate the directed spectrum for each directed pair of channel
    groups in X. Applies Wilson's factorization of the cross power
    spectral density matrix associated with X as intermediate steps in
    the calculation.

    Parameters
    ----------
    X : ndarray, shape (n_channels, n_timepoints)
        Timeseries data from multiple channels. It is assumed that the
        data for each channel are approximately stationary.
    fs : float
        Sampling rate associated with X.
    groups : list of strings, shape (n_channels)
        Names the group associated with each channel. The directed
        spectrum is calculated between each pair of groups. To calculate
        the directed spectrum between each pair of channels, this should
        be a list of channel names.
    pairwise : bool, optional
        If 'True', calculate the pairwise directed spectrum
        (i.e. calculate seperately for each pair). Otherwise, the
        non-pairwise directed spectrum will be calculated.
    fres : float, optional
        Frequency resolution of the calculated spectrum. For example, if
        set to 1, then the directed spectrum will be calculated for
        integer frequency values. If set to 'None' (the default), then
        the frequency resolution will be fs/nperseg.
    [Documentation for the following variables was copied from the
        scipy.signal.spectral module. These variables are used for
        calculating the cross power spectral density matrix.]
    window : str or tuple or array_like, optional
        Desired window to use. If `window` is a string or tuple, it is
        passed to `get_window` to generate the window values, which are
        DFT-even by default. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg. Defaults
        to a Hann window.
    nperseg : int, optional
        Length of each segment. Defaults to None, but if window is str or
        tuple, is set to 256, and if window is array_like, is set to the
        length of the window.
    noverlap: int, optional
        Number of points to overlap between segments. If `None`,
        ``noverlap = nperseg // 2``. Defaults to `None`.

    Returns
    -------
    dir_spec : DirectedSpectrum object
    """
    cpsd, f = _cpsd_mat(X, fs, fres, window, nperseg, noverlap)
    # move frequency to first dimension
    cpsd = np.moveaxis(cpsd, 2, 0)

    gidx, grouplist = _group_indicies(groups)
    group_pairs = combinations(range(len(gidx)), 2)
    ds_array = np.zeros((len(f), len(gidx), len(gidx)))

    if not pairwise:
        H, Sigma = _wilson_factorize(cpsd, fs)
    for gp in group_pairs:
        idx0 = np.array(gidx[gp[0]])
        idx1 = np.array(gidx[gp[1]])
        pair_idx = np.nonzero(idx0 | idx1)[0]
        sub_idx1 = idx1[pair_idx]
        if pairwise:
            sub_cpsd = cpsd.take(pair_idx, axis=1).take(pair_idx, axis=2)
            H, Sigma = _wilson_factorize(sub_cpsd, fs)
            ds01, ds10 = _var_to_ds(H, Sigma, sub_idx1)
        else:
            sub_H = H.take(pair_idx, axis=1).take(pair_idx, axis=2)
            sub_Sigma = Sigma.take(pair_idx, axis=0).take(pair_idx, axis=1)
            ds01, ds10 = _var_to_ds(sub_H, sub_Sigma, sub_idx1)
        # sum across channels within group
        ds_array[:, gp[0], gp[1]] = np.linalg.det(ds01)
        ds_array[:, gp[1], gp[0]] = np.linalg.det(ds10)

    return DirectedSpectrum(ds_array, f, grouplist)

def _cpsd_mat(X, fs, fres, window, nperseg, noverlap):
    """Return cross power spectral density and associated frequencies."""
    C, T = X.shape
    if fres:
        nfft = int(fs/fres)
    elif nperseg:
        nfft = int(nperseg)
    else:
        nfft = 256
        nperseg = nfft
    cpsd = np.zeros((C, C, nfft), dtype=np.cdouble)
    for c1 in range(C):
        for c2 in range(c1, C):
            f, this_csd = csd(X[c1], X[c2], fs, window, nperseg, noverlap,
                              nfft, return_onesided=False, scaling='density')
            cpsd[c1, c2] = this_csd
            cpsd[c2, c1] = this_csd.T.conj()
    return (cpsd, f)

def _group_indicies(groups):
    """Return list of unique groups and associated indices."""
    grouplist = np.unique(groups)
    gidx = [[g1==g2 for g1 in groups] for g2 in grouplist]
    return (gidx, grouplist)

def _wilson_factorize(cpsd, fs, max_iter=1000, tol=1e-9):
    """Factorize CPSD into transfer matrix (H) and covariance (Sigma)

    Implements the algorithm outlined in the following reference:
    G. Tunnicliffe. Wilson, “The Factorization of Matricial Spectral
    Densities,” SIAM J. Appl. Math., vol. 23, no. 4, pp. 420426, Dec.
    1972, doi: 10.1137/0123044.
    """
    psi = _init_psi(cpsd)

    # add small number to cpsd to prevent it from being negative
    # semidefinite due to rounding errors
    this_eps = np.spacing(np.abs(cpsd)).max()
    U = cholesky(cpsd + np.identity(cpsd.shape[-1])*2*this_eps)
    from time import time
    start = time()
    for k in range(max_iter):
        # g = psi \ cpsd / psi* + I
        psi_inv_cpsd = solve(psi, U)
        g = psi_inv_cpsd @ psi_inv_cpsd.conj().transpose(0, 2, 1)
        g = g + np.identity(cpsd.shape[-1])

        gplus = _plus_operator(g)
        S = -np.tril(gplus[0], -1)
        S = S - S.conj().transpose()
        gplus = gplus + S
        psi_prev = psi
        psi = psi @ gplus

        converged = _check_convergence(psi, psi_prev, tol)
        if converged:
            print(f'converged! - {k:d} iters')
            break
    print(f'{time()-start:.3f}s elapsed')

    A = ifft(psi, axis=0).real
    Sigma = A[0] @ A[0].T # fs scaling?!?
    H = (solve(A[0].T, psi.transpose(0, 2, 1))).transpose(0, 2, 1) # right-side solve
    return (H, Sigma)


def _init_psi(cpsd):
    """Return initial psi value for wilson factorization."""
    # try other init?
    gamma = ifft(cpsd, axis=0) # set workers for parallelization?
    gamma0 = np.asmatrix(gamma[0])
    # remove assymetry due to rounding error
    gamma0 = np.real((gamma0+gamma0.H)/2)
    h = np.asarray(cholesky(gamma0).H)
    psi = np.tile(h[np.newaxis], (gamma.shape[0], 1, 1))
    return psi

def _plus_operator(g):
    """Remove all negative lag components from time-domain representation"""
    # remove imaginary components due to rounding error
    gamma = ifft(g, axis=0).real

    # take half of 0 lag
    gamma[0] *= 0.5

    # take half of nyquist component if fft had even # of points
    F = gamma.shape[0]
    N = int(np.floor(F/2))
    if F % 2 == 0:
        gamma[N] *= 0.5

    # zero out negative frequencies
    gamma[N+1:] = 0

    gp = fft(gamma, axis=0)
    return gp

def _check_convergence(x, x0, tol):
    """Determine whether maximum change is lower than tolerance."""
    x_diff = np.abs(x - x0)
    converged = x_diff.max() < tol
    return converged

def _var_to_ds(H, Sigma, idx1):
    """Calculate directed spectrum."""
    idx0 = np.nonzero(~idx1)[0]
    idx1 = np.nonzero(idx1)[0]
    H01 = H.take(idx0, axis=1).take(idx1, axis=2)
    H10 = H.take(idx1, axis=1).take(idx0, axis=2)
    sig00 = Sigma.take(idx0, axis=0).take(idx0, axis=1)
    sig11 = Sigma.take(idx1, axis=0).take(idx1, axis=1)
    sig01 = Sigma.take(idx0, axis=0).take(idx1, axis=1)
    sig10 = Sigma.take(idx1, axis=0).take(idx0, axis=1)
    # conditional covariances
    sig1_0 = sig11 - sig10 @ solve(sig00, sig10.conj().T)
    sig0_1 = sig00 - sig01 @ solve(sig11, sig01.conj().T)

    ds10 = np.real(H01 @ sig1_0 @ H01.conj().transpose(0, 2, 1))
    ds01 = np.real(H10 @ sig0_1 @ H10.conj().transpose(0, 2, 1))
    return (ds01, ds10)
