"""Directed spectrum class and estimator function.

Class
-----
DirectedSpectrum : Represents directed spectrum and relevant labels.

Public Function
---------------
ds : Return a DirectedSpectrum object for multi-channel timeseries data.
"""
from itertools import combinations
from warnings import warn
import numpy as np
from scipy.signal import csd
from scipy.fft import fft, ifft
from numpy.linalg import cholesky, solve, eigh

class DirectedSpectrum(object):
    """Represents directed spectrum and relevant labels.

    Attributes
    ----------
    ds_array : ndarray, shape (n_windows, n_frequencies, n_groups, n_groups)
        Directed spectrum values between each pair of channel
        groups for each frequency.
    f : ndarray, shape (n_frequencies)
        Frequencies associated with the last dimension of ds_array.
    cgroups : ndarray of strings, shape (n_groups)
        Names the channel groups associated with ds_array.
    """

    def __init__(self, ds_array, f, cgroups):
        self.ds_array = np.array(ds_array, dtype=np.float64)
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
    X : ndarray, shape (n_windows, n_channels, n_timepoints)
        Timeseries data from multiple channels. It is assumed that the
        data for each channel are approximately stationary within a window.
        If a 2-D array is input, it is assumed that n_windows is 1.
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
    if X.ndim == 2:
        X = X[np.newaxis]
    cpsd, f = _cpsd_mat(X, fs, fres, window, nperseg, noverlap)
    # move frequency to second dimension.
    cpsd = np.moveaxis(cpsd, 3, 1)

    ds_array = np.zeros_like(cpsd, dtype=np.float64)
    gidx, grouplist = _group_indicies(groups)
    group_pairs = combinations(range(len(gidx)), 2)

    if not pairwise:
        H, Sigma = _wilson_factorize(cpsd, fs)
    for gp in group_pairs:
        # get indices of both groups in current pair.
        idx0 = np.array(gidx[gp[0]])
        idx1 = np.array(gidx[gp[1]])
        pair_idx = np.nonzero(idx0 | idx1)[0]
        sub_idx1 = idx1[pair_idx] # subset of pair_idx in group 1
        if pairwise:
            sub_cpsd = cpsd.take(pair_idx, axis=-2).take(pair_idx, axis=-1)
            H, Sigma = _wilson_factorize(sub_cpsd, fs)
            ds01, ds10 = _var_to_ds(H, Sigma, sub_idx1)
        else:
            sub_H = H.take(pair_idx, axis=-2).take(pair_idx, axis=-1)
            sub_Sigma = Sigma.take(pair_idx, axis=-2).take(pair_idx, axis=-1)
            ds01, ds10 = _var_to_ds(sub_H, sub_Sigma, sub_idx1)
        # average across channels within group.
        # TODO: allow for other options besides average here
        ds_array[:,:, gp[0], gp[1]] = \
            np.diagonal(ds01, axis1=-2, axis2=-1).mean(axis=-1)
        ds_array[:,:, gp[1], gp[0]] = \
            np.diagonal(ds10, axis1=-2, axis2=-1).mean(axis=-1)
    return DirectedSpectrum(ds_array, f, grouplist)

def _cpsd_mat(X, fs, fres, window, nperseg, noverlap):
    """Return cross power spectral density and associated frequencies."""
    # if frequency resolution is set, use it to determine fft length.
    if fres:
        nfft = int(fs/fres)
    elif nperseg:
        nfft = int(nperseg)
    else:
        nfft = 256
        nperseg = nfft
    f, cpsd = csd(X[:,:,np.newaxis], X[:,np.newaxis], fs, window, nperseg,
                  noverlap, nfft, return_onesided=False, scaling='density')
    return (cpsd, f)

def _group_indicies(groups):
    """Return list of unique groups and associated indices."""
    grouplist = np.unique(groups)
    gidx = [[g1==g2 for g1 in groups] for g2 in grouplist]
    return (gidx, grouplist)

def _wilson_factorize(cpsd, fs, max_iter=1000, tol=1e-9):
    """Factorize CPSD into transfer matrix (H) and covariance (Sigma).

    Implements the algorithm outlined in the following reference:
    G. Tunnicliffe. Wilson, “The Factorization of Matricial Spectral
    Densities,” SIAM J. Appl. Math., vol. 23, no. 4, pp. 420426, Dec.
    1972, doi: 10.1137/0123044.

    This code is based on an original implementation in MATLAB provided
    by M. Dhamala (mdhamala@mail.phy-ast.gsu.edu).
    """
    psi, A0 = _init_psi(cpsd)

    w, v = eigh(cpsd)
    w[w<0] = 0 # fix negative eigvals due to precision error
    L = np.sqrt(w[...,np.newaxis,:]) * v

    H = np.zeros_like(psi)
    Sigma = np.zeros_like(A0)
    for w in range(cpsd.shape[0]):
        for k in range(max_iter):
            # These lines implement: g = psi \ cpsd / psi* + I
            psi_inv_cpsd = solve (psi[w], L[w])
            g = psi_inv_cpsd @ psi_inv_cpsd.conj().transpose(0, 2, 1)
            g = g + np.identity(cpsd.shape[-1])
            # TODO: compare these update steps to original Wilson algorithm.
            #       also check if psi[0] should be upper triangular.
            gplus, g0 = _plus_operator(g)

            # S is chosen so that g0 + S is upper triangular; S + S* = 0
            S = -np.tril(g0, -1)
            S = S - S.conj().transpose()
            gplus = gplus + S
            psi_prev = psi[w]
            psi[w] = psi[w] @ gplus
            A0_prev = A0[w]
            A0[w] = A0[w] @ (g0 + S)
            if (_check_convergence(psi[w], psi_prev, tol) and
                    _check_convergence(A0[w], A0_prev, tol)):
                break
            else:
                warn('Wilson factorization failed to converge.', stacklevel=2)

        # right-side solve
        H[w] = (solve(A0[w].T, psi[w].transpose(0, 2, 1))).transpose(0, 2, 1) 
        Sigma[w] = (A0[w] @ A0[w].T) * fs
    return (H, Sigma)


def _init_psi(cpsd):
    """Return initial psi value for wilson factorization."""
    # TODO: provide other initialization options; test which is best.
    gamma = ifft(cpsd, axis=1)
    gamma0 = gamma[:, 0]

    # remove assymetry in gamma0 due to rounding error.
    gamma0 = np.real((gamma0+gamma0.conj().transpose(0, 2, 1))/2)
    h = cholesky(gamma0).conj().transpose(0, 2, 1)
    psi = np.tile(h[:, np.newaxis], (1, gamma.shape[1], 1, 1)).astype(complex)
    return psi, h

def _plus_operator(g):
    """Remove all negative lag components from time-domain representation"""
    # remove imaginary components from ifft due to rounding error.
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
    return gp, gamma[0]

def _check_convergence(x, x0, tol):
    """Determine whether maximum change is lower than tolerance."""
    x_diff = np.abs(x - x0)
    converged = x_diff.max() < tol
    return converged

def _var_to_ds(H, Sigma, idx1):
    """Calculate directed spectrum."""
    # convert to indices from boolean
    idx0 = np.nonzero(~idx1)[0]
    idx1 = np.nonzero(idx1)[0]

    H01 = H.take(idx0, axis=-2).take(idx1, axis=-1)
    H10 = H.take(idx1, axis=-2).take(idx0, axis=-1)
    sig00 = Sigma.take(idx0, axis=-2).take(idx0, axis=-1)
    sig11 = Sigma.take(idx1, axis=-2).take(idx1, axis=-1)
    sig01 = Sigma.take(idx0, axis=-2).take(idx1, axis=-1)
    sig10 = Sigma.take(idx1, axis=-2).take(idx0, axis=-1)
    # conditional covariances
    sig1_0 = sig11 - sig10 @ solve(sig00, sig10.conj().transpose(0, 2, 1))
    sig0_1 = sig00 - sig01 @ solve(sig11, sig01.conj().transpose(0, 2, 1))

    ds10 = np.real(H01 @ sig1_0[:, np.newaxis] @ H01.conj().transpose(0, 1, 3, 2))
    ds01 = np.real(H10 @ sig0_1[:, np.newaxis] @ H10.conj().transpose(0, 1, 3, 2))
    return (ds01, ds10)
