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
from scipy.fft import ifft
from numpy.linalg import cholesky, solve

# for debugging purposes
import matplotlib.pyplot as plt


class DirectedSpectrum(object):
    """Represents directed spectrum and relevant labels.

    Attributes
    ----------
    ds_array : ndarray, shape (n_groups, n_groups, n_frequencies)
        Directed spectrum values between each pair of channel
        groups for each frequency.
    f : ndarray, shape (n_frequencies)
        Frequencies associated with the last dimension of ds_array.
    sources : ndarray of strings, shape (n_groups, n_groups)
        Names the source channel groups for each value in ds_array.
    targets : ndarray of strings, shape (n_groups, n_groups)
        Names the target channel groups for each value in ds_array.
    """

    def __init__(self, ds_array, f, sources, targets):
        self.ds_array = ds_array
        self.f = f
        self.sources = sources
        self.targets = targets

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

    gidx, grouplist = _group_indicies(groups)
    group_pairs = combinations(range(len(gidx)), 2)
    ds_array = np.zeros((len(gidx), len(gidx), len(f)))

    if not pairwise:
        H, Sigma = _wilson_factorize(cpsd, fs)
    for gp in group_pairs:
        idx0 = gidx[gp[0]]
        idx1 = gidx[gp[1]]
        pair_idx = np.hstack((idx0, idx1))
        sub_idx1 = np.hstack((np.zeros(len(idx0)), np.ones(len(idx1))))
        import pdb; pdb.set_trace()
        if pairwise:
            sub_cpsd = cpsd[pair_idx, pair_idx]
            H, Sigma = _wilson_factorize(sub_cpsd, fs)
            ds01, ds10 = _var_to_ds(H, Sigma, sub_idx1)
        else:
            sub_H = H[pair_idx, pair_idx]
            sub_Sigma = Sigma[pair_idx, pair_idx]
            ds01, ds10 = _var_to_ds(sub_H, sub_Sigma, sub_idx1)
        # sum across channels within group
        ds_array[gp[0], gp[1]] = np.sum(ds01)
        ds_array[gp[1], gp[0]] = np.sum(ds10)

    import pdb; pdb.set_trace()
    sources = np.tile(grouplist, (1, len(grouplist)))
    targets = np.tile(grouplist.T, (len(grouplist), 1))

    return DirectedSpectrum(ds_array, f, directed_pairs, sources, targets)

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
                              nfft, return_onesided=False)
            cpsd[c1, c2] = this_csd
            cpsd[c2, c1] = this_csd.T.conj()
    return (cpsd, f)

def _group_indicies(groups):
    """Return list of unique groups and associated indices."""
    grouplist = np.unique(groups)
    gidx = [[g1==g2 for g1 in groups] for g2 in grouplist]
    return (gidx, grouplist)

def _wilson_factorize(cpsd, fs, max_iter=500, tol=1e-9):
    """Factorize CPSD into transfer matrix (H) and covariance (Sigma)

    Implements the algorithm outlined in the following reference:
    G. Tunnicliffe. Wilson, “The Factorization of Matricial Spectral
    Densities,” SIAM J. Appl. Math., vol. 23, no. 4, pp. 420426, Dec.
    1972, doi: 10.1137/0123044.
    """
    psi = _init_psi(cpsd)

    # move frequency to first dimension
    psi = np.moveaxis(psi, 2, 0)
    cpsd = np.moveaxis(cpsd, 2, 0)

    for k in range(max_iter):
        # g = psi \ cpsd / psi* + I
        import pdb; pdb.set_trace()
        psi_inv_cpsd = solve(psi, cpsd)
        g = solve(psi.conj(), psi_inv_cpsd.transpose(0,2,1)).transpose(0,2,1)
        g = g + np.identity(cpsd.shape[-1])

        gplus, gp0 = _plus_operator(g)
        S = -np.tril(gp0, -1)
        S = S - S.H
        gplus = gplus + S # check for upper triang psi0
        psi_prev = psi
        psi = psi @ gplus

        if _check_convergence(psi, psi_prev, tol):
            break

        A = real(ifft(psi)) # real?!?
        Sigma = A[0] @ A[0].T # fs scaling?!?

        H = (solve(A[0].T, psi.T)).T

        return (H, Sigma)

def _init_psi(cpsd):
    """Return initial psi value for wilson factorization."""
    # try other init?
    gamma = ifft(cpsd) # set workers for parallelization?
    gamma0 = np.asmatrix(gamma[...,0])
    # remove assymetry due to rounding error
    gamma0 = np.real((gamma0+gamma0.H)/2)
    h = np.asarray(cholesky(gamma0))
    psi = np.tile(h[...,np.newaxis], (1, 1, gamma.shape[-1]))
    return psi

def _check_convergence(x, x0, tol):
    """Determine whether maximum change is lower than tolerance."""
    import pdb; pdb.set_trace()
    psi_diff = np.abs(x - x0)
    ab_x = np.abs(x)
    ab_x[ab_x < 2*ab_x.eps] = 1
    x_reldiff = x_diff/ab_x
    return x_reldiff.max() < tol

def _var_to_ds(H, Sigma, idx1):
    """Calculate directed spectrum."""
    import pdb; pdb.set_trace()
    idx0 = ~idx1
    H01 = H[:, idx0, idx1]
    H10 = H[:, idx1, idx0]
    sig00 = Sigma[idx0, idx0]
    sig11 = Sigma[idx1, idx1]
    sig01 = Sigma[idx0, idx1]
    sig10 = Sigma[idx1, idx0]
    # conditional covariances
    sig1_0 = sig11 - solve(sig00.T, sig10.T) @ sig01
    sig0_1 = sig00 - solve(sig11.T, sig01.T) @ sig10

    ds10 = np.real(H01 @ sig1_0 @ H01.H)
    ds01 = np.real(H10 @ sig0_1 @ H10.H)
    return (ds01, ds10)
