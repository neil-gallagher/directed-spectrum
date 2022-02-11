"""Directed spectrum class and feature estimator function.

Class
-----
DirectedSpectrum : Represents directed spectrum and relevant labels.

Public Function
---------------
ds : Return a DirectedSpectrum object for multi-channel timeseries data.

Author:  Neil Gallagher
Modified by:  Billy Carson
Date written:    8-27-2021
Last modified:  2-8-2022
"""
from itertools import combinations
from warnings import warn
import numpy as np
from scipy.signal import csd, boxcar
from scipy.fft import fft, ifft
from numpy.linalg import cholesky, solve, eigh

class DirectedSpectrum(object):
    """Directed Spectrum object definition.

    Attributes
    ----------
    ds_array : ndarray
        shape (n_windows, n_frequencies, n_groups, n_groups)
        Directed spectrum values between each pair of channels/groups
        for each frequency and window. Axis 2 corresponds to the source
        channel/group and axis 3 corresponds to the target channel/group.
        For pairwise directed spectrum, elements for which target and source
        are the same are not defined and are returned as NaNs. For 'full'
        models, elements for which the target and source are the same
        correspond to the self-directed spectrum, representing all signal
        in that region that is not explained by any of the other sources in
        the model.
    f : ndarray
        shape (n_frequencies)
        Frequencies associated with axis 1 of ds_array.
    groups : ndarray of strings
        shape (n_groups)
        Names the channels/groups associated with ds_array.
    """

    def __init__(self, ds_array, f, groups):
        # Assign attributes
        self.ds_array = np.array(ds_array, dtype=np.float64)
        self.f = f
        self.groups = groups


def ds(X, f_samp, groups=None, pairwise=False, f_res=None, max_iter=1000,
       tol=1e-6, window='hann', nperseg=None, noverlap=None):
    """Returns a DirectedSpectrum object calculated from data X.

    Calculate the directed spectrum for each directed pair of channel
    groups in X. Applies Wilson's factorization of the cross power
    spectral density matrix associated with X as intermediate steps in
    the calculation.

    Parameters
    ----------
    X : numpy.ndarray
        shape (n_windows, n_channels, n_timepoints)
        Timeseries data from multiple channels. It is assumed that the
        data for each channel are approximately stationary within a window.
        If a 2-D array is input, it is assumed that n_windows is 1.
    f_samp : float
        Sampling rate associated with X.
    groups : list of strings, optional
        shape (n_channels)
        To calculate the Directed Spectrum between groups of channels, this
        should list the group associated with each channel. Otherwise, to
        calculate the directed spectrum between each pair of channels, this
        should be a list of channel names. If 'None' (default), then each
        channel will be given a unique integer index.
    pairwise : bool, optional
        If 'True', calculate the pairwise directed spectrum
        (i.e. calculate seperately for each pair of groups/channels).
        Otherwise, the non-pairwise directed spectrum will be calculated.
        Note the pairwise directed spectrum is not calculated for elements
        where the source and target are the same.
    f_res : float, optional
        Frequency resolution of the calculated spectrum. For example, if
        set to 1, then the directed spectrum will be calculated for
        integer frequency values. If set to 'None' (default), then
        the frequency resolution will be f_samp/nperseg.
    max_iter : int
        Max number of Wilson factorization iterations. If factorization
        does not converge before reaching this value, directed spectrum
        estimates may be inaccurate.
    tol : float
        Wilson factorization convergence tolerance value.
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

    # Check if time series array has appropriate number of dimensions/axes
    # Raise error if time series data does not have enough dimensions
    if X.ndim == 2:
        X = X[np.newaxis]
    elif (X.ndim != 2) & (X.ndim != 3):
        raise ValueError(['Time series data must be 2-dimensional or',
                         '3-dimensional array.'])

    # Create groups list if None passed
    if groups is None:
        groups = list(np.arange(0, X.shape[1], 1))

    cpsd, f = _cpsd_mat(X, f_samp, f_res, window, nperseg, noverlap)
    # move frequency to second dimension.
    cpsd = np.moveaxis(cpsd, 3, 1)

    # Get lists of unique channel groups and associated indices
    group_idx, group_list = _group_indicies(groups)
    G = len(group_list)
    group_pairs = combinations(range(G), 2)

    ds_arr_shape = cpsd.shape[:2] + (G, G)
    ds_array = np.full(ds_arr_shape, np.nan, dtype=np.float64)

    if not pairwise:
        H, Sigma = _wilson_factorize(cpsd, f_samp, max_iter, tol)

    for gp in group_pairs:
        # get indices of both groups in current pair.
        idx0 = np.array(group_idx[gp[0]])
        idx1 = np.array(group_idx[gp[1]])
        pair_idx = np.nonzero(idx0 | idx1)[0]
        sub_idx1 = idx1[pair_idx] # subset of pair_idx in group 1

        if pairwise:
            # Get cross power spectral density matrix corresponding to
            # indices of selected pairs.
            sub_cpsd = cpsd.take(pair_idx, axis=-2).take(pair_idx, axis=-1)

            # Factorize cross power spectral density matrix into transfer
            # matrix (H) and covariance (Sigma).
            H, Sigma = _wilson_factorize(sub_cpsd, f_samp, max_iter, tol)
            ds01, ds10 = _var_to_ds(H, Sigma, sub_idx1)
        else:
            sub_H = H.take(pair_idx, axis=-2).take(pair_idx, axis=-1)
            sub_Sigma = Sigma.take(pair_idx, axis=-2).take(pair_idx, axis=-1)
            ds01, ds10 = _var_to_ds(sub_H, sub_Sigma, sub_idx1)

        # Average across channels within group.
        # TODO: allow for other options besides average here.
        ds_array[:,:, gp[0], gp[1]] = \
            np.diagonal(ds01, axis1=-2, axis2=-1).mean(axis=-1)
        ds_array[:,:, gp[1], gp[0]] = \
            np.diagonal(ds10, axis1=-2, axis2=-1).mean(axis=-1)

    if not pairwise:
        # fill in elements where source equals target with
        # self-directed spectrum
        # = Sum_c(H_gc Sigma_cg) Sigma_gg^-1 Sum_b(H_gb Sigma_bg)*
        for g, g_mask in enumerate(group_idx):
            # TODO: investigate whether calculating H_gg Sigma_gg H_gg
            # separately impact accuracy
            this_H = H[:,:, g_mask]
            this_Sigma =  Sigma[:, np.newaxis, :, g_mask]
            HSig = this_H @ this_Sigma
            HSig_star = HSig.conj().transpose((0,1,3,2))
            g_idx = np.nonzero(g_mask)[0][:,np.newaxis]
            Sigma_gg = Sigma[..., np.newaxis, g_idx, g_idx.T]
            ds_gg =  HSig @ solve(Sigma_gg, HSig_star)

            # TODO: allow for other options matching ds calculation above
            # Diagonal elements should be real, so ignore imaginary portion
            ds_array[..., g, g] = \
                np.diagonal(np.real(ds_gg), axis1=-2, axis2=-1).mean(axis=-1)

    return DirectedSpectrum(ds_array, f, group_list)

def _cpsd_mat(X, f_samp, f_res, window, nperseg, noverlap):
    """Return cross power spectral density and associated frequencies.

    Parameters
    ----------
    X : numpy.ndarray
        shape (n_epochs, n_signals, n_times)
        Timeseries data from multiple signals/channels. Time series data for
        each signal is assumed to be approximately stationary within a
        given epoch. If a 2D array is provided as input, it is assumed that
        n_epcohs is equal to 1.
    f_samp : float
        Sampling rate of time series data X.
    f_res : float
        Frequency resolution of the calculated spectrum. For example, if
        set to 1, then the Directed Spectrum will be calculated for integer
        frequency values. If set to 'None' (default), then nperseg must not
        be 'None' and the frequency resolution will be f_samp/nperseg.
    [Documentation for the following variables was copied and modified from
        the scipy.signal.spectral module. These variables are used for
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
    ------
    (cpsd, f) : tuple
        Tuple consisting of cross power spectral density matrix and array
        associated frequencies.
    """
    # if frequency resolution is set, use it to determine fft length.
    if f_res:
        nfft = int(f_samp/f_res)
    elif nperseg:
        nfft = int(nperseg)
    else:
        raise ValueError('Either nperseg or f_res must be provided.')

    f, cpsd = csd(X[:,:,np.newaxis], X[:,np.newaxis], f_samp, window, nperseg,
                  noverlap, nfft, return_onesided=False, scaling='density')
    # transpose area axes to match convention in paper: positive phase
    # offset indicates source (1st index) lags target (2nd index)
    cpsd = cpsd.transpose(0,2,1,3)
    return (cpsd, f)

def _group_indicies(groups):
    """Return list of unique groups and associated indices.

    Parameters
    ----------
    groups : list of strings
        shape (n_channels)
        Names the group associated with each channel. The directed
        spectrum is calculated between each pair of groups. To calculate
        the directed spectrum between each pair of channels, this should
        be a list of channel names.

    Returns
    -------
    group_list : list
        List of unique group names.
    group_idx : list
        List of indices associated with group names.
    """
    group_list = np.unique(groups)
    group_idx = [[g1==g2 for g1 in groups] for g2 in group_list]
    return (group_idx, group_list)

def _wilson_factorize(cpsd, f_samp, max_iter, tol, eps_multiplier=100):
    """Factorize CPSD into transfer matrix (H) and covariance (Sigma).

    Implements the algorithm outlined in the following reference:
    G. Tunnicliffe. Wilson, “The Factorization of Matricial Spectral
    Densities,” SIAM J. Appl. Math., vol. 23, no. 4, pp. 420426, Dec.
    1972, doi: 10.1137/0123044.

    This code is based on an original implementation in MATLAB provided
    by M. Dhamala (mdhamala@mail.phy-ast.gsu.edu).

    Parameters
    ----------
    cpsd : numpy.ndarray
        Cross power spectral density matrix.
    f_samp : float
        Sampling rate of time series data X.
    max_iter : int
        Max number of Wilson factorization iterations.
    tol : float
        Wilson factorization convergence tolerance value.
    eps_multiplier : int
        Constant multiplier used in stabilizing the Cholesky decomposition
        for positive semidefinite CPSD matrices.

    Returns
    -------
    H : numpy.ndarray
        shape (n_windows, n_frequencies, n_groups, n_groups)
        Wilson factorization solutions for transfer matrix.
    Sigma : numpy.ndarray
        shape (n_windows, n_groups, n_groups)
        Wilson factorization solutions for innovation covariance matrix.
    """
    sign, _ = np.linalg.slogdet(cpsd)
    if np.any(sign == 0):
        warn('CPSD matrix is singular, which may produce inaccurate results.')

    psi, A0 = _init_psi(cpsd)

    # Add diagonal of small values to cross-power spectral matrix to prevent
    # it from being negative semidefinite due to rounding errors
    this_eps = np.spacing(np.abs(cpsd)).max()
    L = cholesky(cpsd + np.eye(cpsd.shape[-1])*this_eps*eps_multiplier)
    #eigval, eigvec = eigh(cpsd)
    #eigval[eigval<0] = 0
    #L = np.sqrt(eigval[...,np.newaxis,:]) * eigvec

    H = np.zeros_like(psi)
    Sigma = np.zeros_like(A0)
    
    for w in range(cpsd.shape[0]):
        for i in range(max_iter):
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
            psi_prev = psi[w].copy()
            psi[w] = psi[w] @ gplus

            A0_prev = A0[w].copy()
            A0[w] = A0[w] @ (g0 + S)

            if (_check_convergence(psi[w], psi_prev, tol) and
                    _check_convergence(A0[w], A0_prev, tol)):
                break
        else:
            warn('Wilson factorization failed to converge.', stacklevel=2)

        # right-side solve
        H[w] = (solve(A0[w].T, psi[w].transpose(0, 2, 1))).transpose(0, 2, 1) 
        Sigma[w] = (A0[w] @ A0[w].T)
    return (H, Sigma)


def _init_psi(cpsd):
    """Return initial psi value for wilson factorization.

    Parameters
    ----------
    cpsd : numpy.ndarray
        Cross power spectral density matrix.

    Returns
    -------
    psi : numpy.ndarray
        shape (n_windows, n_frequencies, n_groups, n_groups)
        Initial value for psi used in Wilson factorization.
    h : numpy.ndarray
        shape (n_windows, n_groups, n_groups)
        Initial value for A0 used in Wilson factorization.
    """
    # TODO: provide other initialization options; test which is best.
    gamma = ifft(cpsd, axis=1)
    gamma0 = gamma[:, 0]

    # remove assymetry in gamma0 due to rounding error.
    gamma0 = np.real((gamma0 + gamma0.conj().transpose(0, 2, 1)) / 2.0)
    h = cholesky(gamma0).conj().transpose(0, 2, 1)
    psi = np.tile(h[:, np.newaxis], (1, gamma.shape[1], 1, 1)).astype(complex)
    return psi, h

def _plus_operator(g):
    """Remove all negative lag components from time-domain representation.

    Parameters
    ----------
    g: numpy.ndarray
        shape (n_frequencies, n_groups, n_groups)
        Frequency-domain representation to which transformation will be applied.

    Returns
    -------
    g_pos : numpy.ndarray
        shape (n_frequencies, n_groups, n_groups)
        Transformed version of g with negative lag components removed.
    gamma[0] : numpy.ndarray
        shape (n_groups, n_groups)
        Zero-lag component of g in time-domain.
    """
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
    """Determine whether maximum relative change is lower than tolerance.

    Parameters
    ----------
    x : numpy.dnarray
        Current matrix/array.
    x0 : numpy.ndarray
        Previous matrix/array
    tol : float
        Tolerance value for convergence check.

    Returns
    -------
    converged : bool
        True indicates convergence has occured, False indicates otherwise.
    """
    x_diff = np.abs(x - x0)
    ab_x = np.abs(x)
    this_eps = np.finfo(ab_x.dtype).eps
    ab_x[ab_x <= 2*this_eps] = 1
    rel_diff = x_diff / ab_x
    converged = rel_diff.max() < tol
    return converged

def _var_to_ds(H, Sigma, idx1):
    """Calculate directed spectrum.

    Parameters
    ----------
    H : numpy.ndarray
        shape (n_windows, n_frequencies, n_groups, n_groups)
        Cross power spectral density transfer matrix.
    Sigma : numpy.ndarray
        shape (n_windows, n_groups, n_groups)
        Cross power spectral density covariance matrix.
    idx1 : numpy.ndarray
        shape (n_groups,)
        Boolean mask indicating which indices are associated with group 1,
        as opposed to group 0.

    Returns
    -------
    (ds01, ds10) : tuple
        Description here.
    """
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
