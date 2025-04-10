"""Directed spectrum class and feature estimator function.

Class
-----
DirectedSpectrum : Represents directed spectrum and relevant labels.

Public Functions
---------------
ds : Return a DirectedSpectrum object for multi-channel timeseries data.
combine_ds : Combine multiple DirectedSpectrum objects into a single object.

Author:  Neil Gallagher
Modified by:  Billy Carson, Neil Gallagher
Date written:   08-27-2021
Last modified:  01-16-2024
"""
from itertools import combinations
import os
from warnings import warn
import numpy as np
from scipy.signal import csd, welch
from scipy.fft import fft, ifft
from scipy.linalg import lstsq, inv
from numpy.linalg import cholesky, solve
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter1d
import warnings


class DirectedSpectrum(object):
    """Directed Spectrum object definition.

    Attributes
    ----------
    ds_array : ndarray
        shape (n_windows, n_frequencies, n_groups, n_groups)
        Directed spectrum values between each pair of channels/groups
        for each frequency and window. Axis 2 corresponds to the source 
        channel/group and axis 3 corresponds to the target 
        channel/group. For 'full' models, elements for which the target 
        and source are the same correspond to the self-directed 
        spectrum, representing all signal in that region that is not 
        explained by any of the other sources in the model. For pairwise 
        directed spectrum, elements for which target and source are the 
        same are not defined and these entries are populated with the 
        power spectrum for the associated region instead.
    f : ndarray
        shape (n_frequencies)
        Frequencies associated with axis 1 of ds_array.
    groups : ndarray of strings
        shape (n_groups)
        Names the channels/groups associated with ds_array.
    params : dictionary
        Contains the parameters that were used to calculate the values in
        ds_array.

    Methods
    -------
    normalize : Normalize values in ds_array.
    """

    def __init__(self, ds_array, f, groups, params=None):
        # Assign attributes
        self.ds_array = np.array(ds_array, dtype=np.float64)
        self.f = f
        self.groups = groups
        if params:
            self.params = params


    def normalize(self, norm_type=('channels', 'diagonals', 'frequency'),
                  fnorm_method='smooth', filter_sd=6.):
        """Normalize values in ds_array for various use cases. NaN values 
            in ds_array will be ignored.

        Parameters
        ----------
        norm_type : one of {'channels', 'diagonals', 'frequency'} or tuple
                    containing a combination of those strings
                    default = ('channels', 'diagonals', 'frequency')
            if 'channels':
                Normalizes values within each target channel separately.
                This is typically only appropriate when you expect that
                channels have different amplitudes.
            if 'diagonals': 
                Normalizes values with the same source and target channel
                separately from those with different source and target. If
                you are using pairwise DS, it is suggested to use this
                form of normalization, as the 'diagonal' terms are
                populated with power spectrum values instead of DS and
                this normalization is required in order for variances to
                be comparable between the two data types.
            if 'frequency':
                Normalizes each values at each frequency separately. This
                is appropriate for something like electrophysiology data,
                where you expect higher frequencies to have lower
                amplitudes, but want them to be weighted equally.
        fnorm_method : one of {None, 'smooth', 'f-inv'}
                       Chooses the method to account for high correlation
                       of nearby frequencies. Only used if norm_type
                       contains 'frequency'.
            If 'smooth':
                A Gaussian smoothing filter is applied to the frequency
                dimension only for the purposes of calculating
                normalization constants. Other than this the smoothed data
                is not used for generating the final result.
            If 'f-inv':
                Directed spectrum values are scaled by the corresponding
                frequency before all other forms of normalization are
                applied. This assumes that the power spectra for the data
                initially display '1/f' scaling.
            If None:
                No special correction is applied.
        filter_sd : float
            Standard deviation of Gaussian filter applied to frequency
            dimension, in Hz. Only used if norm_type contains 'frequency'
            and fnorm_method is 'smooth'.
        """
        if not hasattr(self, 'params'):
            raise AttributeError('normalize method is not defined if the '
                                 'params attribute is not set')
        
        if hasattr(self, 'norm_params'):
            raise AttributeError(f'This DirectedSpectrum object has already been normalized once and should not be normalized a second time. Previous normalization parameters: {self.norm_params}')
        self.norm_params = {'norm_type':norm_type,
                            'fnorm_method':fnorm_method,
                            'filter_sd':filter_sd}
        
        # define root mean square function
        rms = lambda arr : np.sqrt(np.nanmean(arr**2))

        if 'frequency' in norm_type:
            if fnorm_method == 'f-inv':
                # normalize by 1/f then remove freq from norm_type
                norm_fact = self.f[:, np.newaxis, np.newaxis]
                self.ds_array *= norm_fact

                # normalize variance of whole array
                self.ds_array /= rms(self.ds_array)

                norm_type = list(norm_type)
                norm_type.remove('frequency')
            elif fnorm_method == 'smooth':
                f_res = self.f[1]-self.f[0]
                sigma = filter_sd/f_res
            else:
                raise ValueError('fnorm_method must be one of {None, '
                                 '\'smooth\', \'f-inv\'')

        # get list of indices to normalize together based on norm_type
        n_freqs, n_chans = self.ds_array.shape[-3:-1]
        norm_list = [np.full((n_freqs, n_chans), True)]
        if 'frequency' in norm_type:
            norm_list = self._split_norm_list(norm_list, axis=0)
        if 'channels' in norm_type:
            norm_list = self._split_norm_list(norm_list, axis=1)

        if 'diagonals' in norm_type:
            # normalize diagonals
            diag_idx = np.diag_indices(n_chans)
            diags = self.ds_array[..., diag_idx[0], diag_idx[1]]
            if ('frequency' in norm_type) and (fnorm_method == 'smooth'):
                # apply Gaussian kernel to diags
                diags = gaussian_filter1d(diags, sigma=sigma, axis=1,
                                          mode='nearest')

            for n_mask in norm_list:
                norm_fact = rms(diags[:, n_mask])
                diags[:, n_mask] /= norm_fact
            self.ds_array[..., diag_idx[0], diag_idx[1]] = diags

            # sum non-diagonals to get non-self-directed power spectrum,
            # then normalize to balance w/ diagonals
            pow_spec = self._sum_col(include_diags=False)
            pow_spec /= n_chans - 1
            if ('frequency' in norm_type) and (fnorm_method == 'smooth'):
                # apply Gaussian kernel to pow_spec
                pow_spec = gaussian_filter1d(pow_spec, sigma=sigma, axis=1,
                                             mode='nearest')

            # normalize non-diagonals
            nondiags = self._get_nondiags()
            for n_mask in norm_list:
                n_idx = np.nonzero(n_mask)
                norm_fact = rms(pow_spec[:, n_idx[0],n_idx[1]])
                nondiags[:,n_idx[0],:,n_idx[1]] /= norm_fact

            self._set_nondiags(nondiags)
            return

        elif self.params['pairwise']:
            warn('norm_type does not contain diagonals option; this is not'
                 ' recommended for pairwise directed spectrum!')
            # extract power spectrum from diagonals
            pow_spec = self.ds_array[..., np.diag_indices(n_chans)]
        else:
            # sum columns to estimate power spectrum for each target
            pow_spec = self._sum_col()
        
        if ('frequency' in norm_type) and (fnorm_method == 'smooth'):
            # apply Gaussian kernel to pow_spec
            pow_spec = gaussian_filter1d(pow_spec, sigma=sigma, axis=1,
                                         mode='nearest')


        # loop through each set of indices and normalize
        for n_mask in norm_list:
            norm_fact = rms(pow_spec[:, n_mask]) / n_chans
            n_idx = np.nonzero(n_mask)
            self.ds_array[:,n_idx[0],:,n_idx[1]] /= norm_fact


    def _set_nondiags(self, nondiag_vals):
        """Set 'nondiagonal' channel pair elements """
        n_chans = nondiag_vals.shape[-1]
        nondiag_idx = ~np.eye(n_chans, dtype=bool)
        for c in range(n_chans):
            self.ds_array[...,nondiag_idx[:,c], c] = nondiag_vals[...,c]


    def _get_nondiags(self):
        """Get 'nondiagonal' channel pair elements """
        n_win, n_freqs, n_chans = self.ds_array.shape[:3]
        nondiag_idx = ~np.eye(n_chans, dtype=bool)

        # transpose target/source to simplify extraction
        ds_arr = self.ds_array.copy().transpose(0,1,3,2)
        nondiags = ds_arr[...,nondiag_idx]
        nondiags = nondiags.reshape((n_win, n_freqs, n_chans, n_chans-1)
                                   ).transpose(0,1,3,2)
        return nondiags


    def _split_norm_list(self, norm_list, axis):
        """ Split a list of normalization indices along an axis. """
        new_norm_list = []
        arr_shape = norm_list[0].shape
        for norm_idx in norm_list:
            for k in range(arr_shape[axis]):
                this_norm = np.full(arr_shape, False)
                this_vals = norm_idx.take(indices=k, axis=axis)
                if axis == 0:
                    this_norm[k,:] = this_vals
                if axis == 1:
                    this_norm[:,k] = this_vals
                new_norm_list.append(this_norm)
        return new_norm_list


    def _sum_col(self, include_diags=True):
        """ Returns the sum of spectrums with the same target channel. """
        if include_diags:
            return self.ds_array.sum(axis=-2)
        else:
            arr_copy = self.ds_array.copy()
            n_chans = arr_copy.shape[-1]
            diag_idx = np.diag_indices(n_chans)
            arr_copy[..., diag_idx[0], diag_idx[1]] = 0
            return arr_copy.sum(axis=-2)


def ds(X, f_samp, groups=None, pairwise=False, f_res=None,
       return_onesided=False, estimator='Wilson',
       order='multi-aic', max_ord=50, ord_est_epochs=20, n_jobs=None,
       max_iter=1000, tol=1e-6, window='hann', nperseg=None, noverlap=None):
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
        the frequency resolution will be f_samp/nperseg if estimator is
        'Wilson' or f_samp/order if estimator is 'AR' and pairwise is
        False. f_res must be set if estimator is 'AR' and pairwise is
        True.
    return_onesided : bool, optional
        If True, return a one-sided spectrum. If False return a
        two-sided spectrum. Must be False if the input timeseries is
        complex. Defaults to False.
    estimator : {'Wilson', 'AR'}, optional
        Method to use for estimating the directed spectrum. 'Wilson' is
        Wilson's spectral factorization of the data cross-spectral
        density matrix. 'AR' fits an autoregressive model to the data.
        Defaults to 'Wilson'.
    order : int or 'aic' or 'multi-aic', optional
        Autoregressive model order. If 'aic', uses Akaike Information
        Criterion (AIC) to automatically determine one model order for all
        epochs. If 'multi-aic', uses AIC to deterimine different model
        orders for each epoch individually. Used only when estimator is
        'AR'. Defaults to 'multi-aic'.
    max_ord : int, optional
        Maximum autoregressive model order. Only used when estimaotr is
        'AR' and order is 'aic'. Default is 50.
    ord_est_epochs : int, optional
        Number of epochs to sample from full dataset for estimating
        model order. Only used when estimator is 'AR' and order is
        'aic'. The highest AIC value from all sampled windows is used
        to select the model order. Default is 20.
    n_jobs : int, optional
        Maximum number of jobs to use for parallel calculation of AIC.
        Only used when order is 'aic' or 'multi-aic'. If set to 1,
        parallel computing is not used. Default is None, which is
        interpreted as 1 unless the call is performed under a
        parallel_backend context manager that sets another value for
        n_jobs.
    max_iter : int, optional
        Max number of Wilson factorization iterations. If factorization
        does not converge before reaching this value, directed spectrum
        estimates may be inaccurate. Used only when estimator is
        'Wilson'. Defaults to 1000.
    tol : float, optional
        Wilson factorization convergence tolerance value. Used only when
        estimator is 'Wilson'. Defaults to 1e-6.
    [Documentation for the following variables was copied from the
        scipy.signal.spectral module. These variables are used for
        calculating the cross power spectral density matrix when
        estimator is 'Wilson' or to calculate power spectral density
        when estimator is 'AR' and pairwise is True, and are not used
        otherwise.]
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
    # store non-data parameters for saving later
    param_dict = locals().copy()
    param_dict.pop('X')

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

    # Get lists of unique channel groups and associated indices
    group_idx, group_list = _group_indicies(groups)
    G = len(group_list)
    group_pairs = combinations(range(G), 2)
    
    nfft, nperseg = _calc_nfft(f_samp, f_res, nperseg, window)

    estimator = estimator.lower()
    if (estimator == 'ar') and (order == 'multi-aic') and (not nfft):
        raise ValueError('f_res must be set if estimator is \'AR\' and '
                         'order is \'multi-aic\'')
    
    if estimator == 'wilson':
        # demean data
        X -= X.mean(axis=-1, keepdims=True)
        cpsd, f = _cpsd_mat(X, f_samp, window, nperseg, noverlap, nfft)
        if not pairwise:
            H, Sigma = _wilson_factorize(cpsd, f_samp, max_iter, tol)
    elif estimator == 'ar':
        if not pairwise:
            A, Sigma = _fit_var(X, order, max_ord, ord_est_epochs, n_jobs)
            H = _var_to_transfer(A, nfft)
    else:
        raise ValueError(f'Unsupported value for parameter \'estimator\':'
                         f' {estimator}')

    # initialize ds array
    if nfft:
        ds_arr_shape = (X.shape[0], nfft, G, G)
    elif estimator == 'wilson':
        ds_arr_shape = (X.shape[0], len(f), G, G)
    elif not pairwise:
        ds_arr_shape = (X.shape[0], H.shape[1], G, G)
        warn('f_res has not been set, so frequency resolution will be '
             'determined by model order')
    else:
        raise ValueError('f_res must be set to a real value if '
                         '\'estimator\'=\'AR\' and \'pairwise\'=True')
    ds_array = np.full(ds_arr_shape, np.nan, dtype=np.float64)
    
    for gp in group_pairs:
        # get indices of both groups in current pair.
        idx0 = np.array(group_idx[gp[0]])
        idx1 = np.array(group_idx[gp[1]])
        pair_idx = np.nonzero(idx0 | idx1)[0]
        sub_idx1 = idx1[pair_idx] # subset of pair_idx in group 1

        if pairwise:
            if estimator == 'wilson':
                # Get cross power spectral density matrix corresponding to
                # indices of selected pairs.
                sub_cpsd = cpsd.take(pair_idx, axis=-2).take(pair_idx, axis=-1)

                # Factorize cross power spectral density matrix into transfer
                # matrix (H) and covariance (Sigma).
                H, Sigma = _wilson_factorize(sub_cpsd, f_samp, max_iter, tol)
            else: # AR model estimation
                sub_X = X.take(pair_idx, axis=1)
                A, Sigma = _fit_var(sub_X, order, max_ord, ord_est_epochs,
                                    n_jobs, print_ord=False)
                H = _var_to_transfer(A, nfft)

            ds01, ds10 = _var_to_ds(H, Sigma, sub_idx1)
        else:
            sub_H = H.take(pair_idx, axis=-2).take(pair_idx, axis=-1)
            sub_Sigma = Sigma.take(pair_idx, axis=-2).take(pair_idx, axis=-1)
            ds01, ds10 = _var_to_ds(sub_H, sub_Sigma, sub_idx1)

        # Average across channels within group.
        # TODO: allow for other options besides average here.
        ds_array[:,:, gp[0], gp[1]] = np.nanmean( 
                                        np.diagonal(ds01, axis1=-2, axis2=-1),
                                        axis=-1)
        ds_array[:,:, gp[1], gp[0]] = np.nanmean(
                                        np.diagonal(ds10, axis1=-2, axis2=-1),
                                        axis=-1)

    if pairwise:
        # fill in elements where source equals target with power spectrum
        if estimator == 'ar':
            f, psd = welch(X, f_samp, window, nperseg, noverlap, nfft,
                        return_onesided=False, scaling='density')
            psd = psd.swapaxes(-1, -2)
            # compensate for f_samp scaling that will be applied below
            # since it shouldn't actually be applied to diagonals here
            psd *= f_samp
        else:
            psd = np.real(np.diagonal(cpsd, axis1=-2, axis2=-1))

        # average channels within each group
        for g, g_mask in enumerate(group_idx):
            # TODO: allow for other options besides average, matching
            # non-diagonal elements
            ds_array[..., g, g] = psd[..., g_mask].mean(axis=-1)

    else:
        # fill in elements where source equals target with
        # self-directed spectrum
        for g, g_mask in enumerate(group_idx):
            ds_gg = _self_ds(H, Sigma, g_mask)

            # TODO: allow for other options matching ds calculation above
            # Diagonal elements should be real, so ignore imaginary portion
            ds_array[..., g, g] = \
                np.diagonal(np.real(ds_gg), axis1=-2, axis2=-1).mean(axis=-1)

    if estimator=='ar':
        ds_array /= f_samp

    if return_onesided:
        if (estimator=='ar') and not pairwise:
            # f array hasn't been created yet
            f = np.fft.fftfreq(H.shape[1], 1/f_samp)

        # convert to one sided spectrum
        nyquist = np.floor(len(f)/2).astype(int)
        ds_array = ds_array[:,:(nyquist+1)]
        ds_array[:, 1:nyquist] *= 2

        if len(f) % 2 != 0:
            ds_array[:, nyquist] *= 2
        f = np.abs(f[:(nyquist+1)])

    return DirectedSpectrum(ds_array, f, group_list, param_dict)


def combine_ds(ds_list):
    """Combine multiple DirectedSpectrum objects into a single object.
    
    Parameters
    ----------
    ds_list : list of DirectedSpectrum objects
        List of DirectedSpectrum objects to combine.
    """
    # check that all ds objects have same groups, frequencies, parameters,
    # and normalization, if applicable
    first_ds = ds_list[0]
    if not all([np.all((ds.groups == first_ds.groups)) for ds in ds_list]):
        raise ValueError('All DirectedSpectrum objects must have the same '
                         'groups attribute.')
    
    if not all([np.all(ds.f == first_ds.f) for ds in ds_list]):
        raise ValueError('All DirectedSpectrum objects must have the same '
                         'frequencies.')
    
    has_params = [hasattr(ds, 'params') for ds in ds_list]
    if any(has_params):
        # check if any params dictionaries don't match
        if ((not all(has_params)) or 
            (not np.all([[np.all(first_ds.params[k] == v)
                          for k, v in ds.params.items()]
                          for ds in ds_list]))):
            raise ValueError('All DirectedSpectrum objects must have been'
                             ' generated with the same parameters.')
        
    has_norm_params = [hasattr(ds, 'norm_params') for ds in ds_list]
    if any(has_norm_params):
        if ((not all(has_norm_params)) or
            (not all([ds.norm_params == first_ds.norm_params
                      for ds in ds_list]))):
            raise ValueError('All DirectedSpectrum objects must have been'
                             ' normalized with the same parameters.')

    # merge ds arrays
    ds_array = np.concatenate([ds.ds_array for ds in ds_list], axis=0)

    # return new ds object with combined ds array and other attributes
    if hasattr(first_ds, 'params'):
        my_ds = DirectedSpectrum(ds_array, first_ds.f, first_ds.groups,
                                first_ds.params)
    else:
        my_ds = DirectedSpectrum(ds_array, first_ds.f, first_ds.groups)
    if hasattr(first_ds, 'norm_params'):
        my_ds.norm_params = first_ds.norm_params
    return my_ds


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
        shape (n_windows, n_frequencies, n_signals, n_signals)
        Wilson factorization solutions for transfer matrix.
    Sigma : numpy.ndarray
        shape (n_windows, n_signals, n_signals)
        Wilson factorization solutions for innovation covariance matrix.
    """
    cpsd_cond = np.linalg.cond(cpsd)
    if np.any(cpsd_cond > (1/ np.finfo(cpsd.dtype).eps)):
        warn('CPSD matrix is singular within numerical tolerance, which may produce inaccurate results.')
        # Add diagonal of small values to cross-power spectral matrix to prevent
        # it from being negative semidefinite due to rounding errors
        this_eps = np.spacing(np.abs(cpsd)).max()
        cpsd = cpsd + np.eye(cpsd.shape[-1])*this_eps*eps_multiplier

    psi, A0 = _init_psi(cpsd)

    L = cholesky(cpsd)
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


def _fit_var(X, order, max_ord, ord_est_epochs, n_jobs, print_ord=True):
    """Fit vector autoregressive model to data.

    Parameters
    ----------
    X : numpy.ndarray
        shape (n_epochs, n_signals, n_times)
        Timeseries data from multiple signals/channels. Time series data for
        each signal is assumed to be approximately stationary within a
        given epoch.
    order : int, optional
        Autoregressive model order (i.e. number of lags). If set to
        'aic' or 'multi-aic', then order is chosen automatically using
        Akaike information criterion on a small subset of the data.
    max_ord : int, optional
        Maximum autoregressive model order. Only used when order is
        'aic'. Default is 50.
    ord_est_epochs : int, optional
        Number of epochs to sample from full dataset for estimating
        model order. Only used when order is 'aic'. Default is 30.
    n_jobs : int, optional
        Maximum number of jobs to use for parallel calculation of AIC.
        Only used when order is 'aic'. If set to 1, parallel computing
        is not used. Default is None, which is interpreted as 1 unless
        the call is performed under a parallel_backend context manager
        that sets another value for n_jobs.
    print_ord: bool, optional
        Indicates whether to print chosen order when order is 'aic'.
        Defaults to True.
    Returns
    -------
    A : numpy.ndarray
        shape (n_epochs, n_lags, n_signals, n_signals)
        Autoregressive matrices of the VAR model for each epoch.
    Sigma : numpy.ndarray
        shape (n_epochs, n_signals, n_signals)
        Innovation covariance matrix of the VAR model for each epoch.
    """
    # demean data
    X -= X.mean(axis=-1, keepdims=True)

    if order in ('aic', 'multi-aic'):
        # estimate model order using AIC
        rng = np.random.default_rng()
        if (order == 'aic')  and (ord_est_epochs < X.shape[0]):
            # use a subset of the data for estimating model order
            samp_X = rng.choice(X, ord_est_epochs, replace=False)
        else:
            ord_est_epochs = X.shape[0]
            samp_X = X
        n_samps = samp_X.shape[2]

        aic = np.zeros((max_ord, ord_est_epochs))
        max_ord = min(max_ord, n_samps-1)
        # ignore warnings due to poorly conditioned sample windows
        os.environ['PYTHONWARNINGS'] = 'ignore:invalid value:RuntimeWarning'
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            aic_A_Sig = Parallel(n_jobs=n_jobs)(
                            delayed(_calc_aic)(o, samp_X, ord_est_epochs)
                                                for o in range(max_ord))
        os.environ['PYTHONWARNINGS'] = 'default'
        # split aic, A, and Sigma into separate lists
        aic, A, Sigma = zip(*aic_A_Sig)
        aic = np.asarray(aic)

        if order == 'aic':
            try:
                order = np.nanargmin(aic.max(axis=1)) + 1
            except ValueError:
                print('Could not find an AR model order that produced '
                    'consistently stable (spectral radius > 1) models. '
                    'Try preprocssing your data differently to increase '
                    'stationarity, or setting pairwise=False.')
                raise
            if print_ord:
                print(f'Model order {order:d} selected using AIC.')
        else:
            try:
                order = np.nanargmin(aic, axis=0) + 1
                bad_epoch = False
            except ValueError:
                warn('VAR model of data is not stable for at least one epoch '
                '(spectral radius > 1); directed spectrum values for these'
                ' epcohs will be set to NaN. Try changing max model order, '
                'estimating model order with more epochs, '
                'preprocssing your data differently to increase stationarity,'
                ' or setting pairwise=False.')
                bad_epoch = np.where(np.isnan(aic).all(axis=0))[0]
                aic = np.delete(aic, bad_epoch, axis=1)
                order = np.nanargmin(aic, axis=0) + 1
                for be in bad_epoch:
                    order = np.insert(order, be, 1)
                
            if print_ord:
                print(f'Model orders range between {order.min():d} and '
                      f'{order.max():d}, with a mean of {order.mean():.1f}'
                      ' using AIC.')

    if np.isscalar(order):
        # fit final VAR models for non multi-aic case
        A, Sigma, bad_epoch = _fit_var_helper(X, order)
        if np.any(bad_epoch):
            warn('VAR model of data is not stable for at least one epoch '
                '(spectral radius > 1); directed spectrum values for these'
                ' epcohs will be set to NaN. Try changing model order, '
                'estimating model order with more epochs if using AIC, '
                'preprocssing your data differently to increase stationarity,'
                ' or setting pairwise=False.')
    else:
        # select correct A and Sigma values for multi-aic case
        A = [A[o-1][e] for e,o in enumerate(order)]
        Sigma = np.asarray([Sigma[o-1][e] for e,o in enumerate(order)])

    return (A, Sigma)


def _calc_aic(o, samp_X, ord_est_epochs):
    order = o+1
    # return biased ML estimate of sigma for AIC formula
    A, Sigma, bad_epoch = _fit_var_helper(samp_X, order,
                                          sigma_biased=True)
    n_params = A.size/ord_est_epochs
    n_obs = samp_X.shape[2] - order
    sign, logdetsig = np.linalg.slogdet(Sigma)
    # if sign is 0, Sigma is singular, so model is unstable
    logdetsig[sign==0] = np.nan
    aic = logdetsig + 2*n_params/(n_obs-n_params-1)
    return (aic, A, Sigma)


def _var_to_transfer(A, nfft):
    """Calculate transfer matrix (H) from autoregressive matrices.

    Parameters
    ----------
    A : numpy.ndarray -or- list of numpy.ndarrays
        shape (n_epochs, n_lags, n_signals, n_signals) -or- len (n_epochs)
        Autoregressive matrices of the VAR model for each epoch. If a list
        of numpy.ndarrays is passed, then each element of the list should
        be a numpy.ndarray of shape (n_lags_i, n_signals, n_signals), where
        n_lags_i is the number of lags (i.e. order) for the ith epoch.
    nfft: int, optional
        Length of the FFT used, if a zero padded FFT is desired.

    Returns
    -------
    H : numpy.ndarray
        shape (n_windows, n_frequencies, n_groups, n_groups)
        VAR solutions for transfer matrix.
    """
    n_epochs = len(A)
    if nfft:
        n_freqs = nfft
    else:
        n_freqs = A[0].shape[0] + 1
    n_chans = A[0].shape[-1]
    H = np.zeros((n_epochs, n_freqs, n_chans, n_chans), dtype=np.complex128)
    id_mat = np.eye(n_chans)[np.newaxis,...]

    # calc transfer matrix individually for each epoch
    for e, A_e in enumerate(A):
        # concatenate I and -A along order/lag axis, then take FFT
        ia = np.concatenate((id_mat, -A_e), axis=0)
        iaf = fft(ia, nfft, axis=0)

        for l, iaf_l in enumerate(iaf):
            # check for nan values in iaf before invert
            if np.any(np.isnan(iaf_l)):
                H[e,l] = np.nan
            else:
                H[e,l] = inv(iaf_l, check_finite=False, overwrite_a=True)
    return H


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


def _self_ds(H, Sigma, g_mask):
    """Calculate self-directed spectrum"""
    # = Sum_c(H_gc Sigma_cg) Sigma_gg^-1 Sum_b(H_gb Sigma_bg)*
    
    # TODO: investigate whether calculating H_gg Sigma_gg H_gg
    # separately impact accuracy
    this_H = H[:,:, g_mask]
    this_Sigma =  Sigma[:, np.newaxis, :, g_mask]
    HSig = this_H @ this_Sigma
    HSig_star = HSig.conj().transpose((0,1,3,2))
    g_idx = np.nonzero(g_mask)[0][:,np.newaxis]
    Sigma_gg = Sigma[..., np.newaxis, g_idx, g_idx.T]
    ds_gg =  HSig @ solve(Sigma_gg, HSig_star)
    return ds_gg

    
def _cpsd_mat(X, f_samp, window, nperseg, noverlap, nfft):
    """Return cross power spectral density and associated frequencies.

    Parameters
    ----------
    X : numpy.ndarray
        shape (n_epochs, n_signals, n_times)
        Timeseries data from multiple signals/channels. See ds docstring
        for more details.
    f_samp : float
        Sampling rate of time series data X.
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
    nfft: int, optional
        Length of the FFT used, if a zero padded FFT is desired. If None,
        the FFT length is nperseg.

    Returns
    ------
    (cpsd, f) : tuple
        Tuple consisting of cross power spectral density matrix and array
        associated frequencies.
    """
    
    f, cpsd = csd(X[:,:,np.newaxis], X[:,np.newaxis], f_samp, window, nperseg,
                  noverlap, nfft, return_onesided=False,
                  scaling='density')

    # transpose area axes to match convention in paper: positive phase
    # offset indicates source (1st index) lags target (2nd index)
    cpsd = cpsd.transpose(0,2,1,3)

    # move frequency to second dimension.
    cpsd = np.moveaxis(cpsd, 3, 1)

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


def _calc_nfft(f_samp, f_res, nperseg, window):
    """Calculate window length for fft"""
    # if frequency resolution is set, use it to determine fft length.
    if f_res:
        nfft = int(f_samp/f_res)
    else:
        # set to None to use default
        nfft = None
        
    # check if window is an array, which should define nperseg
    if np.ndim(window) == 1:
        nperseg = len(window)
    
    # if nperseg is unset, set it to nfft, which prevents undesired
    # behavior in csd/welch
    if not nperseg:
        nperseg = nfft
        
    return (nfft, nperseg)


def _fit_var_helper(X, order, sigma_biased=False):
    # parse X into unlagged (dependent) and lagged (independent) components
    X_unlag = X[..., order:].swapaxes(1,2)
    n_epochs, n_samps, n_signals = X_unlag.shape
    lag_idx = np.arange(order-1, -1, -1)[:,np.newaxis] + np.arange(n_samps)
    X_lag = X[:, :, lag_idx]
    X_lag = X_lag.reshape((n_epochs, -1, n_samps)).swapaxes(1,2)

    if order > n_samps:
        warn('VAR model is underspecified, which could lead to '
             'inaccurate results. To fix this, increase the number of '
             'samples per epoch, reduce the model order, or switch to '
             'Wilson spectral factorization for estimation.')
    
    # solve VAR model via least squares
    A = np.full((n_epochs, n_signals*order, n_signals), np.nan)
    Sigma = np.full((n_epochs, n_signals, n_signals), np.nan)
    
    for e in range(n_epochs):
        # lstsq expects n_samps as first dim, so transpose axes
        A[e], _,_,_ = lstsq(X_lag[e], X_unlag[e], cond=None, lapack_driver='gelsy')
        resid = X_unlag[e] - X_lag[e]@A[e]
        Sigma[e] = np.cov(resid, rowvar=False, bias=sigma_biased)
            
    # reshape A so that X[...,t] = Sum_p A[:,p] X[...,t-p]
    A = A.reshape((n_epochs, n_signals, order, n_signals)).transpose((0,2,3,1))

    # check spectral radius of A for instability
    bad_epoch = _check_specrad(A)
    A[bad_epoch] = np.nan
    Sigma[bad_epoch] = np.nan
    return (A, Sigma, bad_epoch)


def _check_specrad(A):
    """ Return spectral radius for the associated VAR model"""
    n_epochs, order, n_signals, _ = A.shape
    var_mat1 = A.swapaxes(2,3).reshape((n_epochs, -1, n_signals))
    var_mat2 = np.broadcast_to(np.vstack((np.eye(n_signals*(order-1)),
                                          np.zeros((n_signals,
                                                    n_signals*(order-1))))),
                               (n_epochs, n_signals*order, n_signals*(order-1)))
    var_mat = np.concatenate((var_mat1, var_mat2), axis=2)
    specrad = abs(np.linalg.eigvals(var_mat)).max(axis=1)
    return (specrad >=1)


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
    psi = np.tile(h[:, np.newaxis], (1, cpsd.shape[1], 1, 1)).astype(complex)
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
    N = np.floor(F/2).astype(int)
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
