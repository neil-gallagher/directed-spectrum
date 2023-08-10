import numpy as np
from time import process_time
from dir_spec.ds import ds, combine_ds
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from copy import deepcopy

# from pdb import set_trace

DATA_FILE = 'test_data.npz'
N_CHANS = 5


def test_wsf(plot=False, norm=None, fnorm_method=None, sigma=6.):
    """Test DS estimation via Wilson spectral factorization."""
    X, area, fs = _load_data()
    start = process_time()
    wsf_ds = ds(X, fs, area, return_onesided=True, nperseg=200, noverlap=150)
    end = process_time()
    print(f'WSF DS: {end-start:.3g}s elapsed')
    
    _check_signal_properties(wsf_ds)

    # test normalization
    if norm:
        wsf_ds.normalize(norm, fnorm_method, filter_sd=sigma)
        if ('frequency' in norm) and (fnorm_method == 'f-inv'):
            norm = list(norm)
            norm.remove('frequency')
        _test_norm(wsf_ds, norm, fnorm_method)
    
    if plot:
        _plot_avg_ds(wsf_ds, title='WSF DS')
    
def test_var(plot=False, norm=None, fnorm_method=None, sigma=6.):
    """Test DS estimation via vector autoregressive modeling."""
    X, area, fs = _load_data()
    start = process_time()
    var_ds = ds(X, fs, area, return_onesided=True,
                estimator='AR', f_res=2., max_ord=10)
    end = process_time()
    print(f'VAR DS: {end-start:.3g}s elapsed')
    
    nan_epochs = np.any(np.isnan(var_ds.ds_array), axis=(1,2,3)).mean()
    print(f'VAR DS: {nan_epochs*100:.1f}% of epochs were ignored due to '
          'unstable VAR model')
    
    _check_signal_properties(var_ds)

    # test normalization
    if norm:
        var_ds.normalize(norm, fnorm_method, filter_sd=sigma)
        if ('frequency' in norm) and (fnorm_method == 'f-inv'):
            norm = list(norm)
            norm.remove('frequency')
        _test_norm(var_ds, norm, fnorm_method)
    
    if plot:
        _plot_avg_ds(var_ds, title='VAR DS')
    
    
def test_wsf_pds(plot=False, norm=None, fnorm_method=None, sigma=6.):
    """Test PDS estimation via Wilson spectral factorization."""
    X, area, fs = _load_data()
    start = process_time()
    wsf_ds = ds(X, fs, area, pairwise=True, return_onesided=True,
                nperseg=200, noverlap=150)
    end = process_time()
    print(f'WSF PDS: {end-start:.3g}s elapsed')
    
    _check_signal_properties(wsf_ds)

    if norm:
        wsf_ds.normalize(norm, fnorm_method, filter_sd=sigma)
        if ('frequency' in norm) and (fnorm_method == 'f-inv'):
            norm = list(norm)
            norm.remove('frequency')
        _test_norm(wsf_ds, norm, fnorm_method)

    if plot:
        _plot_avg_ds(wsf_ds, title='WSF PDS')
    
def test_var_pds(plot=False, norm=None, fnorm_method=None, sigma=6.):
    """Test PDS estimation via vector autoregressive modeling."""
    X, area, fs = _load_data()
    start = process_time()
    var_ds = ds(X, fs, area, pairwise=True, return_onesided=True,
                estimator='AR', f_res=2., max_ord=10)
    end = process_time()
    print(f'VAR PDS: {end-start:.3g}s elapsed')
    
    nan_epochs = np.any(np.isnan(var_ds.ds_array), axis=(1,2,3)).mean()
    print(f'VAR PDS: {nan_epochs*100:.1f}% of epochs were ignored due to '
          'unstable VAR model')
    
    _check_signal_properties(var_ds)

    if norm:
        var_ds.normalize(norm, fnorm_method, filter_sd=sigma)
        if ('frequency' in norm) and (fnorm_method == 'f-inv'):
            norm = list(norm)
            norm.remove('frequency')
        _test_norm(var_ds, norm, fnorm_method)
        
    if plot:
        _plot_avg_ds(var_ds, title='VAR PDS')


def test_combine():
    """Test combining DS objects."""
    X, area, fs = _load_data()
    X1 = X[:50]
    X2 = X[50:]

    # create two DS objects for tests
    my_ds1 = ds(X1, fs, area, return_onesided=True, f_res=2.,
                estimator='AR', max_ord=10)
    my_ds2 = ds(X2, fs, area, return_onesided=True, f_res=2.,
                estimator='AR', max_ord=10)

    # test combining two DS objects, for each type of attribute non-matching
    
    # frequency non-matching
    bad_ds = deepcopy(my_ds2)
    bad_ds.f /= 2
    try:
        combine_ds([my_ds1, bad_ds])
    except ValueError:
        pass    # expected error
    else:
        raise AssertionError('Frequency mismatch between DS objects not '
                             'detected')
    
    # groups non-matching
    bad_ds = deepcopy(my_ds2)
    bad_ds.groups = np.roll(bad_ds.groups, 1)
    try:
        combine_ds([my_ds1, bad_ds])
    except ValueError:
        pass    # expected error
    else:
        raise AssertionError('Group mismatch between DS objects not '
                             'detected')
    
    # params non-matching
    bad_ds = deepcopy(my_ds2)
    bad_ds.params['estimator'] = 'WSF'
    try:
        combine_ds([my_ds1, bad_ds])
    except ValueError:
        pass   # expected error
    else:
        raise AssertionError('Params mismatch between DS objects not '
                             'detected')
    delattr(bad_ds, 'params')
    try:
        combine_ds([my_ds1, bad_ds])
    except ValueError:
        pass    # expected error
    else:
        raise AssertionError('Params mismatch between DS objects not '
                             'detected')
    
    # normalization non-matching
    bad_ds = deepcopy(my_ds2)
    try:
        bad_ds.normalize(('channels',))
        combine_ds([my_ds1, bad_ds])
    except ValueError:
        pass    # expected error
    else:
        raise AssertionError('Normalization mismatch between DS objects not '
                             'detected')
    
    norm_ds1 = deepcopy(my_ds1)
    norm_ds1.normalize(('frequency',))
    try:
        combine_ds([norm_ds1, bad_ds])
    except ValueError:
        pass    # expected error
    else:
        raise AssertionError('Normalization mismatch between DS objects not '
                             'detected')

    # test combining two DS objects
    both_ds = combine_ds([my_ds1, my_ds2])
    _check_combine(my_ds1, my_ds2, both_ds)

    # test combining two DS objects with normalization
    my_ds1.normalize(('frequency', 'channels'))
    my_ds2.normalize(('frequency', 'channels'))
    both_ds = combine_ds([my_ds1, my_ds2])
    _check_combine(my_ds1, my_ds2, both_ds)

    # test combining two DS objects without params
    delattr(my_ds1, 'params')
    delattr(my_ds2, 'params')
    both_ds = combine_ds([my_ds1, my_ds2])
    _check_combine(my_ds1, my_ds2, both_ds)


def _check_combine(my_ds1, other_ds, both_ds):
    if hasattr(both_ds, 'params'):
        assert np.all(my_ds1.params[k] == v
                      for k, v in both_ds.params.items())
        
    assert np.all(both_ds.ds_array == np.concatenate((my_ds1.ds_array,
                                                      other_ds.ds_array),
                                                      axis=0))
    
    assert np.all(both_ds.groups == my_ds1.groups)
    assert np.all(both_ds.f == my_ds1.f)


def _check_signal_properties(this_ds):
    mean_ds = np.nanmean(this_ds.ds_array, axis=0)
    
    # signal at 5Hz should occur in all areas and the following directed pairs:
    # A>B, B>C, C>D, E>A
    band5 = _band_power(mean_ds, this_ds.f, 5)
    # signal at 20Hz should be virtually non-existant in this data
    band20 = _band_power(mean_ds, this_ds.f, 20)
    # signal at 30Hz should occur in A, B, C and the following directed pairs:
    # C>B, B>A
    band30 = _band_power(mean_ds, this_ds.f, 30)
    
    # Self-directed spectrum at 5Hz should be stronger than at 20Hz for all directed
    # area pairs.
    assert np.all(band5.diagonal() > band20.diagonal())
    # Directed spectrum at 5Hz should be stronger than at 20Hz for
    # A>B, B>C, C>D.
    assert np.all(band5.diagonal(offset=1)[:3] > band20.diagonal(offset=1)[:3])
    # Directed spectrum at 30Hz should be stronger than at 20Hz for
    # C>B, B>A.
    assert np.all(band30.diagonal(offset=-1)[:2] > band20.diagonal(offset=-1)[:2])
    # Directed spectrum at 5Hz should be stronger than at 20Hz for
    # E>A.
    assert band5[4,0] > band20[4,0]
       
        
def _plot_avg_ds(this_ds, title=None, low_f=1, high_f=50):
    mean_ds = np.nanmean(this_ds.ds_array, axis=0)
    f_idx = (low_f <= this_ds.f) & (this_ds.f <= high_f)
    
    fig, axs = plt.subplots(N_CHANS, N_CHANS, sharex=True, sharey=True)
    for k in range(N_CHANS):
        for l in range(N_CHANS):
            axs[k,l].plot(this_ds.f[f_idx], mean_ds[f_idx,k,l])
    
    if title:
        fig.suptitle(title)
    plt.show()
    
        
def _load_data():
    with np.load(DATA_FILE) as data:
        fs = data['fs']
        area = data['area']
        X = data['X']
    return (X, area, fs)


def _band_power(mean_ds, f, center, bandwidth=2):
    """Sum DS values over a band of frequencies."""
    low_f = center - bandwidth
    high_f = center + bandwidth
    f_idx = (low_f <= f) & (f <= high_f)
    
    return mean_ds[f_idx].sum(axis=0)
        
    
def _test_norm(ds_obj, norm_type, fnorm_method, sigma=6.):
    # define root mean square function
    rms = lambda arr : np.sqrt(np.mean(arr**2))

    # get list of indices to normalize together based on norm_type
    n_freqs, n_chans = ds_obj.ds_array.shape[-3:-1]
    norm_list = [np.full((n_freqs, n_chans), True)]
    if 'frequency' in norm_type:
        norm_list = ds_obj._split_norm_list(norm_list, axis=0)
    if 'channels' in norm_type:
        norm_list = ds_obj._split_norm_list(norm_list, axis=1)

    if 'diagonals' in norm_type:
        # normalize diagonals
        diag_idx = np.diag_indices(n_chans)
        diags = ds_obj.ds_array[..., diag_idx[0], diag_idx[1]]

        if ('frequency' in norm_type) and (fnorm_method == 'smooth'):
            diag_smooth = gaussian_filter1d(diags, sigma=sigma, axis=1,
                                        mode='nearest')

        for n_mask in norm_list:
            if ('frequency' in norm_type) and (fnorm_method == 'smooth'):
                # normalization factor is not exact here
                norm_fact = rms(diag_smooth[:, n_mask])
                assert((0.2 < norm_fact) and (norm_fact < 5))
            else:
                norm_fact = rms(diags[:, n_mask])
                assert(abs(norm_fact - 1.0) < 1e-6)

        # sum non-diagonals to get non-ds_obj-directed power spectrum,
        # then normalize to balance w/ diagonals
        pow_spec = ds_obj._sum_col(include_diags=False)
        pow_spec /= n_chans - 1

        if ('frequency' in norm_type) and (fnorm_method == 'smooth'):
            pow_spec_smooth = gaussian_filter1d(pow_spec, sigma=sigma, axis=1,
                                        mode='nearest')

        # normalize non-diagonals
        nondiags = ds_obj._get_nondiags()
        for n_mask in norm_list:
            n_idx = np.nonzero(n_mask)
            if ('frequency' in norm_type) and (fnorm_method == 'smooth'):
                # normalization factor is not exact here
                norm_fact = rms(pow_spec_smooth[:, n_idx[0],n_idx[1]])
                assert((0.2 < norm_fact) and (norm_fact < 5))
            else:
                norm_fact = rms(pow_spec[:, n_idx[0],n_idx[1]])
                assert(abs(norm_fact - 1.0) < 1e-6)

        ds_obj._set_nondiags(nondiags)
        return

    elif ds_obj.params['pairwise']:
        warn('norm_type does not contain diagonals option; this is not'
                ' recommended for pairwise directed spectrum!')
        # extract power spectrum from diagonals
        pow_spec = ds_obj.ds_array[..., np.diag_indices(n_chans)]
    else:
        # sum columns to estimate power spectrum for each target
        pow_spec = ds_obj._sum_col()

    if ('frequency' in norm_type) and (fnorm_method == 'smooth'):
        # apply Gaussian kernel to pow_spec
        pow_spec_smooth = gaussian_filter1d(pow_spec, sigma=sigma, axis=1,
                                    mode='nearest')

    # loop through each set of indices and normalize
    for n_mask in norm_list:
        if ('frequency' in norm_type) and (fnorm_method == 'smooth'):
            # normalization factor is not exact here
            norm_fact = rms(pow_spec_smooth[:, n_mask])
            fact_rat = norm_fact/n_chans
            assert((0.2 < fact_rat) and (fact_rat < 5))
        else:
            norm_fact = rms(pow_spec[:, n_mask])
            assert(abs(norm_fact - n_chans) < 1e-6)


if __name__ == "__main__":
    test_wsf(plot=True, norm=('frequency', 'channels'),
             fnorm_method='smooth')
    test_var(plot=True, norm=('channels'))
    test_wsf_pds(plot=True, norm=('diagonals', 'frequency'),
                 fnorm_method=None)
    test_var_pds(plot=True, norm=('frequency', 'diagonals', 'channels'),
                 fnorm_method='f-inv')
    test_combine()