import numpy as np
from time import process_time
from dir_spec.ds import ds
import matplotlib.pyplot as plt

DATA_FILE = 'test_data.npz'
N_CHANS = 5


def test_wsf(plot=False, norm=None):
    """Test DS estimation via Wilson spectral factorization."""
    X, area, fs = _load_data()
    start = process_time()
    wsf_ds = ds(X, fs, area, return_onesided=True, nperseg=200, noverlap=150)
    end = process_time()
    print(f'WSF DS: {end-start:.3g}s elapsed')
    
    _check_signal_properties(wsf_ds)

    # test normalization
    if norm:
        wsf_ds.normalize(norm)
        _test_norm(wsf_ds, norm)
    
    if plot:
        _plot_avg_ds(wsf_ds, title='WSF DS')
    
def test_var(plot=False, norm=None):
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
        var_ds.normalize(norm)
        _test_norm(var_ds, norm)
    
    if plot:
        _plot_avg_ds(var_ds, title='VAR DS')
    
    
def test_wsf_pds(plot=False, norm=None):
    """Test PDS estimation via Wilson spectral factorization."""
    X, area, fs = _load_data()
    start = process_time()
    wsf_ds = ds(X, fs, area, pairwise=True, return_onesided=True,
                nperseg=200, noverlap=150)
    end = process_time()
    print(f'WSF PDS: {end-start:.3g}s elapsed')
    
    _check_signal_properties(wsf_ds)

    if norm:
        wsf_ds.normalize(norm)
        _test_norm(wsf_ds, norm)

    if plot:
        _plot_avg_ds(wsf_ds, title='WSF PDS')
    
def test_var_pds(plot=False, norm=None):
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
        var_ds.normalize()
        _test_norm(var_ds, norm)
        
    if plot:
        _plot_avg_ds(var_ds, title='VAR PDS')

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
        
    
def _test_norm(ds_obj, norm_type):
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
        for n_mask in norm_list:
            norm_fact = rms(diags[:, n_mask])
            assert(abs(norm_fact - 1.0) < 1e-6)

        # sum non-diagonals to get non-ds_obj-directed power spectrum,
        # then normalize to balance w/ diagonals
        pow_spec = ds_obj._sum_col(include_diags=False)
        pow_spec /= n_chans - 1

        # normalize non-diagonals
        nondiags = ds_obj._get_nondiags()
        for n_mask in norm_list:
            n_idx = np.nonzero(n_mask)
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

    # loop through each set of indices and normalize
    for n_mask in norm_list:
        norm_fact = rms(pow_spec[:, n_mask])
        assert(abs(norm_fact - n_chans) < 1e-6)


if __name__ == "__main__":
    test_wsf(plot=True, norm=('frequency', 'channels'))
    test_var(plot=True, norm=('channels'))
    test_wsf_pds(plot=True, norm=('diagonals', 'frequency'))
    test_var_pds(plot=True, norm=('frequency', 'diagonals', 'channels'))