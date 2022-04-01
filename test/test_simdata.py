import numpy as np
from time import process_time
from dir_spec.ds import ds
import matplotlib.pyplot as plt

DATA_FILE = 'test_data.npz'
N_CHANS = 5


def test_wsf(plot=False):
    """Test DS estimation via Wilson spectral factorization."""
    X, area, fs = _load_data()
    start = process_time()
    wsf_ds = ds(X, fs, area, return_onesided=True, nperseg=200, noverlap=150)
    end = process_time()
    print(f'WSF DS: {end-start:.3g}s elapsed')
    
    _check_signal_properties(wsf_ds)
    
    if plot:
        _plot_avg_ds(wsf_ds, title='WSF DS')
    
def test_var(plot=False):
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
    
    if plot:
        _plot_avg_ds(var_ds, title='VAR DS')
    
    
def test_wsf_pds(plot=False):
    """Test PDS estimation via Wilson spectral factorization."""
    X, area, fs = _load_data()
    start = process_time()
    wsf_ds = ds(X, fs, area, pairwise=True, return_onesided=True,
                nperseg=200, noverlap=150)
    end = process_time()
    print(f'WSF PDS: {end-start:.3g}s elapsed')
    
    _check_signal_properties(wsf_ds)

    if plot:
        _plot_avg_ds(wsf_ds, title='WSF PDS')
    
def test_var_pds(plot=False):
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
        
    
if __name__ == "__main__":
    test_wsf(plot=True)
    test_var(plot=True)
    test_wsf_pds(plot=True)
    test_var_pds(plot=True)