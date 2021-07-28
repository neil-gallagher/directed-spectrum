""" directed spectrum class and calculation
"""
import numpy as np
from itertools import combinations

class DirectedSpectrum(object):
    """ directed spectrum class"""

    def __init__(self, ds_array, f, sources, targets):
        self.ds_array = ds_array
        self.f = f
        self.sources = sources
        self.targets = targets

def _cpsd_mat(X, fs, f, window, noverlap):

def _group_indicies(groups):
    grouplist = np.unique(groups)
    #[[for g in groups]]
    

def _wilson_factorize(cpsd, fs):

def _var_to_ds(H, Sigma, idx1):

def ds(X, fs, f, groups, pairwise=False, window, noverlap):

    cpsd = _cpsd_mat(X, fs, f, window, noverlap)

    gidx, grouplist = _group_indicies(groups)
    group_pairs = combinations(range(len(gidx)), 2)
    ds_array = np.zeros((len(gidx),len(gidx),len(f)))

    if not pairwise:
        H, Sigma = _wilson_factorize(cpsd, fs)
    for gp in group_pairs:
        idx0 = gidx[gp[0]]
        idx1 = gidx[gp[1]]
        pair_idx = np.hstack((idx0, idx1))
        sub_idx1 = np.hstack((np.zeros(len(idx0)), np.ones(len(idx1))))
        if pairwise:
            sub_cpsd = cpsd[pair_idx, pair_idx]
            H, Sigma = _wilson_factorize(sub_cpsd, fs)
            ds01, ds10 = _var_to_ds(H, Sigma, sub_idx1)
        else:
            sub_H = H[pair_idx, pair_idx]
            sub_Sigma = Sigma[pair_idx, pair_idx]
            ds01, ds10 = _var_to_ds(sub_H, sub_Sigma, sub_idx1)
        # sum across channels within group
        ds_array[gp[0],gp[1]] = np.sum(ds01)
        ds_array[gp[1],gp[0]] = np.sum(ds10)

    sources = np.tile(grouplist, (1, len(grouplist)))
    targets = np.tile(grouplist.T, (len(grouplist),1))

    return DirectedSpectrum(ds_array, f, directed_pairs, sources, targets)
