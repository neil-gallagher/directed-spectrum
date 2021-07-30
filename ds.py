""" directed spectrum class and calculation
"""
import numpy as np
from itertools import combinations
from scipy.signal import csd
from scipy.fft import ifft
from numpy.linalg import cholesky, solve

# ?! ordering of functions ?!

class DirectedSpectrum(object):
    """ directed spectrum class"""

    def __init__(self, ds_array, f, sources, targets):
        self.ds_array = ds_array
        self.f = f
        self.sources = sources
        self.targets = targets

def _cpsd_mat(X, fs, window, noverlap):
    C, T = X.shape

    pdb.set_trace()
    n_freqs = np.floor(T/2 + 1)
    cpsd = np.zeros((C, C, n_freqs))
    for c1 in range(C):
        for c2 in range(c1, C):
            f, this_csd = csd(X[c1], X[c2], fs, window, noverlap=noverlap,
                              return_onesided=False) # check freqs!!!
            cpsd[c1, c2] = this_csd
            cpsd[c2, c1] = this_csd.T.conjugate()

    return (this_csd, f)

def _group_indicies(groups):
    pdb.set_trace()
    grouplist = np.unique(groups)
    gidx = [[g1==g2 for g1 in groups] for g2 in grouplist]
    return (gidx, grouplist)

def _init_psi(cpsd):
    # try other init?
    pdb.set_trace()
    gamma = ifft(cpsd) # set workers for parallelization?
    gamma0 = gamma[...,0]
    # remove assymetry due to rounding error
    # !!! check if real is necessary?
    gamma0 = np.real((gamma0+gamma0.H)/2)
    h = cholesky(gamma0)
    return np.tile(h, (1, 1, gamma.shape[-1]))

def _check_convergence(x, x0, tol):
    pdb.set_trace()
    psi_diff = np.abs(x - x0)
    ab_x = np.abs(x)
    ab_x[ab_x < 2*ab_x.eps] = 1
    x_reldiff = x_diff/ab_x
    return x_reldiff.max() < tol

def _wilson_factorize(cpsd, fs, max_iter=500, tol=1e-9):
    psi = _init_psi(cpsd)

    pdb.set_trace()
    # move frequency to first dimension
    psi = np.moveaxis(psi, 2, 0)
    cpsd = np.moveaxis(cpsd, 2, 0)

    cpsd_half = cholesky(cpsd)
    for k in range(max_iter):
        # g = psi \ cpsd / psi + I
        p = solve(psi, cpsd_half)
        g = p@p.H + np.identity(p.shape[-1])
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

        H = (solve(A[0].T, psi)).T

        return (H, Sigma)

def _var_to_ds(H, Sigma, idx1):
    pdb.set_trace()
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

def ds(X, fs, groups, pairwise=False, window, noverlap):

    cpsd, f = _cpsd_mat(X, fs, f, window, noverlap)

    gidx, grouplist = _group_indicies(groups)
    group_pairs = combinations(range(len(gidx)), 2)
    ds_array = np.zeros((len(gidx), len(gidx), len(f)))
    pdb.set_trace()

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
        ds_array[gp[0], gp[1]] = np.sum(ds01)
        ds_array[gp[1], gp[0]] = np.sum(ds10)

    pdb.set_trace()
    sources = np.tile(grouplist, (1, len(grouplist)))
    targets = np.tile(grouplist.T, (len(grouplist), 1))

    return DirectedSpectrum(ds_array, f, directed_pairs, sources, targets)
