# -*- coding: utf-8 -*-
# Released under the MIT License; see LICENSE.

import numpy as np

def slice_sample(x_start, logpdf_target, D, num_samples=1, burn=1, lag=1,
        w=1.0, rng=None):
    """Slice samples from the univariate disitrbution logpdf_target.

    Parameters
    ----------
    x_start : float
        Initial point.
    logpdf_target : function(x)
        Evaluates the log pdf of target distribution at x.
    D : tuple<float, float>
        Support of target distribution.
    num_samples : int, optional
        Number of samples to return, default 1.
    burn : int, optional
        Number of samples to discard before any are collected, default 1.
    lag : int, optional
        Number of moves between successive samples, default 1.
    rng : np.random.RandomState, optional
        Source of random bits.

    Returns
    -------
    samples : list
        `num_samples` length list.
    metadata : dict
        Dictionary of intermediate slice sampling state, used for plotting.

    """
    if rng is None:
        rng = np.random.RandomState(0)

    M = {'u':[], 'r':[], 'a_out':[], 'b_out':[], 'x_proposal':[], 'samples':[]}
    x = x_start

    num_iters = 0
    while len(M['samples']) < num_samples:
        num_iters += 1
        u = np.log(rng.rand()) + logpdf_target(x)
        a, b, r, a_out, b_out = _find_slice_interval(
            logpdf_target, x, u, D, w=w, rng=rng)

        x_proposal = []

        while True:
            x_prime = rng.uniform(a, b)
            x_proposal.append(x)
            if logpdf_target(x_prime) > u:
                x = x_prime
                break
            else:
                if x_prime > x:
                    b = x_prime
                else:
                    a = x_prime

        if burn <= num_iters and num_iters%lag == 0:
            M['u'].append(u)
            M['r'].append(r)
            M['a_out'].append(a_out)
            M['b_out'].append(b_out)
            M['x_proposal'].append(x_proposal)
            M['samples'].append(x)

    return M['samples'], M

def _find_slice_interval(f, x, u, D, w=1.0, rng=None):
    """Return approximated interval under f at height u."""
    if rng is None:
        rng = np.random.RandomState(0)
    r = rng.rand()
    a = x - r*w
    b = x + (1-r)*w
    a_out = [a]
    b_out = [b]
    if a < D[0]:
        a = D[0]
        a_out[-1]= a
    else:
        while f(a) > u:
            a -= w
            a_out.append(a)
            if a < D[0]:
                a = D[0]
                a_out[-1] = a
                break
    if b > D[1]:
        b = D[1]
        b_out[-1] = b
    else:
        while f(b) > u:
            b += w
            b_out.append(b)
            if b > D[1]:
                b = D[1]
                b_out[-1] = b
                break
    return a, b, r, a_out, b_out
