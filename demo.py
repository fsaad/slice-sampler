# -*- coding: utf-8 -*-
# Released under the MIT License; see LICENSE.

import matplotlib.pyplot as plt
import numpy as np

from scipy.integrate import trapz
from scipy.stats import norm
from scipy.misc import logsumexp

from sampler import slice_sample

# Some sample distributions, add your own!

def bimodal_normal():
    logpdf = lambda x : logsumexp([
        np.log(.5) + norm.logpdf(x, 1, 1),
        np.log(.5) + norm.logpdf(x, 5, .75)])
    domain = (0, float('inf'))
    return domain, logpdf

def trimodal_normal():
    logpdf = lambda x : logsumexp([
        np.log(.2)+norm.logpdf(x, 1, .5),
        np.log(.5)+norm.logpdf(x, 4, .75),
        np.log(.3)+norm.logpdf(x, 7, .9)])
    domain = (0, float('inf'))
    return domain, logpdf

# Obtain target density.
D, logpdf_target = trimodal_normal()

# Initial point.
x_start = 4

# Run sampler starting at x =.
samples, metadata = slice_sample(
    x_start, logpdf_target, D, num_samples=20, burn=1, lag=1, w=.5,
    rng=np.random.RandomState(5))

# Prepare figure.
fig, ax = plt.subplots()
ax.grid()

# Plot target.
xvals = np.linspace(0.1, 10, 100)
yvals = np.exp(map(logpdf_target, xvals))
ax.plot(xvals, yvals/trapz(yvals, xvals), lw=3, alpha=.8, c='red',
    label='Target Density')

# Set x,y limits.
ax.set_xlim([0, 10])
ax.set_ylim([0, ax.get_ylim()[1]])

ax.legend(loc='upper left', framealpha=0)

# Launch interactive.
plt.ion()
plt.show()

# Controls speed.
ptime = .5

last_x = x_start
for i in xrange(len(samples)):
    u = metadata['u'][i]
    r = metadata['r'][i]
    a_out = metadata['a_out'][i]
    b_out = metadata['b_out'][i]
    x_proposal = metadata['x_proposal'][i]
    sample = samples[i]
    # Convert u \in (0,f(x)) to direct space.
    u = np.exp(u)
    # Accumulate artists to delete.
    to_delete = []
    # Plot uniform proposal U(0,f(x))
    plt.pause(ptime)
    to_delete.append(
        ax.vlines(last_x, 0, np.exp(logpdf_target(last_x))/trapz(yvals, xvals),
        color='g', linewidth=1.5))
    # Plot auxiliary variable u.
    plt.pause(ptime)
    to_delete.append(
        ax.scatter(last_x, u, color='r', marker='*', s=100))
    # Plot growing a to the left.
    for a in a_out:
        plt.pause(ptime/2.)
        to_delete.append(ax.hlines(u, a, last_x))
        to_delete.append(ax.vlines(a, u, u+.01))
    # Plot growing b to the right.
    for b in b_out:
        plt.pause(ptime/2.)
        to_delete.append(ax.hlines(u, last_x, b))
        to_delete.append(ax.vlines(b, u, u+.01))
    # Plot final sample.
    plt.pause(ptime)
    ax.scatter(sample, u, color='r', marker='o')
    # Plot hline at final sample.
    plt.pause(ptime)
    ax.vlines(sample, 0, 0.025, linewidth=2)
    last_x = sample
    # Remove uline.
    plt.pause(ptime)
    for td in to_delete:
        td.remove()
