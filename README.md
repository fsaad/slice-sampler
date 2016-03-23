# Simple slice sampling

Implementation of univariate slice sampler from the following paper,

Neal, R. M. (2003) "Slice sampling" (with discussion), Annals of Statistics,
vol. 31, pp. 705-767.

This software is not professionally developed or debugged, please use only
for illustration purposes.

## Running

To run an interactive demo of the sampler:

```
$ python -i demo.py
```

<a href="http://fsaad.mit.edu/slice.html"><img src="http://web.mit.edu/fsaad/www/ti.png" width="250"/></a>

New target densities can be added by creating a new function in `demo.py` and
changing the appropriate call to invoke your new sampler. For instance:

```python
def my_favorite_density():
    logpdf = lambda x : logsumexp([
        np.log(.5) + norm.logpdf(x, 1, 1),
        np.log(.5) + norm.logpdf(x, 5, .75)])
    domain = (float('-inf'), float('inf'))
    return domain, logpdf

# Obtain target density.
D, logpdf_target = my_favorite_density()
```

## Required Modules
- matplotlib
- numpy
- scipy
