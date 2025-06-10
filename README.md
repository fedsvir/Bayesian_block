# Bayesian_block
This repository provides a Python implementation of the Bayesian Blocks algorithm (Scargle et al. 2013), a non-parametric method for optimal segmentation of 1D sequential data into statistically significant variable-length intervals ("blocks"). The algorithm detects change points by maximizing a fitness function and employs dynamic programming to solve both piecewise-constant and piecewise-linear models. For a brief theoretical overview, see: [outline_algorithm.pdf](outline_algorithm.pdf)

![image](/description/image1.png)
![image](/description/image2.png)

### Toy Example:
```python
from BayesianBlock import BayesianBlock

counts = [1, 1, 5, 9, 6, 4, 1, 1, 1]  # Signal values for each bin
lo_edges = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # Lower edges of each bin
hi_edges = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # Upper edges of each bin

BB = BayesianBlock(counts, lo_edges, hi_edges)
ch_points = BB.run_algorithm(ncp_prior=1, mod='constant')
# Returns [0, 2, 6]
# |1, 1, |5, 9, 6, 4, |1, 1, 1
```
### References
[1] [Scargle, J et al. (2013)](https://ui.adsabs.harvard.edu/abs/2013ApJ...764..167S)
