# Bayesian_block
Bayesian block decomposition for time series analysis.

Dynamic programming algorithm is used for solving
piecewise-constant and piecewise-linear models.
This is based on the algorith presented in Scargle
et al (2013).

One of the many applications is the identification 
of signal and background regions in gamma-ray burst data.
![image](/images/light_curve.png)

### Example Code:
```python
counts = [1, 2, 5, 9, 6, 4, 2, 1, 1]  # Counts or signal values for each bin
lo_edges = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # Lower edges of each bin
hi_edges = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # Upper edges of each bin

BB = BayesianBlock(counts, lo_edges, hi_edges)
ch_points = BB.run_algorithm(ncp_prior=10, mod='constant')
```
### References
[1] [Scargle, J et al. (2013)](https://ui.adsabs.harvard.edu/abs/2013ApJ...764..167S)
