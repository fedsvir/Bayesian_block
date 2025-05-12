"""
Bayesian block decomposition for time series analysis.

Dynamic programming algorithm is used for solving
piecewise-constant and piecewise-linear models.
This is based on the algorith presented in Scargle
et al (2013).

One of the many applications is selecting the signal and
background regions of gamma-ray bursts
"""

import numpy as np
from scipy.optimize import  minimize_scalar

class BayesianBlock:
    """
    Class includes the main method: run_algorithm()
    Initial parameters:
        counts (array-like, length N)  data values
        lo_edges (array-like, length N): lower edges of bins
        hi_edges (array-like, length N): upper edges of bins
    Returns:
         An instance of the class
    """
    # Upper bound for the slope parameter 'a' in the piecewise-linear model
    A_MAX = 1

    def __init__(self, counts, lo_edges, hi_edges):
        self._counts = np.array(counts, dtype=float)
        self._counts[self._counts == 0.0] = 1e-10 # to avoid log(0)
        self._lo_edges = np.array(lo_edges)
        self._hi_edges = np.array(hi_edges)
        self._N = len(self._counts)
        self._widths = self._hi_edges - self._lo_edges
        self._ch_points = np.array([0], dtype=int)
        
        self.ncp_prior = self.compute_prior(self._N)
        
    @property
    def ch_points(self):
        return self._ch_points
        
    def compute_prior(self, N, p0=0.01):
        """
        For computing empirical ncp_prior, see eq. 21 in Scargle (2013)
        Note that there was an error in this equation in the original Scargle
        paper (the "log" was missing). The following corrected form is taken
        from https://arxiv.org/abs/1304.2818
        Parameters:
            N (int): number of bins
            p0 (float): false positive rate, such as 0.05
        Returns:
             ncp_prior (float)
        """
        return 4 - np.log(73.53 * p0 * pow(N,-0.478))

    def _f(self, a, cp_i, cp_f, tot_c):
        """
        Auxiliary function for the linear model, using for maximum search.
        Parameters:
            a (float): Slope parameter.
            cp_i (int): First bin number of the block.
            cp_f (int): Last bin number of the block.
            tot_c (float): Total number of data values in the block.

        Returns:
            float: The computed value.
        """
        wn = (1 + a * np.arange(cp_f - cp_i)) * self._widths[cp_i:cp_f]
        N_log_wn = self._counts[cp_i:cp_f] * np.log(wn)
        return tot_c*(np.log(tot_c / np.sum(wn)) - 1) + np.sum(N_log_wn)
    
    def linear_fit(self, cp_i, cp_f, ncp_prior):
        """
        Fitness function for the block of piecewise-linear model.
        """
        tot_c = np.sum(self._counts[cp_i:cp_f])
        # to find maximum of f(a) needs to minimise of -f(a)
        f = lambda a: -self._f(a, cp_i, cp_f, tot_c)
        # maximize the fitness function over parameter 'a'
        res = minimize_scalar(f, bounds=(-1/self._N, self.A_MAX), method='bounded')
        a_max = res.x
            
        return self._f(a_max, cp_i, cp_f, tot_c) - ncp_prior

    def constant_fit(self, cp_i, cp_f, ncp_prior):
        """
        Fitness function for the block of a piecewise-constant model.
        """
        tot_c = np.sum(self._counts[cp_i:cp_f])
        blc_width = self._hi_edges[cp_f - 1] - self._lo_edges[cp_i]
        return tot_c * np.log(tot_c / blc_width) - ncp_prior
        
    def run_algorithm(self, ncp_prior=None, mod='constant'):
        """
        The Scargle algorithm for searching change points
        Parameters:
            ncp_prior (float or None): Parameter that controls the number of blocs,
                            if set to None, an empirical prior is used
            mod (str): Model type, either 'constant' or 'linear'
        Returns:
            np.array: Indices of bins that correspond to change points.
        """
        if ncp_prior != None:
            self.ncp_prior = ncp_prior
        if mod == 'constant':
            self.fitness_func = self.constant_fit
        if mod == 'linear':
            self.fitness_func = self.linear_fit
        
        best = np.zeros(self._N)
        last = np.zeros(self._N, dtype=int)

        # the base case of an induction, i = 0 
        best[0] = self.fitness_func(0, 1, self.ncp_prior)
        last[0] = 0

        for i in range(1, self._N):
            A = np.zeros(i + 1)
            for j in range(i + 1):
                fit = self.fitness_func(j, i + 1, self.ncp_prior)
                A[j] = fit if j == 0 else fit + best[j - 1]
            j_opt = np.argmax(A)
            best[i] = A[j_opt]
            last[i] = j_opt

        # Backtrack to find change points
        ch_points = []
        idx = self._N - 1
        while idx >= 0:
            ch_points.append(last[idx])
            idx = last[idx] - 1

        self._ch_points = np.array(ch_points[::-1], dtype=int)
        return self._ch_points

    def t_xx(self, xx, counts_signal):
        """
        The method is used to find Txx such as T90
        Parameters:
            xx (int): Percent of T100 (e.g., 90 for T05 and T95)
            counts_signal (array-like): Data values of the signal with subtracted background
        Returns:
            tuple: (lo_xx, hi_xx) — the lower and upper time bounds T90
        """

        T100 = (self._ch_points[1], self._ch_points[-1] - 1)
        total_counts = np.sum(counts_signal[T100[0]:T100[1] + 1])

        # Lower bound
        s = 0
        for f in range(T100[0], T100[1] + 1):
            s += counts_signal[f]
            if s >= 0.5 * (1.0 - xx * 0.01) * total_counts:
                lo_xx = self._lo_edges[f]
                break

        # Upper bound
        s = 0
        for i in range(T100[1], T100[0] - 1, -1):
            s += counts_signal[i]
            if s >= 0.5 * (1.0 - xx * 0.01) * total_counts:
                hi_xx = self._hi_edges[i]
                break

        return lo_xx, hi_xx

