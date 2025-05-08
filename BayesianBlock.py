import numpy as np
from scipy.optimize import  minimize_scalar

class BayesianBlock:

    def __init__(self, counts, lo_edges, hi_edges):
        self._counts = np.array(counts, dtype=float)
        self._lo_edges = np.array(lo_edges)
        self._hi_edges = np.array(hi_edges)
        self._N = len(self._counts)
        self._counts[self._counts == 0.0] = 1e-10  # to avoid log(0)
        self._widths = self._hi_edges - self._lo_edges
        self._ch_points = None

    @property
    def ch_points(self):
        return self._ch_points

    def _f(self, a: float, cp_i: int, cp_f: int) -> float:
        block_counts = np.sum(self._counts[cp_i:cp_f])
        wn = (1 - a * np.arange(cp_f - cp_i)) * self._widths[cp_i:cp_f]
        N_log_wn = self._counts[cp_i:cp_f] * np.log(wn)

        return block_counts * (np.log(block_counts / np.sum(wn)) - 1) + np.sum(N_log_wn)

    def fitness_func(self, cp_i, cp_f, ncp_prior):
        res = minimize_scalar(lambda a: -self._f(a, cp_i, cp_f), bounds=(-2, 1), method='bounded')
        max_a = res.x

        return self._f(max_a, cp_i, cp_f) - ncp_prior

    def run_algorithm(self, ncp_prior):
        best = np.zeros(self._N)
        last = np.zeros(self._N, dtype=int)

        # the base case of an induction, i = 0 
        best[0] = self.fitness_func(0, 1, ncp_prior)
        last[0] = 0

        for i in range(1, self._N):
            A = np.zeros(i + 1)
            for j in range(i + 1):
                fit = self.fitness_func(j, i + 1, ncp_prior)
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

    def t_xx(self, xx, lo_edges, hi_edges, counts_signal):
        T100 = (self._ch_points[1], self._ch_points[-1] - 1)
        total_counts = np.sum(counts_signal[T100[0]:T100[1] + 1])

        # Lower bound
        s = 0
        for f in range(T100[0], T100[1] + 1):
            s += counts_signal[f]
            if s >= 0.5 * (1.0 - xx * 0.01) * total_counts:
                lo_xx = lo_edges[f]
                break

        # Upper bound
        s = 0
        for i in range(T100[1], T100[0] - 1, -1):
            s += counts_signal[i]
            if s >= 0.5 * (1.0 - xx * 0.01) * total_counts:
                hi_xx = hi_edges[i]
                break

        return lo_xx, hi_xx
