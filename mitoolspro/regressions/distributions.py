from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import beta


class GeneralizedBetaDistribution:
    def __init__(self, alpha, beta, a, b):
        self.alpha = alpha
        self.beta = beta
        self.a = a
        self.b = b
        self._beta_constant = beta(alpha, beta) * (b - a) ** (alpha + beta - 1)

    @lru_cache(maxsize=1000)
    def pdf(self, x):
        if not (self.a <= x <= self.b):
            return 0.0
        numerator = (x - self.a) ** (self.alpha - 1) * (self.b - x) ** (self.beta - 1)
        return numerator / self._beta_constant

    def pdf_vectorized(self, x):
        x = np.asarray(x)
        mask = (x >= self.a) & (x <= self.b)
        result = np.zeros_like(x, dtype=float)
        result[mask] = (
            (x[mask] - self.a) ** (self.alpha - 1)
            * (self.b - x[mask]) ** (self.beta - 1)
            / self._beta_constant
        )
        return result

    def plot_pdf(self):
        x = np.linspace(self.a, self.b, 1000)
        y = self.pdf_vectorized(x)
        plt.figure(figsize=(8, 4))
        plt.plot(x, y, label=f"Beta({self.alpha}, {self.beta}) on [{self.a}, {self.b}]")
        plt.xlabel("x")
        plt.ylabel("Probability Density")
        plt.title("Generalized Beta Distribution")
        plt.legend()
        plt.grid()
        plt.show()
