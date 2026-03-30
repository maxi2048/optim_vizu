from .base import BaseOptimizer
import numpy as np


class GradientDescent(BaseOptimizer):
    """
    Gradientenabstieg.
    Schritt: x_new = x - lr * grad(x)

    Parameter
    ----------
    lr : Lernrate (Schrittweite in Gradientenrichtung)
    """

    name = "GradientDescent"

    def __init__(self, f, bounds, multistart=False, n_starts=8,
                 tol=1e-6, max_iter=1000, lr=0.01):
        super().__init__(f, bounds, multistart, n_starts, tol, max_iter)
        self.lr = lr

    def gradient(self, x, eps=1e-5):
        """Numerischer Gradient mit zentralen Differenzen."""
        grad = np.zeros_like(x)
        for i in range(len(x)):
            e = np.zeros_like(x)
            e[i] = eps
            grad[i] = (self.f(x + e) - self.f(x - e)) / (2 * eps)
        return grad

    def step(self, x):
        grad = self.gradient(x)
        return x - self.lr * grad
