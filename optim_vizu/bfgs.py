from .base import BaseOptimizer
import numpy as np


class BFGS(BaseOptimizer):
    """
    BFGS (Broyden-Fletcher-Goldfarb-Shanno) Verfahren.

    Quasi-Newton Methode: approximiert die Hesse-Matrix iterativ
    statt sie jedes Mal neu zu berechnen.

    Schritt:
        d      = H_inv * grad(x)
        x_new  = x - lr * d
        H_inv wird nach jedem Schritt mit der BFGS-Formel aktualisiert

    Parameter
    ----------
    lr : Schrittweite (default 1.0)
    """

    name = "BFGS"

    def __init__(self, f, bounds, multistart=False, n_starts=8,
                 tol=1e-6, max_iter=1000, lr=1.0):
        super().__init__(f, bounds, multistart, n_starts, tol, max_iter)
        self.lr = lr
        self.H_inv = None

    def run(self, x0=None):
        """H_inv initialisieren und dann normal run() ausfuehren."""
        if self.multistart:
            if x0 is None:
                starts = self._random_starts(self.n_starts)
            else:
                starts = [np.array(s, dtype=float) for s in x0]
            results = []
            for s in starts:
                self.H_inv = np.eye(len(s))
                results.append(self._run_single(s))
            return results
        else:
            if x0 is None:
                x0 = self._random_starts(1)[0]
            x0 = np.array(x0, dtype=float)
            self.H_inv = np.eye(len(x0))
            return self._run_single(x0)

    def gradient(self, x, eps=1e-5):
        """Numerischer Gradient mit zentralen Differenzen."""
        grad = np.zeros_like(x)
        for i in range(len(x)):
            e = np.zeros_like(x)
            e[i] = eps
            grad[i] = (self.f(x + e) - self.f(x - e)) / (2 * eps)
        return grad

    def step(self, x):
        """
        Ein BFGS Schritt:
        1) Abstiegsrichtung mit aktueller H_inv berechnen
        2) Schritt machen
        3) H_inv fuer naechsten Schritt aktualisieren
        """
        grad = self.gradient(x)

        # Abstiegsrichtung
        d = self.H_inv @ grad
        x_new = x - self.lr * d

        # H_inv aktualisieren (BFGS Formel)
        grad_new = self.gradient(x_new)
        s = x_new - x
        y = grad_new - grad
        sy = s @ y

        if abs(sy) > 1e-10:
            n = len(x)
            rho = 1.0 / sy
            A = np.eye(n) - rho * np.outer(s, y)
            B = np.eye(n) - rho * np.outer(y, s)
            self.H_inv = A @ self.H_inv @ B + rho * np.outer(s, s)

        return x_new
