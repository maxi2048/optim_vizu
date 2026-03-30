import itertools
import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    def __init__(self, f, bounds):
        self.f = f
        self.bounds = bounds
        self.ndim = len(bounds)

    def plot(self, runs):
        """
        runs: Liste von Tupeln (path, values, label)
        path:   np.array shape (n_steps, ndim)
        values: np.array shape (n_steps,)
        label:  string z.B. "Newton"
        """
        if self.ndim == 1:
            self._plot_1d(runs)
        elif self.ndim == 2:
            self._plot_2d(runs)
        elif self.ndim == 3:
            self._plot_3d(runs)
        else:
            self._plot_pairplot(runs)

    def _plot_1d(self, runs):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        ax_f, ax_conv = axes

        lo, hi = self.bounds[0]
        xs = np.linspace(lo, hi, 300)
        ys = [self.f(np.array([x])) for x in xs]
        ax_f.plot(xs, ys, color="gray", lw=2, label="f(x)")
        ax_f.set_xlabel("x")
        ax_f.set_ylabel("f(x)")
        ax_f.set_title("Funktion & Pfad")

        colors = [plt.cm.tab10(i / max(len(runs), 1)) for i in range(len(runs))]
        for i, (path, values, label) in enumerate(runs):
            color = colors[i]
            px = path[:, 0]

            ax_f.scatter(px, values, color=color, s=30, zorder=3)
            ax_f.plot(px, values, color=color, lw=1, alpha=0.5)
            ax_f.scatter(px[0], values[0], color=color, s=100, marker="^", zorder=4, label=f"{label} Start")
            ax_f.scatter(px[-1], values[-1], color=color, s=100, marker="*", zorder=4, label=f"{label} Ende")

            iterations = list(range(len(values)))
            ax_conv.plot(iterations, values, color=color, lw=2, label=label)

        ax_f.legend()
        ax_conv.set_xlabel("Iteration")
        ax_conv.set_ylabel("f(x)")
        ax_conv.set_title("Konvergenz")

        max_len = max(len(values) for _, values, _ in runs)
        if max_len <= 15:
            ax_conv.set_xticks(list(range(max_len)))
        else:
            step = max(1, max_len // 10)
            ticks = list(range(0, max_len, step))
            if ticks[-1] != max_len - 1:
                ticks.append(max_len - 1)
            ax_conv.set_xticks(ticks)

        ax_conv.set_xlim(0, max_len - 1 if max_len > 0 else 1)
        ax_conv.legend()
        ax_conv.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def _plot_2d(self, runs):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax_contour, ax_conv = axes

        lo0, hi0 = self.bounds[0]
        lo1, hi1 = self.bounds[1]
        xs = np.linspace(lo0, hi0, 300)
        ys = np.linspace(lo1, hi1, 300)
        X, Y = np.meshgrid(xs, ys)
        Z = np.vectorize(lambda u, v: self.f(np.array([u, v])))(X, Y)

        ax_contour.contourf(X, Y, Z, levels=50, cmap="viridis", alpha=0.7)
        ax_contour.contour(X, Y, Z, levels=50, colors="white", linewidths=0.3, alpha=0.4)
        ax_contour.set_xlabel("x0")
        ax_contour.set_ylabel("x1")
        ax_contour.set_title("Konturplot & Pfad")

        colors = [plt.cm.tab10(i / max(len(runs), 1)) for i in range(len(runs))]
        max_len = 0

        for i, (path, values, label) in enumerate(runs):
            color = colors[i]
            px = path[:, 0]
            py = path[:, 1]

            ax_contour.plot(px, py, color=color, lw=1.5, alpha=0.8)
            ax_contour.scatter(px, py, color=color, s=15, alpha=0.6)
            ax_contour.scatter(px[0], py[0], color=color, s=120, marker="^", zorder=5, label=f"{label} Start")
            ax_contour.scatter(px[-1], py[-1], color=color, s=150, marker="*", zorder=5, label=f"{label} Ende")

            iterations = list(range(len(values)))
            ax_conv.plot(iterations, values, color=color, lw=2, label=label)
            max_len = max(max_len, len(values))

        ax_contour.legend()

        ax_conv.set_xlabel("Iteration")
        ax_conv.set_ylabel("f(x)")
        ax_conv.set_title("Konvergenz")

        if max_len <= 15:
            ax_conv.set_xticks(list(range(max_len)))
        else:
            step = max(1, max_len // 10)
            ticks = list(range(0, max_len, step))
            if ticks[-1] != max_len - 1:
                ticks.append(max_len - 1)
            ax_conv.set_xticks(ticks)

        ax_conv.set_xlim(0, max_len - 1 if max_len > 0 else 1)
        ax_conv.legend()
        ax_conv.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def _plot_3d(self, runs, resolution=15):
        """3D Scatter der Funktion plus Optimierungspfade für 3D-Probleme."""
        (lo0, hi0), (lo1, hi1), (lo2, hi2) = self.bounds

        ax0 = np.linspace(lo0, hi0, resolution)
        ax1 = np.linspace(lo1, hi1, resolution)
        ax2 = np.linspace(lo2, hi2, resolution)
        X0, X1, X2 = np.meshgrid(ax0, ax1, ax2)
        X0f = X0.flatten()
        X1f = X1.flatten()
        X2f = X2.flatten()

        Z = np.array([self.f(np.array([x0, x1, x2])) for x0, x1, x2 in zip(X0f, X1f, X2f)])

        fig = plt.figure(figsize=(14, 6))
        ax_surface = fig.add_subplot(1, 2, 1, projection="3d")
        ax_conv = fig.add_subplot(1, 2, 2)

        sc = ax_surface.scatter(X0f, X1f, X2f, c=Z, cmap="viridis", alpha=0.15, s=10)
        plt.colorbar(sc, ax=ax_surface, label="f(x)")

        colors = [plt.cm.tab10(i / max(len(runs), 1)) for i in range(len(runs))]
        max_len = 0

        for i, (path, values, label) in enumerate(runs):
            color = colors[i]

            ax_surface.plot(path[:, 0], path[:, 1], path[:, 2], color=color, lw=2, label=label)
            ax_surface.scatter(path[0, 0], path[0, 1], path[0, 2], color=color, s=80, marker="^")
            ax_surface.scatter(path[-1, 0], path[-1, 1], path[-1, 2], color=color, s=150, marker="*")

            iterations = list(range(len(values)))
            ax_conv.plot(iterations, values, color=color, lw=2, label=label)
            max_len = max(max_len, len(values))

        ax_surface.set_xlabel("x0")
        ax_surface.set_ylabel("x1")
        ax_surface.set_zlabel("x2")
        ax_surface.set_title("3D Funktion — Farbe = f(x)")
        ax_surface.legend()

        ax_conv.set_xlabel("Iteration")
        ax_conv.set_ylabel("f(x)")
        ax_conv.set_title("Konvergenz")

        if max_len <= 15:
            ax_conv.set_xticks(list(range(max_len)))
        else:
            step = max(1, max_len // 10)
            ticks = list(range(0, max_len, step))
            if ticks[-1] != max_len - 1:
                ticks.append(max_len - 1)
            ax_conv.set_xticks(ticks)

        ax_conv.set_xlim(0, max_len - 1 if max_len > 0 else 1)
        ax_conv.grid(True, alpha=0.3)
        ax_conv.legend()

        plt.tight_layout()
        plt.show()

    def _plot_contour(self, runs):
        """Nur der Konturplot ohne Konvergenzplot."""
        lo0, hi0 = self.bounds[0]
        lo1, hi1 = self.bounds[1]
        xs = np.linspace(lo0, hi0, 300)
        ys = np.linspace(lo1, hi1, 300)
        X, Y = np.meshgrid(xs, ys)
        Z = np.vectorize(lambda u, v: self.f(np.array([u, v])))(X, Y)

        fig, ax = plt.subplots(figsize=(7, 6))
        contour = ax.contourf(X, Y, Z, levels=50, cmap="viridis", alpha=0.7)
        ax.contour(X, Y, Z, levels=50, colors="white", linewidths=0.3, alpha=0.4)
        plt.colorbar(contour, ax=ax, label="f(x, y)")

        colors = [plt.cm.tab10(i / max(len(runs), 1)) for i in range(len(runs))]
        for i, (path, values, label) in enumerate(runs):
            color = colors[i]
            px = path[:, 0]
            py = path[:, 1]
            ax.plot(px, py, color=color, lw=1.5, alpha=0.8)
            ax.scatter(px[0], py[0], color=color, s=120, marker="^", zorder=5, label=f"{label} Start")
            ax.scatter(px[-1], py[-1], color=color, s=150, marker="*", zorder=5, label=f"{label} Ende")

        ax.set_xlabel("x0")
        ax.set_ylabel("x1")
        ax.set_title("Konturplot & Pfad")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.show()

    def _plot_convergence(self, runs):
        """Nur der Konvergenzplot."""
        fig, ax = plt.subplots(figsize=(7, 4))

        colors = [plt.cm.tab10(i / max(len(runs), 1)) for i in range(len(runs))]
        max_len = 0

        for i, (path, values, label) in enumerate(runs):
            iterations = list(range(len(values)))
            ax.plot(iterations, values, color=colors[i], lw=2, label=label)
            max_len = max(max_len, len(values))

        ax.set_xlabel("Iteration")
        ax.set_ylabel("f(x)")
        ax.set_title("Konvergenzverlauf")

        if max_len <= 15:
            ax.set_xticks(list(range(max_len)))
        else:
            step = max(1, max_len // 10)
            ticks = list(range(0, max_len, step))
            if ticks[-1] != max_len - 1:
                ticks.append(max_len - 1)
            ax.set_xticks(ticks)

        ax.set_xlim(0, max_len - 1 if max_len > 0 else 1)

        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def _plot_pairplot(self, runs):
        """Pairplot-artige 2D-Projektionen für alle Dimensionspaare."""
        pairs = list(itertools.combinations(range(self.ndim), 2))
        colors = [plt.cm.tab10(i / max(len(runs), 1)) for i in range(len(runs))]

        for dim_i, dim_j in pairs:
            plt.figure(figsize=(5.5, 4.5))
            for run_idx, (path, _values, label) in enumerate(runs):
                color = colors[run_idx]
                px = path[:, dim_i]
                py = path[:, dim_j]

                plt.plot(px, py, lw=1.5, color=color, alpha=0.9, label=label)
                plt.scatter(px, py, color=color, s=18, alpha=0.7)
                plt.scatter(px[0], py[0], marker="^", color=color, s=100)
                plt.scatter(px[-1], py[-1], marker="*", color=color, s=120)

            plt.xlabel(f"x{dim_i}")
            plt.ylabel(f"x{dim_j}")
            plt.title(f"Projektion: x{dim_i} vs x{dim_j}")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()