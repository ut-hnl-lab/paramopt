from typing import Any, Callable, List, Optional, Tuple, Union
from warnings import simplefilter

from matplotlib import cm, gridspec, pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter
import numpy as np

from .base import BaseGraph
from ..structures.dataset import Dataset
from ..structures.parameter import ExplorationSpace


simplefilter('ignore')


class Distribution(BaseGraph):
    """Class for visualizing GPR fitting process."""
    PNG_PREFIX = "dist-"

    def __init__(
        self,
        exploration_space: 'ExplorationSpace',
        objective_fn: Optional[Callable] = None,
        acquisition_name: Optional[str] = None
    ) -> None:
        super().__init__()
        self.exploration_space = exploration_space
        self.objective_fn = objective_fn
        self.acquisition_name = acquisition_name

    def plot(
        self,
        dataset: 'Dataset',
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        acquisition: Optional[np.ndarray] = None,
        next_X: Optional[Union[Tuple[Any], Any]] = None,
        *args: Any,
        **kwargs: Any
    ) -> None:
        """Plots the distribution of the dataset, the predicted mean and
        standard deviation, and the acquisition.

        This supports up to 2D space.
        If given, valiable-length arguments is used to setup
        `matplotlib.pyplot.figure`.
        """
        plt.close()

        if self.exploration_space.dimension != dataset.dimension_X:
            raise ValueError(
                "exploration dimension does not match dataset dimension")

        ndim = self.exploration_space.dimension
        if ndim == 1:
            self.fig = _plot_process_1d(
                fig=plt.figure(*args, **kwargs),
                X=dataset.X,
                Y=dataset.Y,
                X_grid=self.exploration_space.grid_spaces[0],
                mean=mean,
                std=std,
                acquisition=acquisition,
                next_X=next_X,
                objective_fn=self.objective_fn,
                x_label=self.exploration_space.names[0],
                y_label=dataset.Y_names[0],
                acq_label=self.acquisition_name)
        elif ndim == 2:
            self.fig = _plot_process_2d(
                fig=plt.figure(*args, **kwargs),
                X=dataset.X,
                Y=dataset.Y,
                X_grids=self.exploration_space.grid_spaces,
                mean=mean,
                acquisition=acquisition,
                next_X=next_X,
                objective_fn=self.objective_fn,
                X_labels=self.exploration_space.names,
                z_label=dataset.Y_names[0],
                acq_label=self.acquisition_name)
        else:
            raise NotImplementedError(f"{ndim}D plot is not supported")
        self.fig.tight_layout()


def _plot_process_1d(
    fig: Figure,
    X: np.ndarray,
    Y: np.ndarray,
    X_grid: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    acquisition: Optional[np.ndarray] = None,
    next_X: Optional[Any] = None,
    objective_fn: Optional[Callable] = None,
    x_label: str = "x",
    y_label: str = "y",
    acq_label: str = "y"
) -> Figure:
    """Plot function for 1D parameter space."""
    # Axes generation and initial settings
    if acquisition is not None:
        gs = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[5, 2])
        gs.update(hspace=0.25)
        ax_upper = fig.add_subplot(gs[0])
        ax_lower = fig.add_subplot(gs[1], sharex=ax_upper)
        ax_lower.set_ymargin(0.4)
        ax_lower.set_xlim(X_grid[0], X_grid[-1])
        ax_lower.set_xlabel(x_label)
        ax_lower.set_ylabel(acq_label)
        ax_lower.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax_lower.ticklabel_format(style="sci", axis="y", scilimits=(-2, 2))

    else:
        ax_upper = fig.add_subplot()
        ax_lower = None
        ax_upper.set_xlabel(x_label)
    ax_upper.set_ymargin(0.4)
    ax_upper.set_xlim(X_grid[0], X_grid[-1])
    ax_upper.set_ylabel(y_label)
    ax_upper.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax_upper.ticklabel_format(style="sci", axis="y", scilimits=(-2, 2))

    # Observation
    ax_upper.plot(X[:-1], Y[:-1], 'ko', label="Observation", zorder=3)
    ax_upper.plot(X[-1], Y[-1], 'ro', zorder=3)

    # Posterior mean
    if mean is not None:
        ax_upper.plot(X_grid, mean, 'b-', label="Mean")

    # Posterior uncertainty
    if std is not None:
        ax_upper.fill(
            np.concatenate([X_grid, X_grid[::-1]]),
            np.concatenate([mean -1.96*std, (mean + 1.96*std)[::-1]]),
            "p-", alpha=.5, label="95% confidence interval")

    # Objective function
    if objective_fn is not None:
        ax_upper.plot(
            X_grid, objective_fn(X_grid), 'k:', alpha=.5,
            label="Objective function")

    # Acquisition function
    if acquisition is not None:
        ax_lower.plot(X_grid, acquisition, 'g-', label="Acquisition function")

    # Next location
    if next_X is not None:
        ax_upper.axvline(next_X, color="red", linewidth=0.8)
        if ax_lower is not None:
            ax_lower.axvline(next_X, color="red", linewidth=0.8,
            label="Acquisition max")

    # Additional axes settings
    ax_upper.legend(loc='lower left')
    if ax_lower is not None:
        ax_lower.legend(loc='lower left')

    return fig


def _plot_process_2d(
    fig: Figure,
    X: np.ndarray,
    Y: np.ndarray,
    X_grids: List[np.ndarray],
    mean: Optional[np.ndarray] = None,
    acquisition: Optional[np.ndarray] = None,
    next_X: Optional[Tuple[Any]] = None,
    objective_fn: Optional[Callable] = None,
    X_labels: List[str] = ["x1", "x2"],
    z_label: str = "y",
    acq_label: Optional[str] = None
) -> Figure:
    """Plot function for 2D parameter space."""
    Xmeshes = np.meshgrid(X_grids[0], X_grids[1])

    # Axes generation and initial settings
    ax = fig.add_subplot(projection="3d")
    ax.set_zmargin(0.4)
    ax.set_xlim(X_grids[0][0], X_grids[0][-1])
    ax.set_ylim(X_grids[1][0], X_grids[1][-1])
    ax.set_xlabel(X_labels[0])
    ax.set_ylabel(X_labels[1])
    ax.set_zlabel(z_label)
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=True))
    ax.ticklabel_format(style="sci", axis="z", scilimits=(-2, 2))

    # Observation plot
    ax.scatter(X[:-1, 0], X[:-1, 1], Y[:-1], color="black", label="Observation")
    ax.scatter(X[-1, 0], X[-1, 1], Y[-1], color="red")
    z_from, z_to = ax.get_zlim()
    for i in range(X.shape[0]-1):
        ax.plot(
            [X[i, 0]]*2, [X[i, 1]]*2, [z_from, Y[i]], color='black',
            linewidth=0.8, linestyle='--', zorder=99)
    ax.plot(
        [X[-1, 0]]*2, [X[-1, 1]]*2, [z_from, Y[-1]], color='red', linestyle='--',
        linewidth=0.8, zorder=99)

    # Posterior mean plot
    if mean is not None:
        mean = mean.reshape(X_grids[0].shape[0], X_grids[1].shape[0])
        ax.plot_wireframe(
            Xmeshes[0], Xmeshes[1], mean.T, color="blue", alpha=0.5,
            linewidth=0.5, label="Mean")

    # Objective function plot
    if objective_fn is not None:
        ax.plot_wireframe(
            Xmeshes[0], Xmeshes[1], objective_fn(Xmeshes[0], Xmeshes[1]),
            color="black", alpha=0.5, linewidth=0.5, label="Objective function")

    # Acquisition function plot
    if acquisition is not None:
        acquisition = acquisition.reshape(
            X_grids[0].shape[0], X_grids[1].shape[0])
        contf = ax.contourf(
            Xmeshes[0], Xmeshes[1], acquisition.T, zdir="z",
            offset=z_from, levels=100, alpha=0.6)
        cb = fig.colorbar(
            contf, pad=0.11, shrink=0.7,
            label="Acquisition function"
                 + (f" ({acq_label})" if acq_label is not None else ""))
        cb.formatter.set_powerlimits((0, 0))
        ax.set_zlim(z_from, z_to)

    # Next location plot
    if next_X is not None:
        ax.plot(
            [next_X[0]]*2, [next_X[1]]*2, [z_from, z_to], color='red', linestyle='-',
            linewidth=0.8, zorder=99, label="Acquisition max")

    # Additional axes settings
    leg = ax.legend(loc='upper right')
    for line in leg.get_lines():
        line.set_linewidth(1.5)
        line.set_alpha(1.0)

    return fig
