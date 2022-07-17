from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from .base import BaseGraph
from ..structures.dataset import Dataset
from ..structures.parameter import ExplorationSpace


COLORS = list(TABLEAU_COLORS.keys())
MARKERS = ["o", "s", "^", "D", "v", "*"]


class Transition(BaseGraph):
    """Class for visualizing transitions of process parameters and objective
    core.
    """
    PNG_PREFIX = "trans-"

    def __init__(self) -> None:
        self.fig = None

    def plot(
        self,
        exploration_space: 'ExplorationSpace',
        dataset: 'Dataset',
        *args: Any,
        **kwargs: Any
    ) -> None:
        """Plots the transition of the values of the selected parameters and
        objective score.

        No limitation on the parameter dimension.
        If given, valiable-length arguments are used to setup
        `matplotlib.pyplot.figure`.
        """
        plt.close()

        if exploration_space.dimension != dataset.dimension_X:
            raise ValueError(
                "exploration dimension does not match dataset dimension")

        self.fig = _plot_transition(
            fig=plt.figure(*args, **kwargs),
            X_spaces=exploration_space.spaces,
            X=dataset.X,
            Y=dataset.Y,
            X_names=exploration_space.names,
            Y_names=dataset.Y_names
        )
        self.fig.tight_layout()


def _plot_transition(
    fig: Figure,
    X_spaces: List[np.ndarray],
    X: np.ndarray,
    Y: np.ndarray,
    X_names: List[str],
    Y_names: List[str],
    Y_bounds: Optional[Tuple[int]] = None
) -> Figure:
    """Plot function for X, Y transition."""
    # Axes generation and initial settings
    ax_left = fig.add_subplot()
    ax_right = ax_left.twinx()
    if Y_bounds is not None:
        ax_left.set_ylim(Y_bounds)
    ax_left.set_xlabel("Iteration Number")
    ax_left.set_ylabel("Objective Score")
    ax_right.set_ylabel("Parameter Value (normalized)", labelpad=10)
    ax_left.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax_left.ticklabel_format(style="sci", axis="y", scilimits=(-2, 2))

    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)

    # Evaluation score and parameter value plot
    for i in range(Y.shape[1]):
        mi, ci = int(i%len(MARKERS)), int(i%len(COLORS))
        ax_left.plot(
            Y[:, i], f'-{MARKERS[mi]}', label=Y_names[i], color=COLORS[ci])
    for j in range(X.shape[1]):
        X_vec = X[:, j:j+1]
        scaler = MinMaxScaler().fit(np.atleast_2d(X_spaces[j]).T)
        scaled_X_vec = scaler.transform(X_vec)
        mj, cj = int((i+j+1)%len(MARKERS)), int((i+j+1)%len(COLORS))
        ax_right.plot(
            scaled_X_vec, MARKERS[mj], label=X_names[j], color=COLORS[cj])

    # Additional settings
    hl, ll = ax_left.get_legend_handles_labels()
    hr, lr = ax_right.get_legend_handles_labels()
    ax_left.legend(
        hl+hr, ll+lr, loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=3)

    return fig
