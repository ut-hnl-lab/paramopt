from typing import Any, Callable, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
from matplotlib.figure import Figure
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from paramopt.structures.dataset import Dataset

from .base import BaseGraph
from ..structures.parameter import ExplorationSpace


COLORS = list(TABLEAU_COLORS.keys())
MARKERS = ["o", "s", "^", "D", "v", "*"]


class Transition(BaseGraph):

    PNG_PREFIX = "trans-"

    def __init__(self) -> None:
        self.fig = None

    def plot(
        self,
        exploration_space: "ExplorationSpace",
        dataset: "Dataset",
        *args: Any,
        **kwargs: Any
    ) -> None:
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
    scaler = MinMaxScaler()
    ax_left = fig.add_subplot()
    ax_right = ax_left.twinx()

    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)


    for i in range(Y.shape[1]):
        ax_left.plot(Y, f'-{MARKERS[i]}', label=Y_names[i], color=COLORS[i])
    for j in range(X.shape[1]):
        X_space = X_spaces[j].reshape(-1, 1)
        fitted_scaler = scaler.fit(X_space)
        scaled_X = fitted_scaler.transform(X[:, j:j+1])
        ax_right.plot(
            scaled_X, MARKERS[i+j+1], label=X_names[j], color=COLORS[i+j+1])

    ax_left.set_xlabel("Iteration Number")
    ax_left.set_ylabel("Objective Score")
    ax_right.set_ylabel("Parameter Value (normalized)")
    hl, ll = ax_left.get_legend_handles_labels()
    hr, lr = ax_right.get_legend_handles_labels()
    ax_left.legend(
        hl+hr, ll+lr, loc='lower center', bbox_to_anchor=(.5, 0.97), ncol=3,
        frameon=False)
    if Y_bounds is not None:
        ax_left.set_ylim(Y_bounds)

    return fig
