"""グラフ描写関数群"""

from typing import Callable, List, Optional
from matplotlib import cm, gridspec, pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from .parameter import ProcessParameter


fig = None
overwrite = False


def plot(pp: ProcessParameter, X: np.ndarray, y: np.ndarray,
    mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None,
    acq: Optional[np.ndarray] = None, objective_fn: Optional[Callable] = None,
    y_name: str = 'y'
) -> None:
    ndim = pp.ndim

    if ndim == 1:
        plot_1d(
            pp.grids[0], X, y, mean, std, acq, objective_fn, pp.names[0], y_name)
    elif ndim == 2:
        plot_2d(pp.grids, X, y, mean, acq, objective_fn, pp.names, y_name)
    else:
        raise NotImplementedError(f'{ndim}D plot not supported')


def plot_1d(
    X_grid: np.ndarray, X: np.ndarray, y: np.ndarray,
    mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None,
    acq: Optional[np.ndarray] = None, objective_fn: Optional[Callable] = None,
    x_name: str = 'x', y_name: str = 'y'
) -> None:
    global fig

    _refresh()
    if acq is not None:
        spec = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[3, 1])
        ax2 = fig.add_subplot(spec[1])
        ax2.plot(X_grid, acq, 'r-')
        ax2.set_xlabel(x_name)
        ax2.set_ylabel('Acquisition')
        ax = fig.add_subplot(spec[0])
    else:
        ax = fig.add_subplot(111)

    if objective_fn is not None:
        ax.plot(
            X_grid, objective_fn(X_grid), 'k:', alpha=.5,
            label='Objective fn')

    if mean is not None:
        ax.plot(X_grid, mean, 'b-', label='Prediction')
        if std is not None:
            ax.fill(
                np.concatenate([X_grid, X_grid[::-1]]),
                np.concatenate([mean -1.96*std, (mean + 1.96*std)[::-1]]),
                'p-', alpha=.5, label='95% CI')

    ax.plot(X, y, 'k.', label='Observations')
    ax.plot(X[-1], y[-1], 'r*', markersize=10)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.legend()
    _show()


def plot_2d(
    X_grid: List[np.ndarray], X: np.ndarray, y: np.ndarray,
    mean: Optional[np.ndarray] = None,
    acq: Optional[np.ndarray] = None, objective_fn: Optional[Callable] = None,
    x_name: List[str] = ['x1', 'x2'], y_name: str = 'y'
) -> None:
    global fig

    _refresh()
    X_grid1, X_grid2 = X_grid
    Xmesh1, Xmesh2 = np.meshgrid(X_grid1, X_grid2)
    ax = fig.add_subplot(111, projection='3d')

    if objective_fn is not None:
        ax.plot_wireframe(
            Xmesh1, Xmesh2, objective_fn(Xmesh1, Xmesh2),
            color='k', alpha=0.5, linewidth=0.5, label='Objective fn')

    if mean is not None:
        mean = mean.reshape(X_grid1.shape[0], X_grid2.shape[0])
        ax.plot_wireframe(
            Xmesh1, Xmesh2, mean.T, color='b', alpha=0.6, linewidth=0.5,
            label='Prediction')

    ax.scatter(
        X[:-1, 0], X[:-1, 1], y[:-1], c='black',
        label='Observations')
    ax.scatter(
        X[-1, 0], X[-1, 1], y[-1], c='red', marker='*',
        s=50)

    if acq is not None:
        acq = acq.reshape(X_grid1.shape[0], X_grid2.shape[0])
        contf = ax.contourf(
            Xmesh1, Xmesh2, acq.T, zdir='z', offset=ax.get_zlim()[0],
            cmap=cm.jet, levels=100)
        fig.colorbar(contf, pad=0.08, shrink=0.6, label='Acquisition')

    ax.set_xlabel(x_name[0])
    ax.set_ylabel(x_name[1])
    ax.set_zlabel(y_name)
    ax.legend()
    _show()


def savefig(path: str) -> None:
    global fig
    fig.savefig(path)


def _refresh():
    global fig
    if overwrite:
        if fig is None:
            fig = plt.figure()
        else:
            fig.clear()
    else:
        plt.close()
        fig = plt.figure()


def _show() -> None:
    global fig
    plt.tight_layout()
    if overwrite:
        plt.pause(0.1)
    else:
        plt.show(block=False)
