from inspect import signature
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union
import warnings

import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.gaussian_process import GaussianProcessRegressor

from ..structures.parameter import ExplorationSpace


def fitting_score(mean: np.ndarray) -> float:
    """Evaluates how reasonable the given prediction is."""
    modal_value = mode(np.round(mean.flatten(), 1))
    score = 1 - modal_value.count[0] / mean.size
    return score


class AutoHPGPR:
    """A GPR model that can automatically adjust hyperparameter for each fitting.

    Parameters
    ----------
    workdir : pathlib.Path or str
        Working directory where csv files are output.
    exploration_space : paramopt.ExplorationSpace
        An exploration space definition.
    gpr_generator : Callable
        A function that generates gpr model. This should receive hyperparameters.
    stop_fitting_score: float, default is `0.9`
        Fitting score to stop adjusting if exceeded.
    **hparams: Any
        Hyperparameter spaces. When `gpr_generator` takes arguments named `a`
        and `b`, this keyword argument takes list of their possible values like
        `a=[1,2,3], b=[4,5]`

    Raises
    ------
    ValueError
        Raises when the number of `gpr_generator` arguments does not match the
        number of hyperparameter spaces.
    """

    def __init__(
        self,
        workdir: Union[Path, str],
        exploration_space: 'ExplorationSpace',
        gpr_generator: Callable,
        stop_fitting_score: Optional[float] = 0.9,
        **hparams: Any
    ) -> None:
        if signature(gpr_generator).parameters.keys() != hparams.keys():
            raise ValueError("gpr_generator arguments does not match hparams")

        self.workdir = workdir
        self.space_grid = exploration_space.grid_combinations()
        self.gpr_generator = gpr_generator
        self.hparams = hparams
        self.stop_fitting_score = stop_fitting_score
        self.best_gpr = gpr_generator(
            **{key: vals[0] for key, vals in hparams.items()})

        self.hp_history = HPHistory(*hparams.keys(), "score")

    def fit(self, X: Any, y: Any) -> None:
        best_score = -1
        for hps in self._iter_hp_combinations():
            gpr = self.gpr_generator(**hps)
            self._fit_to_gpr(gpr, X, y)
            mean = self._predict_with_gpr(gpr, self.space_grid)
            score = fitting_score(mean)
            if score > best_score:
                best_score = score
                best_gpr = gpr
                best_hps = hps
            if score >= self.stop_fitting_score:
                break
        else:
            warnings.warn(f"Hyper parameter adjustment failed.", UserWarning)

        self.best_gpr = best_gpr
        self.hp_history.append(**best_hps, score=best_score)

    def predict(
        self, X: Any, return_std: bool = False, return_cov: bool = False
    ) -> Any:
        self.hp_history.to_csv(self.workdir)
        return self._predict_with_gpr(self.best_gpr, X, return_std, return_cov)

    def _fit_to_gpr(self, gpr: GaussianProcessRegressor, X: Any, y: Any) -> Any:
        gpr.fit(X, y)

    def _predict_with_gpr(
        self, gpr: GaussianProcessRegressor, X: Any, return_std: bool = False,
        return_cov: bool = False
    ) -> Any:
        return gpr.predict(X, return_std, return_cov)

    def _iter_hp_combinations(self) -> Dict[str, Any]:
        keys, values = self.hparams.keys(), self.hparams.values()
        for combo in product(*values):
            yield {key: val for key, val in zip(keys, combo)}


class HPHistory:
    """Accumulates and exports the history of given values with given names."""
    EXPORT_NAME = "hp_history.csv"

    def __init__(self, *names) -> None:
        if len(names) != len(set(names)):
            raise ValueError("duplicated name")

        self.values = {key: [] for key in names}

    def append(self, **kwargs):
        for key, val in kwargs.items():
            self.values[key].append(val)

    def to_csv(self, directory: Union[Path, str]) -> None:
        df = pd.DataFrame(self.values)

        directory_ = Path(directory)
        directory_.mkdir(exist_ok=True, parents=True)
        df.to_csv(directory_/self.EXPORT_NAME, index=False)
