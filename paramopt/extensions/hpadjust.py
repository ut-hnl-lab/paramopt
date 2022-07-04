from dataclasses import asdict, dataclass, field
from inspect import signature
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from scipy.stats import mode

from itertools import product

from paramopt.structures.parameter import ExplorationSpace


def fitting_score(mean: np.ndarray) -> float:
    modal_value = mode(np.round(mean.flatten(), 1))
    score = 1 - modal_value.count[0] / mean.size
    return score


class AutoHPGPR:

    MIN_FITTING_SCORE = 0.8

    def __init__(
        self,
        workdir: Union[Path, str],
        exploration_space: 'ExplorationSpace',
        gpr_generator: Callable,
        **hparams: Any
    ) -> None:
        if signature(gpr_generator).parameters.keys() != hparams.keys():
            raise ValueError("gpr_generator arguments does not match **hparams")

        self.workdir = workdir
        self.space_grid = exploration_space.grid_conbinations()
        self.gpr_generator = gpr_generator
        self.hparams = hparams
        self.best_gpr = None

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
            if score >= self.MIN_FITTING_SCORE:
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
