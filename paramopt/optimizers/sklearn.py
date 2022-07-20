from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

from .base import BaseOptimizer
from ..acquisitions.base import BaseAcquisition
from ..structures.parameter import ExplorationSpace
from ..structures.dataset import Dataset


class BayesianOptimizer(BaseOptimizer):
    """Bayesian optimizer based on
    `sklearn.gaussian_process.GaussianProcessRegressor`.

    This class read and write raw data via `Dataset` class, and explore
    parameter space defined by `ExplorationSpace` class, using sklearn gpr model
    and acquisition class that inherits `BaseAcquisition` class.

    Parameters
    ----------
    workdir: pathlib.Path or str
        Working directory where csv and png files are output.
    exploration_space: ExplorationSpace
        Definition of the exploration space.
    dataset: Dataset
        Dataset used for training input and prediction output.
    model: sklearn.gaussian_process.GaussianProcessRegressor
        scikit-learn gpr model.
    acquisition: Acquisition
        Acquisition function implemented as a acquisition class.
    objective_fn: Callable
        Objective function that represents true distribution.
    random_seed: int, optional
        Random seed.
    """
    def __init__(
        self,
        workdir: Union[Path, str],
        exploration_space: 'ExplorationSpace',
        dataset: 'Dataset',
        model: 'GaussianProcessRegressor',
        acquisition: 'BaseAcquisition',
        objective_fn: Optional[Callable] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        self.model = model
        self.acquisition = acquisition
        self.random_seed = random_seed
        if random_seed is not None:
            self._fix_random_state()

        super().__init__(
            workdir, exploration_space, dataset, acquisition, objective_fn)

    def _fit_to_model(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.model.fit(X, Y)

    def _predict_with_model(self, X: np.ndarray) -> Tuple[np.ndarray, ...]:
        mean, std = self.model.predict(X, return_std=True)
        return (mean, std)

    def _fix_random_state(self) -> None:
        np.random.seed(self.random_seed)
        self.model.random_state = self.random_seed
