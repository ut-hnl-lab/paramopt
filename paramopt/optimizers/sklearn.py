from typing import Any, Callable, Generator, Optional, Tuple, Union

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

from .base import BaseOptimizer
from ..acquisitions.base import BaseAcquisition
from ..structures.parameter import ExplorationSpace
from ..structures.dataset import Dataset


class BayesianOptimizer(BaseOptimizer):
    """sklearnガウス過程回帰モデルベースのベイジアンオプティマイザ.

    Parameters
    ----------
        workdir: 学習履歴を書き出すディレクトリ
        kernel: GPRのカーネル
        acquisition: 獲得関数
            acquisition(mean, std)形式で計算.
        random_seed: 乱数シード
            再現性確保のために指定推奨.
    """
    def __init__(
        self,
        workdir: str,
        exploration_space: 'ExplorationSpace',
        dataset: 'Dataset',
        model: 'GaussianProcessRegressor',
        acquisition: 'BaseAcquisition',
        objective_fn: Optional[Callable] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            workdir, exploration_space, dataset, acquisition, objective_fn)
        self.model = model
        self.acquisition = acquisition
        self.random_seed = random_seed

        if random_seed is not None:
            self.fix_random_state()

    def _fit_to_model(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.model.fit(X, Y)

    def _predict_with_model(self, X: np.ndarray) -> Tuple[np.ndarray, ...]:
        mean, std = self.model.predict(X, return_std=True)
        return (mean, std)

    def fix_random_state(self) -> None:
        np.random.seed(self.random_seed)
        self.model.random_state = self.random_seed
