from itertools import product
import os
from typing import Any, Callable, Generator, Optional, Tuple, Union

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as sk

from .base import BaseOptimizer
from ..acquisitions.base import BaseAcquisition
from .. import plot


class BayesianOptimizer(BaseOptimizer):
    """sklearnガウス過程回帰モデルベースのベイジアンオプティマイザ.

    Parameters
    ----------
        savedir: 学習履歴を書き出すディレクトリ
        kernel: GPRのカーネル
        acqfunc: 獲得関数
            acqfunc(mean, std)形式で計算.
        random_seed: 乱数シード
            再現性確保のために指定推奨.
    """
    def __init__(
        self, savedir: str, kernel: sk.Kernel, acqfunc: BaseAcquisition,
        random_seed: Optional[int] = None,
    ) -> None:
        super().__init__(savedir)
        self.model = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
        self.acqfunc = acqfunc
        if random_seed is not None:
            self._fix_seed(random_seed)

    def add_parameter(self, name: str, space: Union[list, Generator]) -> None:
        super().add_parameter(name, space)
        self.X_combos = np.array(list(product(*self.params.values)))
        self.X_grid_combos = np.array(list(product(*self.params.grids)))

    def next(self) -> Tuple[Any]:
        # 全てのパラメータの組み合わせに対する平均と分散を計算
        mean, std = self.model.predict(self.X_combos, return_std=True)

        # 獲得関数値を基に次のパラメータの組み合わせを選択
        acq = self.acqfunc.compute(mean, std, self.X, self.y)
        next_idx = np.argmax(acq)  # 獲得関数を最大化するパラメータの組み合わせ
        next_X = tuple(self.X_combos[next_idx])
        return next_X

    def plot(
        self, objective_fn: Optional[Callable] = None, overwrite: bool = False,
    ) -> None:
        """学習の経過をグラフ化する.

        Parameters
        ----------
            objective_fn: 目的関数
                真の関数が分かっている場合(=テスト時)に指定すると, 共に描画.
            overwrite: 1つのウィンドウに対してグラフを上書き更新するか否か
                jupyter notebook使用時はFalseを推奨
        """
        mean, std = self.model.predict(self.X_grid_combos, return_std=True)
        acq = self.acqfunc.best_idx(mean, std, self.X, self.y)
        plot.overwrite = overwrite
        plot.plot(
            self.params, self.X, self.y, mean, std, acq, objective_fn, self.y_name)
        plot.savefig(os.path.join(self.savedir, f'plot-{self.labels[-1]}.png'))

    def _fit(self) -> None:
        self.model.fit(self.X, self.y)

    def _fix_seed(self, random_seed) -> None:
        np.random.seed(random_seed)
        self.model.random_state = random_seed
