from itertools import product
import os
from typing import Any, Generator, Literal, Optional, Tuple, Union

import GPyOpt
import numpy as np

from .base import BaseOptimizer
from .. import plot


class BayesianOptimizer(BaseOptimizer):
    """GPyOptガウス過程回帰モデルベースのベイジアンオプティマイザ.

    Parameters
    ----------
        savedir: 学習履歴を書き出すディレクトリ
        acqfunc: 獲得関数の種類
        random_seed: 乱数シード
            再現性確保のために指定推奨.
    """
    def __init__(
        self, savedir: str, acqfunc: Literal['EI', 'EI_MCMC', 'MPI', 'MPI_MCMC',
        'LCB', 'LCB_MCMC'], random_seed: Optional[int] = None, **kwargs: Any
    ) -> None:
        super().__init__(savedir)
        self.acqfunc = acqfunc
        self.kwargs = kwargs
        self.model = None
        self.domain = []
        if random_seed is not None:
            self._fix_seed(random_seed)

    def add_parameter(self, name: str, values: Union[list, Generator]) -> None:
        super().add_parameter(name, values)
        self.domain.append(
            {'name': name, 'type': 'discrete', 'domain': tuple(values)})
        self.X_grid_combos = np.array(list(product(*self.params.grids)))

    def next(self) -> Tuple[Any]:
        suggested_locations = self.model._compute_next_evaluations()
        next_X = tuple(suggested_locations[0])
        return next_X

    def graph(
        self, overwrite: bool = False, gpystyle: bool = False, **kwargs: Any
    ) -> None:
        """学習の経過をグラフ化する.

        Parameters
        ----------
            overwrite: 1つのウィンドウに対してグラフを上書き更新するか否か
                jupyter notebook使用時はFalseを推奨
            gpystyle: GPyOpt固有のグラフ描写関数を用いるか否か
        """
        if gpystyle:
            self.model.plot_acquisition(
                filename=os.path.join(self.savedir, f'{self.tags[-1]}.png'),
                label_x = self.params.names[0],
                label_y = self.y_name if len(self.params.names) < 2 else self.params.names[1])
        else:
            mean, std = self.model.model.model.predict(self.X_grid_combos)
            acq = -self.model.acqfunc.acquisition_function(self.X_grid_combos)
            plot.overwrite = overwrite
            plot.plot(
                self.params, self.model.model.model.X, self.model.model.model.Y,
                mean, std, acq, objective_fn=None)
            plot.savefig(os.path.join(self.savedir, f'plot-{self.tags[-1]}.png'))

    def _fit(self) -> None:
        if self.model is None:
            self.model = GPyOpt.methods.BayesianOptimization(
                f=None, domain=self.domain, acquisition_type=self.acqfunc,
                X=self.X, Y=self.y[:, np.newaxis], **self.kwargs)
        else:
            self.model.X, self.model.Y = self.X, self.y[:, np.newaxis]
        self.model._update_model()

    def _fix_seed(self, random_seed) -> None:
        np.random.seed(random_seed)
