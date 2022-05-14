"""プロセスパラメータ空間を探索し, 値を逐次的に更新するプログラム."""

from itertools import product
import os
from typing import Any, Callable, Dict, Generator, Literal, Optional, Tuple, Union

import GPy.kern as gk
import GPyOpt
import numpy as np
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as sk

from .acquisitions.base import BaseAcquisition
from .parameter import ProcessParameter
from . import utils
from . import plot

import warnings


warnings.filterwarnings('ignore')


class BaseLearner:
    """ベースとなる学習クラス.

    プロセスパラメータとその組み合わせの管理, 追加したデータの蓄積, 学習履歴の保存,
    学習経過のグラフ出力をサポートする. 継承は任意.

    Parameters
    ----------
        savedir: 学習履歴を書き出すディレクトリ
    """
    def __init__(self, savedir: str) -> None:
        self.savedir = savedir
        self.params = ProcessParameter()
        self.y_name = 'y'
        self.tag_name = 'tag'
        self.X = np.empty((0, 0))
        self.y = np.empty(0)
        self.tags = []
        self.fig = None

    def add_parameter(self, name: str, space: Union[list, Generator]) -> None:
        """プロセスパラメータを追加する.

        Parameters
        ----------
            name: パラメータ名
            values: パラメタが取り得る値のリストもしくは範囲のジェネレータ(range等)
        """
        array = np.array(space)
        self.X = np.hstack((self.X, np.empty((0, 1))))
        self.params.add(name, array)

    def add_parameter_from_dict(
        self, dict_: Dict[str, Union[list, Generator]]
    ) -> None:
        """辞書形式でプロセスパラメータを追加する.

        Parameters
        ----------
            dict_: Dict[str, Union[list, Generator]]
                {"param1": [1, 2, 3, 4, 5]} の形式.
        """
        for key, values in dict_.items():
            self.add_parameter(key, values)

    def prefit(self, csvpath: str = None) -> None:
        """既存のcsvデータを学習させ, 続きから学習を始める.

        Parameters
        ----------
            csvpath: 学習履歴のcsvパス
        """
        df = pd.read_csv(csvpath).dropna(subset=[self.y_name])
        length = len(df)

        if length == 0:
            return

        self.X = df[self.params.names].values
        self.y = df[self.y_name].values
        self.tags = df[self.tag_name].fillna('').tolist()
        self._fit()

    def fit(self, X: Any, y: Any, tag: Optional[Any] = None) -> None:
        """モデルに新しいデータを学習させる.

        Parameters
        ----------
            X: 実験に用いたプロセスパラメータの組み合わせ
            y: 実験したときの中間データの測定値
            tag: csvで保存する際に付け加えるタグ
                デフォルトは日時.
        """
        self.X = np.vstack((self.X, X))
        self.y = np.append(self.y, y)
        if tag is None:
            tag = utils.formatted_now()
        self.tags.append(tag)

        self._fit()
        self._save()

    def next(self) -> Any:
        """次の探索パラメータを決定し取得する.

        Returns
        -------
            次のパラメータの組み合わせ
        """
        raise NotImplementedError

    def _fit(self) -> None:
        raise NotImplementedError

    def _save(self) -> None:
        df = pd.DataFrame(self.X, columns=self.params.names)
        df[self.y_name] = self.y
        df[self.tag_name] = self.tags

        os.makedirs(self.savedir, exist_ok=True)
        df.to_csv(os.path.join(self.savedir, 'search_history.csv'), index=False)


class GPR(BaseLearner):
    """sklearnベースのガウス過程回帰モデル.

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

    def graph(
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
        plot.savefig(os.path.join(self.savedir, f'plot-{self.tags[-1]}.png'))

    def _fit(self) -> None:
        self.model.fit(self.X, self.y)

    def _fix_seed(self, random_seed) -> None:
        np.random.seed(random_seed)
        self.model.random_state = random_seed


class GPyBO(BaseLearner):
    """GPyOptをラップしたガウス過程回帰モデル.

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
