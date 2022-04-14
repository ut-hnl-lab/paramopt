"""プロセスパラメータ空間を探索し, 値を逐次的に更新するプログラム."""

from itertools import product
import os
from typing import Any, Callable, Generator, List, Literal, Optional, Tuple, Union

import GPy.kern as gk
import GPyOpt
from matplotlib import cm, gridspec, pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as sk

from .acquisition import UCB, EI
from . import utils

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
        self.X_names = []
        self.y_name = 'y'
        self.tag_name = 'tag'
        self.Xs = []
        self.X_grids = []
        self.X = np.empty((0, 0))
        self.y = np.empty(0)
        self.tags = []
        self.fig = None

    def add_parameter(self, name: str, values: Union[list, Generator]) -> None:
        """プロセスパラメータを追加する.

        Parameters
        ----------
            name: パラメータ名
            values: パラメタが取り得る値のリストもしくは範囲のジェネレータ(range等)
        """
        array = np.array(values)
        self.X_names.append(name)
        self.Xs.append(array)
        step = (np.max(array)-np.min(array))/100
        self.X_grids.append(np.arange(np.min(array), np.max(array)+step, step))
        self.X = np.hstack((self.X, np.empty((0, 1))))

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

        self.X = df[self.X_names].values
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
        df = pd.DataFrame(self.X, columns=self.X_names)
        df[self.y_name] = self.y
        df[self.tag_name] = self.tags

        os.makedirs(self.savedir, exist_ok=True)
        df.to_csv(os.path.join(self.savedir, 'search_history.csv'), index=False)

    def graph(
        self, X: np.ndarray, y:np.ndarray, mean: np.ndarray, std: np.ndarray,
        acq: np.ndarray, objective_fn: Optional[Callable], onewindow: bool = True
    ) -> None:
        """学習の経過をグラフ化する.

        Parameters
        ----------
            objective_fn: 目的関数
                真の関数が分かっている場合(=テスト時)に指定すると, 共に描画.
            onewindow: 1つのウィンドウに対してグラフを上書き更新するか否か
                jupyter notebook使用時はFalseを推奨
        """
        if onewindow:
            if self.fig is None:
                self.fig = plt.figure()
            else:
                self.fig.clear()
        else:
            plt.close()
            self.fig = plt.figure()

        dim = len(self.Xs)
        if dim == 1:  # 1次元パラメータ→2D描写
            X_grid = self.X_grids[0]
            spec = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[3, 1])

            ax = self.fig.add_subplot(spec[0])
            if objective_fn is not None:
                ax.plot(
                    X_grid, objective_fn(X_grid), 'k:', alpha=.5,
                    label='Objective fn')
            ax.plot(X_grid, mean, 'b-', label='Prediction')
            ax.fill(
                np.concatenate([X_grid, X_grid[::-1]]),
                np.concatenate([mean -1.96*std, (mean + 1.96*std)[::-1]]),
                'p-', alpha=.5, label='95% CI')
            ax.plot(X, y, 'k.', label='Observations')
            ax.plot(X[-1], y[-1], 'r*', markersize=10)
            ax.set_xlabel(self.X_names[0])
            ax.set_ylabel(self.y_name)
            ax.legend()

            ax2 = self.fig.add_subplot(spec[1])
            ax2.plot(X_grid, acq, 'r-')
            ax2.set_xlabel(self.X_names[0])
            ax2.set_ylabel('Acquisition')

        elif dim == 2:  # 2次元パラメータ→3D描写
            X_grid1, X_grid2 = self.X_grids
            Xmesh1, Xmesh2 = np.meshgrid(X_grid1, X_grid2)
            mean = mean.reshape(X_grid1.shape[0], X_grid2.shape[0])
            acq = acq.reshape(X_grid1.shape[0], X_grid2.shape[0])

            ax = self.fig.add_subplot(111, projection='3d')
            if objective_fn is not None:
                ax.plot_wireframe(
                    Xmesh1, Xmesh2, objective_fn(Xmesh1, Xmesh2),
                    color='k', alpha=0.5, linewidth=0.5, label='Objective fn')
            ax.plot_wireframe(
                Xmesh1, Xmesh2, mean.T, color='b', alpha=0.6, linewidth=0.5,
                label='Prediction')
            ax.scatter(
                X[:-1, 0], X[:-1, 1], y[:-1], c='black',
                label='Observations')
            ax.scatter(
                X[-1, 0], X[-1, 1], y[-1], c='red', marker='*',
                s=50)
            contf = ax.contourf(
                Xmesh1, Xmesh2, acq.T, zdir='z', offset=ax.get_zlim()[0],
                cmap=cm.jet, levels=100)
            self.fig.colorbar(contf, pad=0.08, shrink=0.6, label='Acquisition')
            ax.set_xlabel(self.X_names[0])
            ax.set_ylabel(self.X_names[1])
            ax.set_zlabel(self.y_name)
            ax.legend()

        else:
            raise NotImplementedError(f'{dim}D plot not supported')

        plt.tight_layout()
        os.makedirs(self.savedir, exist_ok=True)
        self.fig.savefig(os.path.join(self.savedir, f'{self.tags[-1]}.png'))
        if onewindow:
            plt.pause(0.1)
        else:
            plt.show(block=False)
            self.fig = None




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
        self, savedir: str, kernel: sk.Kernel, acqfunc: Callable,
        random_seed: Optional[int] = None,
    ) -> None:
        super().__init__(savedir)
        self.model = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
        self.acqfunc = acqfunc
        if random_seed is not None:
            self._fix_seed(random_seed)

    def add_parameter(self, name: str, values: Union[list, Generator]) -> None:
        super().add_parameter(name, values)
        self.X_combos = np.array(list(product(*self.Xs)))
        self.X_grid_combos = np.array(list(product(*self.X_grids)))

    def next(self) -> Tuple[Any]:
        # 全てのパラメータの組み合わせに対する平均と分散を計算
        mean, std = self.model.predict(self.X_combos, return_std=True)

        # 獲得関数値を基に次のパラメータの組み合わせを選択
        acq = self.acqfunc(mean=mean, std=std, ymax=np.max(self.y))
        next_idx = np.argmax(acq)  # 獲得関数を最大化するパラメータの組み合わせ
        next_X = tuple(self.X_combos[next_idx])
        return next_X

    def graph(
        self, objective_fn: Optional[Callable] = None, onewindow: bool = True,
        **kwargs: Any
    ) -> None:
        mean, std = self.model.predict(self.X_grid_combos, return_std=True)
        acq = self.acqfunc(mean=mean, std=std, ymax=np.max(self.y))
        super().graph(self.X, self.y, mean, std, acq, objective_fn, onewindow)

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
        'UCB', 'UCB_MCMC', 'UCB', 'UCB_MCMC'],
        random_seed: Optional[int] = None, **kwargs: Any
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
        self.X_grid_combos = np.array(list(product(*self.X_grids)))

    def next(self) -> Tuple[Any]:
        suggested_locations = self.model._compute_next_evaluations()
        next_X = tuple(suggested_locations[0])
        return next_X

    def graph(
        self, onewindow: bool = True, use_original: bool = True, **kwargs: Any
    ) -> None:
        """学習の経過をグラフ化する.

        Parameters
        ----------
            onewindow: 1つのウィンドウに対してグラフを上書き更新するか否か
                jupyter notebook使用時はFalseを推奨
            use_original: GPyOpt固有のグラフ描写関数を用いるか否か
        """
        if use_original:
            self.model.plot_acquisition(
                filename=os.path.join(self.savedir, f'{self.tags[-1]}.png'),
                label_x = self.X_names[0],
                label_y = self.y_name if len(self.X_names) < 2 else self.X_names[1])
        else:
            mean, std = self.model.model.model.predict(self.X_grid_combos)
            acq = -self.model.acquisition.acquisition_function(self.X_grid_combos)
            super().graph(
                self.model.model.model.X, self.model.model.model.Y, mean, std,
                acq, objective_fn=None, onewindow=onewindow)

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
