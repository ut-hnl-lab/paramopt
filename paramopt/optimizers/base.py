"""プロセスパラメータ空間を探索し, 値を逐次的に更新するプログラム."""

from pathlib import Path
from typing import Any, Callable, Dict, Generator, Optional, Tuple, Union
import warnings

import numpy as np

from ..graphs.distribution import Distribution
from ..graphs.transition import Transition

from .exceptions import *
from ..acquisitions.base import BaseAcquisition
from ..structures.parameter import ExplorationSpace
from ..structures.dataset import Dataset
from .. import utils


class BaseOptimizer:
    """ベースとなる最適化クラス.

    プロセスパラメータとその組み合わせの管理, 追加したデータの蓄積, 学習履歴の保存,
    学習経過のグラフ出力をサポートする. 継承は任意.

    Parameters
    ----------
        workdir: 学習履歴を書き出すディレクトリ
    """
    def __init__(
        self,
        workdir: Union[Path, str],
        exploration_space: 'ExplorationSpace',
        dataset: 'Dataset',
        acquisition: 'BaseAcquisition',
        objective_fn: Optional[Callable] = None
    ) -> None:
        if len(exploration_space.names) != len(dataset.X_names):
            raise ValueError(
                f"exploration space length ({len(exploration_space.names)}) does",
                f" not match observation names ({len(dataset.X_names)})")
        if exploration_space.names != dataset.X_names:
            warnings.warn(
                f"exploration space names ({exploration_space.names}) does not",
                f" match observation names ({dataset.X_names})",
                UserWarning)
        self.workdir = Path(workdir)
        if self.workdir.exists():
            warnings.warn(
                f"'{workdir}' already exists. The contents may be replaced!",
                UserWarning)

        self.exploration_space = exploration_space
        self.dataset = dataset
        self.acquisition = acquisition
        self.objective_fn = objective_fn
        self.distribution = Distribution()
        self.transition = Transition()

        self.exploration_space.to_json(self.workdir)
        self.dataset.to_csv(self.workdir)

    def update(self, X: Any, y: Any, label: Optional[Any] = None) -> None:
        """モデルに新しいデータを学習させる.

        Parameters
        ----------
            X: 実験に用いたプロセスパラメータの組み合わせ
            y: 実験したときの中間データの測定値
            label: csvで保存する際に付け加えるラベル
                デフォルトは日時.
        """
        dataset_added = self.dataset.add(
            X, y, label if label is not None else utils.formatted_now())
        self._fit_to_model(dataset_added.X, dataset_added.Y)
        dataset_added.to_csv(self.workdir)
        self.dataset = dataset_added

    def suggest(self) -> Tuple[Any, ...]:
        """次の探索パラメータを決定し取得する.

        Returns
        -------
            次のパラメータの組み合わせ
        """
        param_conbinations = self.exploration_space.conbinations()
        mean, std = self._predict_with_model(param_conbinations)
        acq = self.acquisition(mean, std, self.dataset.X, self.dataset.Y)
        return tuple(param_conbinations[np.argmax(acq)])

    def plot(self) -> None:
        """学習の経過をグラフ化する.

        Parameters
        ----------
            objective_fn: 目的関数
                真の関数が分かっている場合(=テスト時)に指定すると, 共に描画.
            overwrite: 1つのウィンドウに対してグラフを上書き更新するか否か
                jupyter notebook使用時はFalseを推奨
        """
        param_conbinations = self.exploration_space.grid_conbinations()
        mean, std = self._predict_with_model(param_conbinations)
        acq = self.acquisition(mean, std, self.dataset.X, self.dataset.Y)

        if self.exploration_space.dimension <= 2:
            self.distribution.plot(
                self.exploration_space, self.dataset, mean, std, acq,
                self.objective_fn)
            self.distribution.show()
            self.distribution.to_png(self.workdir, self.dataset.last_label)

        self.transition.plot(self.exploration_space, self.dataset)
        self.transition.show()
        self.transition.to_png(self.workdir, self.dataset.last_label)

    def _fit_to_model(self, X: np.ndarray, Y: np.ndarray) -> None:
        raise NotImplementedError

    def _predict_with_model(self, X: np.ndarray) -> Tuple[np.ndarray, ...]:
        raise NotImplementedError
