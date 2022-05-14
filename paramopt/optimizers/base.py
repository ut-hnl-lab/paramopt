"""プロセスパラメータ空間を探索し, 値を逐次的に更新するプログラム."""

import os
from typing import Any, Dict, Generator, Optional, Union
import warnings

import numpy as np
import pandas as pd

from .exceptions import *
from ..parameter import ProcessParameter
from .. import utils


class BaseOptimizer:
    """ベースとなる最適化クラス.

    プロセスパラメータとその組み合わせの管理, 追加したデータの蓄積, 学習履歴の保存,
    学習経過のグラフ出力をサポートする. 継承は任意.

    Parameters
    ----------
        savedir: 学習履歴を書き出すディレクトリ
    """
    def __init__(self, savedir: str) -> None:
        if os.path.isdir(savedir):
            warnings.warn(
                f'{savedir} already exists.'
                +' The contents with the same name will be replaced!',
                UserWarning)

        self.savedir = savedir
        self.params = ProcessParameter()
        self.y_name = 'y'
        self.label_name = 'label'
        self.csv_name = 'exploration_history.csv'
        self.X = None
        self.y = np.empty(0)
        self.labels = []
        self.fig = None

    def add_parameter(self, name: str, space: Union[list, Generator]) -> None:
        """プロセスパラメータを追加する.

        Parameters
        ----------
            name: パラメータ名
            space: パラメタが取り得る値のリストもしくは範囲のジェネレータ(range等)
        """
        if self.y.shape[0] > 0:
            raise ParameterError(
                'Process parameters cannot be added after exploration started')

        self.params.add(name, np.array(space))
        self.X = np.empty((0, self.params.ndim))

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
        if self.y.shape[0] > 0:
            raise FittingError(
                'Existing dataset cannot be fitted after exploration started')

        df = pd.read_csv(csvpath).dropna(subset=[self.y_name])
        length = len(df)
        if length == 0:
            return

        self.X = df[self.params.names].values
        self.y = df[self.y_name].values
        self.labels = df[self.label_name].fillna('').tolist()
        self._fit()

    def fit(self, X: Any, y: Any, label: Optional[Any] = None) -> None:
        """モデルに新しいデータを学習させる.

        Parameters
        ----------
            X: 実験に用いたプロセスパラメータの組み合わせ
            y: 実験したときの中間データの測定値
            label: csvで保存する際に付け加えるラベル
                デフォルトは日時.
        """
        if self.X is None:
            raise ParameterError(
                'At least one process parameter must be added before fitting')

        self.X = np.vstack((self.X, X))
        self.y = np.append(self.y, y)
        if label is None:
            label = utils.formatted_now()
        self.labels.append(label)

        self._fit()
        self._save()

    def next(self) -> Any:
        """次の探索パラメータを決定し取得する.

        Returns
        -------
            次のパラメータの組み合わせ
        """
        raise NotImplementedError

    def plot(self, *args, **kwargs) -> None:
        """学習の経過をグラフ化する."""
        raise NotImplementedError

    def _fit(self) -> None:
        raise NotImplementedError

    def _save(self) -> None:
        df = pd.DataFrame(self.X, columns=self.params.names)
        df[self.y_name] = self.y
        df[self.label_name] = self.labels

        os.makedirs(self.savedir, exist_ok=True)
        df.to_csv(os.path.join(self.savedir, self.csv_name), index=False)
