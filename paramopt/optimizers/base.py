"""プロセスパラメータ空間を探索し, 値を逐次的に更新するプログラム."""

import os
from typing import Any, Dict, Generator, Optional, Union
import warnings

import numpy as np
import pandas as pd

from ..parameter import ProcessParameter
from .. import utils


warnings.filterwarnings('ignore')


class BaseOptimizer:
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

    def graph(self, *args, **kwargs) -> None:
        """学習の経過をグラフ化する."""
        raise NotImplementedError

    def _fit(self) -> None:
        raise NotImplementedError

    def _save(self) -> None:
        df = pd.DataFrame(self.X, columns=self.params.names)
        df[self.y_name] = self.y
        df[self.tag_name] = self.tags

        os.makedirs(self.savedir, exist_ok=True)
        df.to_csv(os.path.join(self.savedir, 'search_history.csv'), index=False)
