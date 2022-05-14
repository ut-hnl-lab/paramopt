import numpy as np

from .base import BaseAcquisition


class UCB(BaseAcquisition):
    """Upper Confidence Bounds.

    Parameters
    ----------
        c: 探索と活用のトレードオフのウェイトを決めるパラメータ. C値
            大きいほど広く探索し, 小さいほど早く収束する
    """
    def __init__(self, c: float) -> None:
        self.c = c

    def __call__(
        self, mean: np.ndarray, std: np.ndarray, X: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """
        Parameters
        ----------
            mean: 予測される分布の平均
            std: 予測される分布の標準偏差

        Returns
        -------
            獲得関数値の分布
        """
        return mean + self.c * std
