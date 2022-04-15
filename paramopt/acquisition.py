from typing import Any
import numpy as np

from scipy.stats import norm
from scipy.special import ndtr, erfc


class UCB:
    """Upper Confidence Bounds.

    Parameters
    ----------
        c: 探索と活用のトレードオフのウェイトを決めるパラメータ. C値
            大きいほど広く探索し, 小さいほど早く収束する
    """
    def __init__(self, c: float) -> None:
        self.c = c

    def __call__(
        self, mean: np.ndarray, std: np.ndarray, *args: Any, **kwargs: Any
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


class EI:
    """Expected Improvement.

    Parameters
    ----------
        xi: オフセット
    """
    def __init__(self, xi: float = 0.0) -> None:
        self.xi = xi

    def __call__(
        self, mean: np.ndarray, std: np.ndarray, ymax: float, *args: Any,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Parameters
        ----------
            mean: 予測される分布の平均
            std: 予測される分布の標準偏差
            ymax: これまでのyの最大値

        Returns
        -------
            獲得関数値の分布
        """
        z = (mean - ymax - self.xi) / std
        phi = np.exp(-0.5 * z**2) / np.sqrt(2*np.pi)
        Phi = 0.5 * erfc(-z / np.sqrt(2))
        # return (mean - ymax - self.xi)*ndtr(z) + std*norm.pdf(z)
        return  std * (z * Phi + phi)
