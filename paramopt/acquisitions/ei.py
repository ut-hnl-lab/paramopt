import numpy as np
from scipy.special import erfc

from .base import BaseAcquisition


class EI(BaseAcquisition):
    """Expected Improvement.

    Parameters
    ----------
        xi: オフセット
    """
    def __init__(self, xi: float = 0.0) -> None:
        self.xi = xi

    def __call__(
        self, mean: np.ndarray, std: np.ndarray, X: np.ndarray, y: np.ndarray
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
        if y.shape[0] == 0:
            ymax = 0
        else:
            ymax = np.max(y)

        z = (mean - ymax - self.xi) / std
        phi = np.exp(-0.5 * z**2) / np.sqrt(2*np.pi)
        Phi = 0.5 * erfc(-z / np.sqrt(2))
        return  std * (z * Phi + phi)
