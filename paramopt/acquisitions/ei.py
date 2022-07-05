import numpy as np
from scipy.special import erfc

from .base import BaseAcquisition


class EI(BaseAcquisition):
    """Expected Improvement.

    Parameters
    ----------
    xi: float
        Offset.
    """
    def __init__(self, xi: float = 0.0) -> None:
        self.xi = xi

    def __call__(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        X: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        if y.shape[0] == 0:
            ymax = 0
        else:
            ymax = np.max(y)

        z = (mean - ymax - self.xi) / std
        phi = np.exp(-0.5 * z**2) / np.sqrt(2*np.pi)
        Phi = 0.5 * erfc(-z / np.sqrt(2))
        return  std * (z * Phi + phi)
