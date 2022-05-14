import numpy as np


class BaseAcquisition:
    """獲得関数の基底クラス. 継承してcomputeメソッドに計算を記述する."""

    def __call__(
        self, mean: np.ndarray, std: np.ndarray, X: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """獲得関数値を計算する.

        Parameters
        ----------
        mean : 平均
        std : 標準偏差
        X : 今までのパラメータ組合せ
        y : 今までの計測値

        Returns
        -------
        獲得関数値
        """
        raise NotImplementedError
