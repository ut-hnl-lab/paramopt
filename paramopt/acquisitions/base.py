import numpy as np


class BaseAcquisition:
    """Base class for acquisition function classes.

    Inherit this class and
    write calculation in overrided `__call__()` method to create a new one.
    """

    def __call__(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        X: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """Calculate acquisition

        Parameters
        ----------
        mean : numpy.ndarray
            Predicted mean.
        std : numpy.ndarray
            Predicted standard deviation.
        X : numpy.ndarray
            X of known datasets.
        y : numpy.ndarray
            y of known datasets.

        Returns
        -------
        numpy.ndarray
            Acquisition.
        """
        raise NotImplementedError
