import numpy as np


class BaseAcquisition:
    """Base class for acquisition function classes.

    Inherit this class and
    write calculation in overrided `__call__()` method to create a new one.
    """

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                + ", ".join(f"{key}={val}" for key, val in self.__dict__.items())
                + ")")

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
