from pathlib import Path
from typing import Any, List, Optional, Union
import numpy as np
import pandas as pd

from .. import utils


class Dataset:
    """Data structure with X, Y and label. A label can be assigned to each
    dataset.

    Parameters
    ----------
    X_names : str or list of str
        Category names of X.
    Y_names : str or list of str
        Category names of Y.
    label_name : str, optional
        Category name of label.
    X : numpy.ndarray, optional
        X data.
    Y : numpy.ndarray, optional
        Y data.
    labels : list, optional
        Labels for each dataset.

    Raises
    ------
    ValueError
        Raises if the length of names and data are different.
    """
    EXPORT_NAME: str = "dataset.csv"

    def __init__(
        self,
        X_names: Union[List[str], str],
        Y_names: Union[List[str], str],
        label_name: str = 'label',
        X: Optional[np.ndarray] = None,
        Y: Optional[np.ndarray] = None,
        labels: Optional[List[Any]] = None
    ) -> None:
        self.X_names = X_names if isinstance(X_names, list) else [X_names]
        self.Y_names = Y_names if isinstance(Y_names, list) else [Y_names]
        n_X_names, n_Y_names = len(self.X_names), len(self.Y_names)

        self.__X = np.atleast_2d(X) if X is not None else np.empty((0, n_X_names))
        self.__Y = np.atleast_2d(Y) if Y is not None else np.empty((0, n_Y_names))
        n_X, n_Y = self.__X.shape[1], self.__Y.shape[1]

        if n_X_names != n_X:
            raise ValueError(
                f"X_names length ({n_X}) does not match X length ({n_X_names})")
        if n_Y_names != n_Y:
            raise ValueError(
                f"Y_names length ({n_Y}) does not match Y length ({n_Y_names})")

        self.__label_name = label_name
        self.__labels = labels if labels is not None else []

    def __len__(self) -> int:
        return self.__X.shape[0]

    @property
    def X(self) -> np.ndarray:
        return self.__X

    @property
    def Y(self) -> np.ndarray:
        return self.__Y

    @property
    def dimension_X(self) -> int:
        return self.__X.shape[1]

    @property
    def dimension_Y(self) -> int:
        return self.__Y.shape[1]

    @property
    def last_label(self) -> str:
        if len(self.__labels) == 0:
            raise ValueError("no data added")
        return self.__labels[-1]

    def add(self, X: Any, Y: Any, label: Optional[str] = None) -> 'Dataset':
        """Add new data with a label.

        Parameters
        ----------
        X : Any
            Numeric value or array.
        Y : Any
            Numeric value or array.
        label: str, optional
            The label assigned to the data.
            If the label is set to 'None', current time is used instead.

        Returns
        -------
        Dataset
            New `Dataset` containing added data.

        Raises
        ------
        ValueError
            Raises when trying to add two or more pieces of data at once.
        """
        X_adding = np.atleast_2d(X)
        Y_adding = np.atleast_2d(Y)

        if X_adding.shape[0] != 1 or Y_adding.shape[0] != 1:
            raise ValueError("input arrays must be 1-dimensional")

        X = np.vstack((self.__X, X_adding))
        Y = np.vstack((self.__Y, Y_adding))
        labels = self.__labels \
                 + [label if label is not None else utils.formatted_now()]

        return Dataset(
            self.X_names, self.Y_names, self.__label_name, X, Y, labels)

    @staticmethod
    def from_csv(
        filepath: Union[Path, str], n_X: int, n_Y: int = 1
    ) -> 'Dataset':
        """Reads data from a csv file and generates a `Dataset` instance.

        Parameters
        ----------
        filepath : pathlib.Path or str
            Csv file path.
        n_X : int
            Number of category X.
        n_Y : int, optional
            Number of category Y.

        Returns
        -------
        Dataset
            New `Dataset` containing csv data.

        Raises
        ------
        ValueError
            Raises if the total number of categories and `n_X + n_Y + 1 (number
            of label category)` are different.
        """
        df = pd.read_csv(Path(filepath))
        if len(df.columns) != n_X + n_Y + 1:
            raise ValueError("column length does not match given data length")

        X_names = df.columns[:n_X].to_list()
        Y_names = df.columns[n_X:n_X+n_Y].to_list()
        label_name = df.columns[-1]

        df_ = df.dropna(subset=Y_names)
        X = df_[X_names].values
        Y = df_[Y_names].values
        labels = df_[label_name].fillna('').to_list()

        return Dataset(X_names, Y_names, label_name, X, Y, labels)

    def to_csv(self, directory: Union[Path, str]) -> None:
        """Write data to a csv file in the given directory.

        If the csv file with the `Dataset.EXPORT_NAME` does not exist in the
        directory, new one is created.

        Parameters
        ----------
        directory : pathlib.Path or str
            Directory where csv files are output.
        """
        X_df = pd.DataFrame(self.__X, columns=self.X_names)
        Y_df = pd.DataFrame(self.__Y, columns=self.Y_names)
        label_df = pd.DataFrame(self.__labels, columns=[self.__label_name])
        concated = pd.concat([X_df, Y_df, label_df], axis=1)

        directory_ = Path(directory)
        directory_.mkdir(exist_ok=True, parents=True)
        concated.to_csv(directory_/self.EXPORT_NAME, index=False)
