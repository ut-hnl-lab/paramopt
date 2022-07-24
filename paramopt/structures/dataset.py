from pathlib import Path
from typing import Any, List, Optional, Union
import numpy as np
import pandas as pd

from ..utils.string import indent_repr


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
        labels: Optional[np.ndarray] = None
    ) -> None:
        self.X_names = X_names if isinstance(X_names, list) else [X_names]
        self.Y_names = Y_names if isinstance(Y_names, list) else [Y_names]
        n_X_names, n_Y_names = len(self.X_names), len(self.Y_names)

        self.__X = np.atleast_2d(X) if X is not None else np.empty((0, n_X_names))
        self.__Y = np.atleast_2d(Y) if Y is not None else np.empty((0, n_Y_names))
        n_X, n_Y = self.__X.shape[1], self.__Y.shape[1]

        if n_X_names != n_X:
            raise ValueError(
                f"X_names length ({n_X_names}) does not match X length ({n_X})")
        if n_Y_names != n_Y:
            raise ValueError(
                f"Y_names length ({n_Y_names}) does not match Y length ({n_Y})")

        self.label_name = label_name
        self.__labels = np.array(labels) if labels is not None else np.empty(0)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(\n"
                + indent_repr(",\n".join([
                    f"X_names={self.X_names}",
                    f"Y_names={self.Y_names}",
                    f"label_name=\"{self.label_name}\"",
                    f"X={repr(self.__X)}".replace("\n      ",""),
                    f"Y={repr(self.__Y)}".replace("\n      ",""),
                    f"labels={self.__labels}"
                ]))
                + "\n)")

    def __len__(self) -> int:
        return self.__X.shape[0]

    def __getitem__(self, item) -> 'Dataset':
        X = self.__X[item, :]
        Y = self.__Y[item, :]
        labels = self.__labels[item]
        return Dataset(self.X_names, self.Y_names, self.label_name, X, Y, labels)

    @property
    def X(self) -> np.ndarray:
        return self.__X

    @property
    def Y(self) -> np.ndarray:
        return self.__Y

    @property
    def labels(self) -> List[Any]:
        return self.__labels.tolist()

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

    def add(self, X: Any, Y: Any, label: Any = "") -> 'Dataset':
        """Add new data with a label.

        Parameters
        ----------
        X : Any
            Numeric value or array.
        Y : Any
            Numeric value or array.
        label: Any, default to ""
            The label assigned to the data.

        Returns
        -------
        Dataset
            New `Dataset` containing added data.

        Raises
        ------
        ValueError
            Raises when the number of X, Y, and label is different.
        """
        X_adding = np.atleast_2d(X)
        Y_adding = np.atleast_2d(Y)
        n_X, n_Y = X_adding.shape[0], Y_adding.shape[0]

        if n_X != n_Y:
            raise ValueError(
                f"length of X ({n_X}) does not match length of Y ({n_Y})")

        label_adding = np.array([label for _ in range(n_X)]) \
            if not isinstance(label, (list, tuple, np.ndarray)) \
            else np.array(label)
        n_label = label_adding.shape[0]

        if n_label != n_X:
            raise ValueError(
                f"number of label ({n_label}) does not match" \
                + f" length of array ({n_X})")

        X = np.vstack((self.__X, X_adding))
        Y = np.vstack((self.__Y, Y_adding))
        labels = np.append(self.__labels, label_adding)

        return Dataset(self.X_names, self.Y_names, self.label_name, X, Y, labels)

    @classmethod
    def from_csv(
        cls, path: Union[Path, str] = None, n_X: int = 1, n_Y: int = 1,
    ) -> 'Dataset':
        """Reads thee data in a csv file and generates a `Dataset` instance.

        Parameters
        ----------
        path : pathlib.Path or str, default is `None`
            Path to the input csv file or the directory where the file exists.
            If set to `None`, `pathlib.Path.cwd()/Dataset.EXPORT_NAME` is used.
        n_X : int, default is `1`
            Number of category X.
        n_Y : int, default is `1`
            Number of category Y.

        Returns
        -------
        Dataset
            New `Dataset` containing imported csv data.

        Raises
        ------
        ValueError
            Raises if the total number of categories and `n_X + n_Y + 1 (number
            of label category)` are different.
        """
        path_ = Path(path) if path is not None else Path.cwd()

        if path_.is_file():
            filepath = path_
        elif path_.is_dir():
            filepath = path_/cls.EXPORT_NAME
        else:
            raise FileNotFoundError('no such file or directory')

        df = pd.read_csv(filepath)
        if len(df.columns) != n_X + n_Y + 1:
            raise ValueError(
                f"column length (X={n_X}, Y={n_Y}) does not match" \
                + " given data length")

        X_names = df.columns[:n_X].to_list()
        Y_names = df.columns[n_X:n_X+n_Y].to_list()
        label_name = df.columns[-1]

        df_ = df.dropna(subset=Y_names)
        X = df_[X_names].values
        Y = df_[Y_names].values
        labels = df_[label_name].fillna('').to_list()

        return Dataset(X_names, Y_names, label_name, X, Y, labels)

    def to_csv(self, path: Union[Path, str] = None) -> None:
        """Crates a csv file and writes data in it.

        Parameters
        ----------
        path : pathlib.Path or str, default is `None`
            Path of the output csv file or the directory where the file is
            exported.
            If set to `None`, `pathlib.Path.cwd()/Dataset.EXPORT_NAME` is used.
        """
        path_ = Path(path) if path is not None else Path.cwd()

        if path_.suffix != "":
            path_.parent.mkdir(exist_ok=True, parents=True)
            filepath = path_
        else:
            path_.mkdir(exist_ok=True, parents=True)
            filepath = path_/self.EXPORT_NAME

        X_df = pd.DataFrame(self.__X, columns=self.X_names)
        Y_df = pd.DataFrame(self.__Y, columns=self.Y_names)
        label_df = pd.DataFrame(self.__labels, columns=[self.label_name])
        concated = pd.concat([X_df, Y_df, label_df], axis=1)

        concated.to_csv(filepath, index=False)
