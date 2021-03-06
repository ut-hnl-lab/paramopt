from pathlib import Path
import re
from typing import Any, Callable, Optional, Tuple, Union
import warnings

import numpy as np

from .cache import PredictionCache
from ..acquisitions.base import BaseAcquisition
from ..graphs.distribution import Distribution
from ..graphs.transition import Transition
from ..structures.parameter import ExplorationSpace
from ..structures.dataset import Dataset
from ..utils.string import formatted_now, indent_repr


def _map_to_builtin_types(np_array: np.ndarray) -> Tuple[Any, ...]:
    return tuple(val.item() for val in np_array)


class BaseOptimizer:
    """Base optimizer class.

    It is expected to inherit this class and override
    the `_fit_to_model()` and `_predict_with_model()` methods.
    """
    def __init__(
        self,
        workdir: Union[Path, str],
        exploration_space: 'ExplorationSpace',
        dataset: 'Dataset',
        acquisition: 'BaseAcquisition',
        objective_fn: Optional[Callable] = None
    ) -> None:
        if len(exploration_space.names) != len(dataset.X_names):
            raise ValueError(
                f"exploration space length ({len(exploration_space.names)}) does"
                f" not match observation names ({len(dataset.X_names)})")
        if exploration_space.names != dataset.X_names:
            warnings.warn(
                f"exploration space names ({exploration_space.names}) does not"
                f" match observation names ({dataset.X_names})",
                UserWarning, stacklevel=2)
        self.workdir = Path(workdir)
        if self.workdir.exists():
            warnings.warn(
                f"'{workdir}' already exists. The contents will be replaced!",
                UserWarning, stacklevel=2)

        self.exploration_space = exploration_space
        self.dataset = dataset
        self.acquisition = acquisition
        self._distribution = Distribution(
            exploration_space=exploration_space,
            objective_fn=objective_fn,
            acquisition_name=acquisition.name)
        self._transition = Transition(
            exploration_space=exploration_space)
        self._next_combination = None

        self.__sugg_cache = PredictionCache("sugg_cache")
        self.__plot_cache = PredictionCache("plot_cache")

        self.exploration_space.to_json(workdir)
        self.dataset.to_csv(workdir)

        if len(dataset) > 0:
            self._fit_to_model(dataset.X, dataset.Y)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(\n"
                + indent_repr(",\n".join(
                    f"{key}={repr(val)}" for key, val \
                    in dict(filter(
                        lambda d: not d[0].startswith(f"_"),
                        self.__dict__.items()
                    )).items()))
                + "\n)")

    def update(self, X: Any, Y: Any, label: Optional[Any] = None) -> None:
        """Update dataset and model with new X and corresponding y.

        The dataset csv file is also updated.

        Parameters
        ----------
        X: Any
            Numeric value or array.
        Y: Any
            Numeric value or array. This is usually acquired by real
            experiments.
        label: Any, optional
            The label assigned to the data.
            If the label is set to 'None', current time is used instead.
        """
        dataset_added = self.dataset.add(
            X, Y, label if label is not None else formatted_now())
        self._fit_to_model(dataset_added.X, dataset_added.Y)
        dataset_added.to_csv(self.workdir)
        self.dataset = dataset_added
        self._next_combination = None

    def suggest(self) -> Tuple[Any, ...]:
        """Determines the next combination of parameters based on gpr predictions
        and given acquisition function.

        The predictions are cached in a `sugg_cache.npz` file.

        Returns
        -------
        tuple
            Parameter conbination.
        """
        param_conbinations = self.exploration_space.combinations()
        mean_, std_ = self._predict_with_model(param_conbinations)
        mean, std = mean_.reshape(-1, 1), std_.reshape(-1, 1)
        acq = self.acquisition(mean, std, self.dataset.X, self.dataset.Y)
        self._next_combination = param_conbinations[np.argmax(acq)]

        self.__sugg_cache.stack(
            self.dataset.last_label, mean, std, acq, self._next_combination)
        self.__sugg_cache.save(self.workdir)
        return _map_to_builtin_types(self._next_combination)

    def plot(self, display: bool = False) -> None:
        """Plots the distributions of dataset and gpr predictions, and the
        transition of parameter values and the objective score. The plots are
        saved as png files.

        The predictions are cached in a `plot_cache.npz` file.

        Parameters
        ----------
        display: bool, default to `False`
            If `True`, the plots are displayed on separated windows.
        """
        param_conbinations = self.exploration_space.grid_combinations()
        mean_, std_ = self._predict_with_model(param_conbinations)
        mean, std = mean_.reshape(-1, 1), std_.reshape(-1, 1)
        acq = self.acquisition(mean, std, self.dataset.X, self.dataset.Y)

        self._transition.plot(dataset=self.dataset)
        if display:
            self._transition.show()
        self._transition.to_png(self.workdir, self.dataset.last_label)

        if self.exploration_space.dimension > 2:
            return

        self._distribution.plot(
            dataset=self.dataset, mean=mean, std=std, acquisition=acq,
            next_X=self._next_combination)
        if display:
            self._distribution.show()
        self._distribution.to_png(self.workdir, self.dataset.last_label)

        self.__plot_cache.stack(
            self.dataset.last_label, mean, std, acq, self._next_combination)
        self.__plot_cache.save(self.workdir)

    def _fit_to_model(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Trains the model using the entire dataset."""
        raise NotImplementedError

    def _predict_with_model(self, X: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Predicts data distribution with the trained model."""
        raise NotImplementedError
