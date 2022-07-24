from dataclasses import dataclass, asdict, field
import json
from pathlib import Path
from typing import List, ClassVar, Optional, Union

from dacite import from_dict
import numpy as np

from ..utils.string import indent_repr


@dataclass
class ExplorationSpace:
    """Exploration space composed of process parameters.

    Parameters
    ----------
    process_parameters: list of paramopt.ProcesParameter
        Space definition. N-parameters means N-dimensional exploration.
    """
    EXPORT_NAME: ClassVar[str] = "exploration_space.json"

    process_parameters: List['ProcessParameter'] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.__names = [p.name for p in self.process_parameters]
        for name in self.__names:
            if self.__names.count(name) > 1:
                raise ValueError(f"duplicated name: '{name}'")

        self.__spaces = [
            np.array(param.values) for param in self.process_parameters]
        self.__grid_spaces = [
            np.array(param.grid_values) for param in self.process_parameters]

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(process_parameters=[\n"
                + indent_repr(", \n".join([
                    *(f"{p}" for p in self.process_parameters)
                ]))
                + "\n])")

    @property
    def dimension(self) -> int:
        return len(self.process_parameters)

    @property
    def names(self) -> List[str]:
        return self.__names

    @property
    def spaces(self) -> List[List[Union[int, float]]]:
        return self.__spaces

    @property
    def grid_spaces(self) ->  List[List[Union[int, float]]]:
        return self.__grid_spaces

    @classmethod
    def from_json(cls, path: Union[Path, str] = None) -> 'ExplorationSpace':
        """Reads the definition of the exploration space in a json file and
        generates a new `ExplorationSpace` instance.

        Parameters
        ----------
        path : pathlib.Path or str, default is `None`
            Path to the input json file or the directory where the file exists.
            If set to `None`, `pathlib.Path.cwd()/ExplorationSpace.EXPORT_NAME`
            is used.

        Returns
        -------
        ExplorationSpace
            New `ExplorationSpace` containing imported json data.
        """
        path_ = Path(path) if path is not None else Path.cwd()

        if path_.is_file():
            filepath = path_
        elif path_.is_dir():
            filepath = path_/cls.EXPORT_NAME
        else:
            raise FileNotFoundError('no such file or directory')

        with filepath.open(mode='r') as f:
            data = json.load(f)
        return from_dict(ExplorationSpace, data)

    def to_json(self, path: Union[Path, str] = None) -> None:
        """Creates a json file and writes the exploration space in it.

        Parameters
        ----------
        path : pathlib.Path or str, default is `None`
            Path of the output json file or the directory where the file is
            exported.
            If set to `None`, `pathlib.Path.cwd()/ExplorationSpace.EXPORT_NAME`
            is used.
        """
        path_ = Path(path) if path is not None else Path.cwd()

        if path_.suffix != "":
            path_.parent.mkdir(exist_ok=True, parents=True)
            filepath = path_
        else:
            path_.mkdir(exist_ok=True, parents=True)
            filepath = path_/self.EXPORT_NAME

        with filepath.open(mode='w') as f:
            json.dump(asdict(self), f, indent=4)

    def combinations(self) -> np.ndarray:
        """Creates all combinations of process parameter values.

        Returns
        -------
        numpy.ndarray
            Array of combinations.
        """
        return np.array(
            np.meshgrid(*self.spaces)).T.reshape(-1, self.dimension)

    def grid_combinations(self) -> np.ndarray:
        """Creates all combinations of grid values of process parameter spaces.

        Returns
        -------
        np.ndarray
            Array of combinations.
        """
        return np.array(
            np.meshgrid(*self.grid_spaces)).T.reshape(-1, self.dimension)


@dataclass
class ProcessParameter:
    """Process parameter definition composed of its name and possible values."""
    N_GRID_SPLITS: ClassVar[int] = 100

    name: str
    values: List[Union[int, float]]

    def __post_init__(self) -> None:
        length = len(self.values)
        if length < 1:
            raise ValueError("values cannot be empty")
        elif length == 1:
            self.grid_values = self.values.copy()
        else:
            vmax, vmin = max(self.values), min(self.values)
            self.grid_values = np.linspace(
                vmin, vmax, self.N_GRID_SPLITS).tolist()
