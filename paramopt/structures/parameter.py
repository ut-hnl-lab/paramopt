from dataclasses import dataclass, asdict, field
import json
from pathlib import Path
from typing import List, ClassVar, Union
from attr import s

from dacite import from_dict
import numpy as np


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

    @staticmethod
    def from_json(filepath: Union[Path, str]) -> 'ExplorationSpace':
        """Reads the the exploration space definition from a json file.

        Parameters
        ----------
        filepath : pathlib.Path or str
            Json file path

        Returns
        -------
        ExplorationSpace
            New `ExplorationSpace` containing json data.
        """
        filepath_ = Path(filepath)
        with filepath_.open(mode='r') as f:
            data = json.load(f)
        return from_dict(ExplorationSpace, data)

    def to_json(self, directory: Union[Path, str]) -> None:
        """Writes the exploration space definition to a csv file in the given
        directory.

        If the json file with the `ExplorationSpace.EXPORT_NAME` does not exist
        in the directory, new one is created.

        Parameters
        ----------
        directory : pathlib.Path or str
            Directory where json files are output.
        """
        directory_ = Path(directory)
        directory_.mkdir(exist_ok=True, parents=True)
        with (directory_/self.EXPORT_NAME).open(mode='w') as f:
            json.dump(asdict(self), f, indent=4)

    def conbinations(self) -> np.ndarray:
        """Creates all combinations of process parameter values.

        Returns
        -------
        numpy.ndarray
            Array of combinations.
        """
        return np.array(
            np.meshgrid(*self.spaces)).T.reshape(-1, self.dimension)

    def grid_conbinations(self) -> np.ndarray:
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
            raise ValueError("values must not be empty")
        elif length == 1:
            self.grid_values = self.values.copy()
        else:
            vmax, vmin = max(self.values), min(self.values)
            self.grid_values = np.linspace(
                vmin, vmax, self.N_GRID_SPLITS).tolist()
