from dataclasses import dataclass, asdict, field
import json
from pathlib import Path
from typing import List, ClassVar, Union

from dacite import from_dict
import numpy as np


@dataclass
class ExplorationSpace:

    EXPORT_NAME: ClassVar[str] = "exploration_space.json"

    process_parameters: List['ProcessParameter'] = field(default_factory=list)

    def __post_init__(self) -> None:
        names = [p.name for p in self.process_parameters]
        for name in names:
            if names.count(name) > 1:
                raise ValueError(f"duplicated name: '{name}'")

    @property
    def dimension(self) -> int:
        return len(self.process_parameters)

    @property
    def names(self) -> List[str]:
        return [param.name for param in self.process_parameters]

    @property
    def spaces(self) -> List[List[Union[int, float]]]:
        return [np.array(param.values) for param in self.process_parameters]

    @property
    def grid_spaces(self) ->  List[List[Union[int, float]]]:
        return [np.array(param.grid_values) for param in self.process_parameters]

    @staticmethod
    def from_json(filepath: Union[Path, str]) -> 'ExplorationSpace':
        filepath_ = Path(filepath)
        with filepath_.open(mode='r') as f:
            data = json.load(f)
        return from_dict(ExplorationSpace, data)

    def to_json(self, directory: Union[Path, str]) -> None:
        directory_ = Path(directory)
        directory_.mkdir(exist_ok=True, parents=True)
        with (directory_/self.EXPORT_NAME).open(mode='w') as f:
            json.dump(asdict(self), f, indent=4)

    def conbinations(self) -> np.ndarray:
        return np.array(
            np.meshgrid(*self.spaces)).T.reshape(-1, self.dimension)

    def grid_conbinations(self) -> np.ndarray:
        return np.array(
            np.meshgrid(*self.grid_spaces)).T.reshape(-1, self.dimension)


@dataclass
class ProcessParameter:

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
