from dataclasses import dataclass
from typing import List, Union

import numpy as np


class ProcessParameter:
    def __init__(self) -> None:
        self.n_grid_split = 100
        self._data: List['ParamData'] = []

    def add(self, name: str, values: np.ndarray) -> None:
        vmax, vmin = np.max(values), np.min(values)
        step = (vmax-vmin)/self.n_grid_split
        grid = np.arange(vmin, vmax+step, step)
        self._data.append(ParamData(name, values, grid))

    @property
    def ndim(self) -> int:
        return len(self._data)

    @property
    def names(self) -> List[str]:
        return [d.name for d in self._data]

    @property
    def values(self) -> List[List[Union[int, float]]]:
        return [d.values for d in self._data]

    @property
    def grids(self) -> List[List[Union[int, float]]]:
        return [d.grid for d in self._data]


@dataclass
class ParamData:
    name: str
    values: List[Union[int, float]]
    grid: List[Union[int, float]]
