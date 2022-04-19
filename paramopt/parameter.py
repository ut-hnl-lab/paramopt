import numpy as np


class ProcessParameter:
    def __init__(self) -> None:
        self.n_grid_split = 100
        self.names = []
        self.values = []
        self.grids = []

    def add(self, name: str, values: np.ndarray) -> None:
        self.names.append(name)
        self.values.append(values)
        vmax, vmin = np.max(values), np.min(values)
        step = (vmax-vmin)/self.n_grid_split
        self.grids.append(np.arange(vmin, vmax+step, step))

    @property
    def dim(self):
        return len(self.values)
