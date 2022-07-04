from pathlib import Path
from typing import Optional, Union
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from paramopt import utils


class BaseGraph:

    PNG_PREFIX = ""

    def __init__(self) -> None:
        self.fig: Optional[Figure] = None

    def plot() -> None:
        raise NotImplementedError

    def show(self) -> None:
        if self.fig is None:
            raise ValueError("no figure to show")
        plt.show(block=False)

    def to_png(
        self, directory: Union[Path, str], label: Optional[str] = None
    ) -> None:
        if self.fig is None:
            raise ValueError("no figure to export")
        if label is None:
            label = utils.formatted_now()
        directory_ = Path(directory)
        directory_.mkdir(exist_ok=True, parents=True)
        self.fig.savefig((directory_/(self.PNG_PREFIX+label)).with_suffix(".png"))
