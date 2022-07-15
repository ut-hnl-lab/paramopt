from pathlib import Path
from typing import Optional, Union

from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from .. import utils


class BaseGraph:
    """Base class for visualizing numeric data."""
    PNG_PREFIX = ""

    def __init__(self) -> None:
        self.fig: Optional[Figure] = None

    def plot(self) -> None:
        """Plots data on a `matplotlib.pyplot.figure` instance.

        This method should be overridden by subclasess to have specific plotting
        program. The program must plot data on `self.fig`, which is used to save
        plots. `plt.close()` must be called at the beggining of the overriding
        method.
        """
        raise NotImplementedError

    def show(self) -> None:
        """Displays a graph on a separated window."""
        if self.fig is None:
            raise ValueError("no figure to show")
        plt.show()

    def to_png(
        self, directory: Union[Path, str], label: Optional[str] = None
    ) -> None:
        """Saves a graph in the form of png. `show()` method must be called in
        advance.

        Parameters
        ----------
        directory: pathlib.Path or str
            Directory where png files are output.
        label: str, optional
            Png files are saved as '[PNG_PREFIX][label].png'.
            If the label is set to `None`, current time is used instead.

        Raises
        ------
        ValueError
            Raises if `show()` method is not called before calling this method.
        """
        if self.fig is None:
            raise ValueError("no figure to export")
        if label is None:
            label = utils.formatted_now()
        directory_ = Path(directory)
        directory_.mkdir(exist_ok=True, parents=True)
        self.fig.savefig((directory_/(self.PNG_PREFIX+label)).with_suffix(".png"))
