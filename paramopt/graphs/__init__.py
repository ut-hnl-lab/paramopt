from matplotlib import pyplot as plt

from .base import BaseGraph
from .distribution import Distribution, _plot_process_1d, _plot_process_2d
from .transition import Transition, _plot_transition


plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.size'] = 16
plt.rcParams['legend.fontsize'] = 13
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['font.family'] = 'Times New Roman'
