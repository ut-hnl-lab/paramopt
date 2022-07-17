from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.axis3d import Axis
import numpy as np


plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.size'] = 16
plt.rcParams['legend.fontsize'] = 13
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['grid.color'] = 'black'
plt.rcParams['grid.linewidth'] = 0.1


def _get_nomargin_coord_info(self, renderer):
        mins, maxs = np.array([
            self.axes.get_xbound(),
            self.axes.get_ybound(),
            self.axes.get_zbound(),
        ]).T
        centers = (maxs + mins) / 2.
        deltas = (maxs - mins) / 12.

        # mins = mins - deltas / 24.
        # maxs = maxs + deltas / 24.

        vals = mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]
        tc = self.axes.tunit_cube(vals, renderer.M)
        avgz = [tc[p1][2] + tc[p2][2] + tc[p3][2] + tc[p4][2]
                for p1, p2, p3, p4 in self._PLANES]
        highs = np.array([avgz[2*i] < avgz[2*i+1] for i in range(3)])

        return mins, maxs, centers, deltas, tc, highs


# Remove small margins in 3D plots
Axis._get_coord_info = _get_nomargin_coord_info
