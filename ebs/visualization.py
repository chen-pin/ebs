import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import matplotlib as mpl
from scipy.ndimage.filters import gaussian_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable


class Visualizer:
    def __init__(self):
        self.solution = None

    def plot_exptimes(self, param1, param2, exp_times, param1_name='', param2_name='', colormap='viridis', vmin=0,
                      vmax=50, n_colors=3, filter=True, sigma=3, plot_contours=False, plot_pos=(1, 1, 1)):
        """

        :param param1: 1D array
            First plotting parameter.
        :param param2: 1D array
            Second plotting parameter.
        :param exp_times: 2D array
            Calculated exposure times.
        :param param1_name: str
            Name of the first parameter.
        :param param2_name: str
            Name of the second parameter.
        :param colormap: str
            Name of the matplotlib colormap to use.
        :param vmin: int or float
            Minimum value for colorbar.
        :param vmax: int or float
            Maximum value for colorbar
        :param n_colors: int
            number of discrete colors to use
        :param filter: bool
            If True will apply a gaussian filter to the esposure times before plotting. Recommended if
            plot_contours=True.
        :param sigma: int
            Sigma for the gaussian filter to use if filter=True.
        :param plot_contours: bool
            If True will plot contours of constant exposure time over the plot.
        :return:
        """
        n_colors = n_colors
        ax = f.add_subplot(plot_pos[0], plot_pos[1], plot_pos[2])
        divider = make_axes_locatable(ax)

        ref_map = mpl.cm.get_cmap(colormap)
        color_idxs = np.linspace(0, 1, n_colors)
        use_colors = [ref_map(i) for i in color_idxs][::-1]
        cmap = colors.ListedColormap(use_colors)
        norm = mpl.colors.BoundaryNorm(np.linspace(vmin, vmax, n_colors), cmap.N)

        if plot_contours:
            x = np.arange(len(param1))
            y = np.arange(len(param2))
            X, Y = np.meshgrid(x, y)
            Z = exp_times

            if filter:
                Z = gaussian_filter(Z, sigma)

            CS = ax.contour(X, Y, Z, colors='white')
            ax.clabel(CS, inline=True, fontsize=10)

        ax.set_xticks(np.arange(len(param1)), param1)
        ax.set_yticks(np.arange(len(param2)), param2)

        ax.set_xlabel(param1_name, fontsize=14)
        ax.set_ylabel(param2_name, fontsize=14)
        cmap.set_bad('black')
        im = ax.imshow(exp_times, cmap=cmap, norm=norm)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = f.colorbar(im, cax=cax)
        cb.ax.set_ylabel('Exposure Time (hours)', fontsize=12)
        return ax


if __name__ == '__main__':
    # random values to test visualization
    param1 = np.round(np.linspace(7, 10, 20), 1)
    param2 = np.round(np.linspace(0, 5, 20), 1)
    int_times = np.zeros((param1.shape[0], param1.shape[0]))


    for (x, y), val in np.ndenumerate(int_times):
        num = np.random.random(1)[0] * 50
        if num > 80:
            num = np.nan
        int_times[x, y] = x*y + num

    print(int_times)
    vis = Visualizer()
    f = plt.figure()
    ax = vis.plot_exptimes(param1, param2, int_times, param1_name='Contrast', param2_name='Dark Current',vmax=200, vmin=0,
                      n_colors=10, colormap='viridis', filter=False, plot_contours=False)
    print('done')

