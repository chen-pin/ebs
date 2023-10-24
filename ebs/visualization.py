import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import matplotlib as mpl
from scipy.ndimage.filters import gaussian_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from matplotlib.lines import Line2D


# Update rcParams for all visualizations
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'lines.linewidth': 3})
plt.rcParams.update({'axes.linewidth': 3})
plt.rcParams.update({'ytick.major.size': 8})
plt.rcParams.update({'ytick.major.width': 2})
plt.rcParams.update({'xtick.major.size': 8})
plt.rcParams.update({'xtick.major.width': 2})
plt.rcParams.update({'xtick.labelsize': 'large'})
plt.rcParams.update({'ytick.labelsize': 'large'})


def plot_exptimes(param1, param2, exp_times, param1_name='', param2_name='', colormap='viridis', vmin=0,
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
    f = plt.figure()
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

    ax.set_xticks(np.arange(len(param1)), param1, fontsize=16)
    ax.set_yticks(np.arange(len(param2)), param2, fontsize=16)

    ax.set_xlabel(param1_name, fontsize=18)
    ax.set_ylabel(param2_name, fontsize=18)
    cmap.set_bad('black')
    im = ax.imshow(exp_times, cmap=cmap, norm=norm)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb = f.colorbar(im, cax=cax)
    cb.ax.set_ylabel('Exposure Time (hours)', fontsize=12)
    return ax


def plot_ebs_output(error_budget, spectral_dict, parameter, values, int_times, save_dir, save_name):
    fig, ax = plt.subplots(figsize=(16, 9))
    plt.title(f"Required Integration Time (hr, SNR=7) vs. {parameter.capitalize()}", fontsize=26)

    colors = ['#01a075','#00cc9e','#6403fa', '#ff9400', '#cf5f00']

    semilog_x, semilog_y, loglog = False, False, False

    if np.abs(np.nanmax(values)/np.nanmin(values)) > 1000:
        semilog_x = True

    if np.abs(np.nanmax(int_times)/np.nanmin(int_times)) > 1000:
        semilog_y = True

    if semilog_x and semilog_y:
        loglog = True

    for i, (k, v) in enumerate(spectral_dict.items()):
        txt = 'HIP%s\n%s, EEID=%imas' % (k, v, np.round(error_budget.eeid[i] * 1000))

        if loglog:
            plt.loglog(values, 24 * int_times[:, i, 0], label=txt + 'inner', color=colors[i], linewidth=3)
            plt.loglog(values, 24 * int_times[:, i, 1], linestyle='dashed', label=txt + 'mid', color=colors[i], linewidth=3)
            plt.loglog(values, 24 * int_times[:, i, 2], linestyle='dashdot', label=txt + 'outer', color=colors[i], linewidth=3)
        elif semilog_x:
            plt.semilogx(values, 24 * int_times[:, i, 0], label=txt + 'inner', color=colors[i], linewidth=3)
            plt.semilogx(values, 24 * int_times[:, i, 1], linestyle='dashed', label=txt + 'mid', color=colors[i], linewidth=3)
            plt.semilogx(values, 24 * int_times[:, i, 2], linestyle='dashdot', label=txt + 'outer', color=colors[i], linewidth=3)
        elif semilog_y:
            plt.semilogy(values, 24 * int_times[:, i, 0], label=txt + 'inner', color=colors[i], linewidth=3)
            plt.semilogy(values, 24 * int_times[:, i, 1], linestyle='dashed', label=txt + 'mid', color=colors[i], linewidth=3)
            plt.semilogy(values, 24 * int_times[:, i, 2], linestyle='dashdot', label=txt + 'outer', color=colors[i], linewidth=3)
        else:
            plt.plot(values, 24 * int_times[:, i, 0], label=txt + 'inner', color=colors[i], linewidth=3)
            plt.plot(values, 24 * int_times[:, i, 1], linestyle='dashed', label=txt + 'mid', color=colors[i], linewidth=3)
            plt.plot(values, 24 * int_times[:, i, 2], linestyle='dashdot', label=txt + 'outer', color=colors[i], linewidth=3)

    plt.xlabel(f'{parameter.capitalize()}', fontsize=24)
    plt.ylabel('Integration Time (hours)', fontsize=24)

    legend_elements = []
    for i, (k, v) in enumerate(spectral_dict.items()):
        legend_elements.append(Line2D([0], [0], color=colors[i], lw=3, label=f'HIP {k}, {v}'))
    linestyles = ['solid', 'dashed', 'dashdot']
    labels = ['inner HZ', 'mid HZ', 'outer HZ']
    for i in range(3):
        legend_elements.append(Line2D([0], [0], color='black', linestyle=linestyles[i], lw=3,
                                      label=labels[i]))

    ax.legend(handles=legend_elements, loc='upper left')

    plt.savefig(os.path.join(save_dir, save_name))
    plt.show()
    print(plt.rcParams.keys())