import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D
import numpy as np

# Update rcParams for all visualizations
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'lines.linewidth': 2})
plt.rcParams.update({'axes.linewidth': 2})

plt.rcParams.update({'ytick.major.size': 8})
plt.rcParams.update({'ytick.major.width': 3})
plt.rcParams.update({'ytick.minor.width': 2})
plt.rcParams.update({'ytick.minor.size': 4})

plt.rcParams.update({'xtick.major.size': 8})
plt.rcParams.update({'xtick.major.width': 3})
plt.rcParams.update({'xtick.minor.width': 2})
plt.rcParams.update({'xtick.minor.size': 4})

plt.rcParams.update({'xtick.labelsize': 16})
plt.rcParams.update({'ytick.labelsize': 16})


def plot_ebs_output(error_budget, spectral_dict, parameter, values, int_times, force_linear=False, plot_stars=None,
                    fill=False, plot_by_spectype=True, save_dir='', save_name=''):
    """Plots the EBS parameter that was swept over as a function of calculated integration time.

    All observing scenarios are plotted on the same plot with different colors denoting the different stars and
    different linestyles indicating the inner, mid, or outer habitable zones. The plots will automatically scale to be
    semilog or loglog if one or both of the variables span more than three orders of magnitude.

    :param error_budget: ebs.error_budget.ErrorBudget
        Error Budget class from which results are desired to be plotted.
    :param spectral_dict: dict
        Dictionary of HIP numbers and spectral types of stars observed.
    :param parameter: str
        Parameter ovr which to plot.
    :param values: list or array
        Values of the parameter that were swept over.
    :param int_times: list or array
        Calculated integration times.
    :param force_linear: bool
        If true will force the plot to have linear scaling for both parameters.
    :param plot_stars: arr or list
        names of the stars to include in the plot. Must match the names used in the spectral_dict.
    :param fill: bool
        whether to fill in the space between the inner HZ and outer HZ exposure time values with color.
    :param plot_by_spectype: bool
        whether to plot each spectral type in its own panel or over-plot them on a single panel.
    :param save_dir: str
        Path to save the output plot.
    :param save_name: str
        Name to save the output plot under.
    :return: None
    """

    unique_types_to_plot = []
    for i, key in enumerate(spectral_dict.keys()):
        if key in plot_stars:
            unique_types_to_plot.append(spectral_dict[key][0])

    unique_types_to_plot = np.array(list(set(unique_types_to_plot)))
    num_types = len(unique_types_to_plot)

    if plot_by_spectype and num_types > 1:
        fig, axes = plt.subplots(num_types, 1, figsize=(14, 18), sharex=False)
    else:
        fig, axes = plt.subplots(1, 1, figsize=(16, 9))
        axes = [axes]

    if plot_by_spectype and num_types > 1:
        for i, type in enumerate(unique_types_to_plot):
            use_stars = [key for key in spectral_dict.keys() if
                         spectral_dict[key].startswith(unique_types_to_plot[i])]
            plot_panel(axes[i], error_budget, spectral_dict, values, int_times, force_linear=force_linear,
                       plot_stars=list(set(use_stars) & set(plot_stars)), fill=fill)
    else:
        plot_panel(axes[0], error_budget, spectral_dict, values, int_times, force_linear=force_linear,
                   plot_stars=plot_stars, fill=fill)

    plt.suptitle(f"Required Integration Time (hr, SNR={error_budget.exosims_pars_dict['observingModes'][0]['SNR']}) vs. "
                 f"{parameter.capitalize()}", fontsize=24)

    fig.supxlabel(f'{parameter.capitalize()}', fontsize=20)
    fig.supylabel('Integration Time (hours)', fontsize=20)

    for ax in axes:
        ax.set_xticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, save_name))
    plt.show()


def plot_panel(ax, error_budget, spectral_dict, values, int_times, force_linear=False, plot_stars=None, fill=False,
               plot_text=None, colors=np.array(['#01a075', '#00cc9e', '#6403fa', '#ff9400', '#cf5f00'])):
    """Plots the EBS parameter that was swept over as a function of calculated integration time.

    All observing scenarios are plotted on the same plot with different colors denoting the different stars and
    different linestyles indicating the inner, mid, or outer habitable zones. The plots will automatically scale to be
    semilog or loglog if one or both of the variables span more than three orders of magnitude.

    :param: matplotlib Axes object:
        axes on which to create the figure.
    :param error_budget: ebs.error_budget.ErrorBudget
        Error Budget class from which results are desired to be plotted.
    :param spectral_dict: dict
        Dictionary of HIP numbers and spectral types of stars observed.
    :param parameter: str
        Parameter ovr which to plot.
    :param values: list or array
        Values of the parameter that were swept over.
    :param int_times: list or array
        Calculated integration times.
    :param force_linear: bool
        If true will force the plot to have linear scaling for both parameters.
    :param plot_stars: arr or list
        names of the stars to include in the plot. Must match the names used in the spectral_dict.
    :param fill: bool
        whether to fill in the space between the inner HZ and outer HZ exposure time values with color.
    :param plot_text: str
        text to include as an inset on the plot.
    :param colors: arr or list
        hex colors or list of tuples of RGB values to use for each star. Defaults are color-blind friendly.
    :return: None
    """
    int_times_for_calc = np.copy(int_times)

    semilog_x, semilog_y, loglog = False, False, False

    if np.abs(np.nanmax(values)/np.nanmin(values)) > 100 and not force_linear:
        semilog_x = True

    if np.abs(np.nanmax(int_times_for_calc)/np.nanmin(int_times_for_calc)) > 100 and not force_linear:
        semilog_y = True

    if semilog_x and semilog_y:
        loglog = True

    use_idxs = []
    if not plot_stars:
        use_idxs = np.arange(len(spectral_dict.keys()))
    else:
        for i, star in enumerate(spectral_dict.keys()):
            if star in plot_stars:
                use_idxs.append(i)

    max = np.nanmax(int_times_for_calc)

    if semilog_y or loglog:
        y_lim = 24 * np.nanmin(int_times_for_calc) * 0.5, np.min([24 * max * 10, 1000])
        int_times_for_calc[np.isnan(int_times_for_calc)] = 24 * max * 100
    else:
        y_lim = 0, np.min([24 * max * 1.5, 1000])
        int_times_for_calc[np.isnan(int_times_for_calc)] = 24 * max * 2

    for i, (k, v) in enumerate(spectral_dict.items()):
        if i in use_idxs:
            txt = 'HIP%s\n%s, EEID=%imas' % (k, v, np.round(error_budget.eeid[i] * 1000))

            ax.scatter(values, 24 * int_times_for_calc[:, i, 0], c=colors[i], marker='o', s=70)
            ax.scatter(values, 24 * int_times_for_calc[:, i, 2], c=colors[i], marker='s', s=70)

            if fill:
                if np.all(np.isnan(int_times[:, i, 0])):
                    pass
                else:
                    upper_vals = []
                    lower_vals = []
                    for j, time in enumerate(int_times_for_calc[:, i, 0]):
                        upper_vals.append(np.max([24 * time, 24 * int_times_for_calc[:, i, 2][j]]))
                        lower_vals.append(np.min([24 * time, 24 * int_times_for_calc[:, i, 2][j]]))
                    ax.fill_between(values, lower_vals, upper_vals,
                                    color=colors[i], alpha=0.2)

            if loglog:
                ax.loglog(values, 24 * int_times_for_calc[:, i, 0], label=txt + 'inner', color=colors[i], linewidth=3)
                ax.loglog(values, 24 * int_times_for_calc[:, i, 2], linestyle='dashdot', label=txt + 'outer', color=colors[i], linewidth=3)
            elif semilog_x:
                ax.semilogx(values, 24 * int_times_for_calc[:, i, 0], label=txt + 'inner', color=colors[i], linewidth=3)
                ax.semilogx(values, 24 * int_times_for_calc[:, i, 2], linestyle='dashdot', label=txt + 'outer', color=colors[i], linewidth=3)
            elif semilog_y:
                ax.semilogy(values, 24 * int_times_for_calc[:, i, 0], label=txt + 'inner', color=colors[i], linewidth=3)
                ax.semilogy(values, 24 * int_times_for_calc[:, i, 2], linestyle='dashdot', label=txt + 'outer', color=colors[i], linewidth=3)
            else:
                ax.plot(values, 24 * int_times_for_calc[:, i, 0], label=txt + 'inner', color=colors[i], linewidth=3)
                ax.plot(values, 24 * int_times_for_calc[:, i, 2], linestyle='dashdot', label=txt + 'outer', color=colors[i], linewidth=3)
        else:
            pass

    # place a text box in upper left in axes coords
    if plot_text:
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax.text(0.23, 0.08, plot_text, transform=ax.transAxes, fontsize=20, verticalalignment='bottom',
                horizontalalignment='center', bbox=props)

    legend_elements = []
    for i, (k, v) in enumerate(spectral_dict.items()):
        if i in use_idxs:
            legend_elements.append((Line2D([0], [0], color=colors[i], lw=2, label=f'HIP {k}, {v}'), ))
        else:
            pass

    linestyles = ['solid', 'dashdot']
    labels = ['inner HZ', 'outer HZ']
    marker_styles = ['o', 's']

    for i in range(2):
        legend_elements.append((Line2D([], [], color='black', linestyle='None', marker=marker_styles[i]),
                                Line2D([0], [0], color='black', linestyle=linestyles[i], lw=2, label=labels[i])))

    labels = []

    for i in legend_elements:
        labels.append(i[0]._label)
        try:
            labels.append(i[1]._label)
        except IndexError:
            pass

    labels = [i for i in labels if i]
    ax.legend(handles=legend_elements, labels=labels, loc='upper left')
    ax.set_ylim(y_lim)
