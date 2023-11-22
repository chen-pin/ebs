import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D
import numpy as np

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


def plot_ebs_output(error_budget, spectral_dict, parameter, values, int_times, force_linear=False, save_dir='',
                    save_name=''):
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
    :param save_dir: str
        Path to save the output plot.
    :param save_name: str
        Name to save the output plot under.
    :return: None

    """
    fig, ax = plt.subplots(figsize=(16, 9))
    plt.title(f"Required Integration Time (hr, SNR={error_budget.input_dict['observingModes'][0]['SNR']}) vs. "
              f"{parameter.capitalize()}", fontsize=26)

    colors = ['#01a075','#00cc9e','#6403fa', '#ff9400', '#cf5f00']

    semilog_x, semilog_y, loglog = False, False, False

    if np.abs(np.nanmax(values)/np.nanmin(values)) > 1000 and not force_linear:
        semilog_x = True

    if np.abs(np.nanmax(int_times)/np.nanmin(int_times)) > 1000 and not force_linear:
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