import os
import time
import datetime
import pickle
import numpy as np
import matplotlib.pyplot as plt
from ebs.error_budget import ErrorBudget
from ebs.logger import logger


class ParameterSweep:
    def __init__(self, config, parameter, values, error_budget):
        """

        Parameters
        ----------
        config: dict
            Config dict generated from the YAML.
        parameter: (str, str)
            (subsystem, name) of the parameter to be swept over.
        values: ndarray or list
            Values to sweep over.
        error_budget: ErrorBudget
            Initialized ErrorBudget object to use for the sweep. Only the
             parameter will be iterated over.
        """
        self.config = config
        self.input_dir = self.config["paths"]["input"]
        self.output_dir = self.config["paths"]["output"]

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.parameter, self.subparameter = parameter
        self.values = values
        self.result_dict = {}
        self.error_budget = error_budget
        self.error_budget.initialize_for_exosims()
        self.angles = self.error_budget.angles
        self.result_dict = {
            'C_p': np.empty((len(values), len(config['targets']), 3)),
            'C_b': np.empty((len(values), len(config['targets']), 3)),
            'C_sp': np.empty((len(values), len(config['targets']), 3)),
            'C_star': np.empty((len(values), len(config['targets']), 3)),
            'C_sr': np.empty((len(values), len(config['targets']), 3)),
            'C_z': np.empty((len(values), len(config['targets']), 3)),
            'C_ez': np.empty((len(values), len(config['targets']), 3)),
            'C_dc': np.empty((len(values), len(config['targets']), 3)),
            'C_rn': np.empty((len(values), len(config['targets']), 3)),
            'int_time': np.empty((len(values), len(config['targets']), 3))
        }

    def plot_output(self, spectral_dict, parameter, values, int_times, save_dir, save_name):
        """Saves and displays a results plot of intergation times vs. sweep parameter

        Parameters
        ----------
        spectral_dict: dict
        parameter: (str, str)
            (subsystem, name) of the parameter that was swept over.
        values: ndarray or list
            Values that were swept over.
        int_times: ndarray or list
            Resulting integration time.
        save_dir: str
            Directory where to save the final plot.
        save_name: str
            Name of the final plot save file.
        """
        logger.info("Plotting sweep output")

        plt.figure(figsize=(16, 9))
        plt.rcParams.update({'font.size': 12})
        plt.suptitle(f"Required Integration time (hr, SNR=7) vs. "
                     f"{parameter}", fontsize=20)

        for i, (k, v) in enumerate(spectral_dict.items()):
            plt.subplot(1, 5, i + 1)
            txt = ('HIP%s\n%s, EEID=%imas' %
                   (k, v, np.round(self.error_budget.eeid[i] * 1000)))

            plt.title(txt)
            plt.plot(values, 24 * int_times[:, i, 0], label='inner HZ')
            plt.plot(values, 24 * int_times[:, i, 1], label='mid HZ')
            plt.plot(values, 24 * int_times[:, i, 2], label='outer HZ')
            plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, save_name))
        plt.show()

    def run_sweep(self, save_output_dict=True):
        """Runs the single parameter sweep.

        Parameters
        ----------
        save_output_dict: bool
            If True saves the results dictionary to a pickle file.
        """
        logger.info(f"Running sweep on {self.parameter}")

        # 3 contrasts, 5 stars, 3 zones.
        for i, value in enumerate(self.values):
            if self.parameter == 'contrast':
                new_file = f"contrast_{value}.csv"
                np.savetxt(self.input_dir + "/" + new_file,
                           np.column_stack((self.angles, value
                                            * np.ones(len(self.angles))))
                           , delimiter=","
                           , header=('r_as,core_contrast')
                           , comments="")
                self.error_budget.contrast_filename = new_file
            elif self.parameter == 'throughput':
                new_file = f"throughput_{value}.csv"
                np.savetxt(self.input_dir + "/" + new_file
                           , np.column_stack((self.angles, value
                                              * np.ones(len(self.angles))))
                           , delimiter=","
                           , header=('r_as,core_thruput')
                           , comments="")
                self.error_budget.throughput_filename = new_file

            self.error_budget.run(subsystem=self.parameter,
                                  name=self.subparameter,
                                  value=value)

            for key in self.result_dict.keys():
                arr = self.result_dict[key]
                arr[i] = np.array(getattr(self.error_budget, key))
                self.result_dict[key] = arr

        if save_output_dict:
            time_stamp = time.time()
            dt_str = datetime.datetime.fromtimestamp(time_stamp).strftime("%Y-%m-%dT%H-%M-%S")
            if self.subparameter is not None:
                save_name = self.output_dir + '/' + f'{self.parameter}_{self.subparameter}_sweep_results_{dt_str}.pkl'
            else:
                save_name = self.output_dir + '/' + f'{self.parameter}_sweep_results_{dt_str}.pkl'
            with open(save_name, 'wb') as f:
                pickle.dump(self.result_dict, f)
        return self.result_dict, self.error_budget