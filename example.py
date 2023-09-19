import os 
import numpy as np
import yaml
import matplotlib.pyplot as plt
import json as js
from astropy.io import fits
import astropy.units as u
#import EXOSIMS.MissionSim as ems
from ebs import ebs


def nemati2020_vvc6():
    """
    An example of how to run <ebs> using sensitivities for the 
    "6-m off-axis segmented with VVC-6" case in the reference.

    Reference
    ---------
    Nemati et al. (2020) JATIS, Fig 26

    """
    # Instantiate the ErrorBudget object
    config = yaml.load('parameters.yml', Loader=yaml.SafeLoader)

    error_budget = ebs.ErrorBudget(input_dir=os.path.join(".", "inputs"),
                                   ref_json_filename="example_ref.json",
                                   pp_json_filename="example_pp.json",
                                   contrast_filename="example_contrast.csv",
                                   target_list=[32439, 77052, 79672, 26779, 113283],
                                   luminosity=[0.2615, -0.0788, 0.0391, -0.3209, -0.70],
                                   eeid=[0.07423, 0.06174, 0.07399, 0.05633, 0.05829],
                                   eepsr=[6.34e-11, 1.39e-10, 1.06e-10, 2.42e-10, 5.89e-10],
                                   exo_zodi=5*[3.0])
    
    # Specify Spectral Type of stars in target_list
    spectral_dict = {32439: 'F8V', 77052:'G5V', 79672: 'G2Va', 26779: 'K1V', 113283: 'K4Ve'}

    # Specify  WFE, WFS&C, and sensitivity input data
    num_spatial_modes = 13
    num_temporal_modes = 6
    num_angles = 5
    angles = np.linspace(0.055, 0.5, num_angles)  # Angular separation [as]
    wfe = (np.sqrt(1.72**2/num_temporal_modes) * np.ones((num_temporal_modes, num_spatial_modes)))  # [pm]
    wfsc_factor = 0.5*np.ones_like(wfe)  # Fractional residual WFE post-WFS&C 
    sensitivity = (np.array(num_angles*[3.21, 4.64, 4.51, 3.78, 5.19, 5.82, 10.6, 8.84, 9.09, 3.68, 9.33, 6.16, 0.745])
                   .reshape(num_angles, num_spatial_modes))  # [ppt/pm]
  
    # Specify multiple contrast and throughput scenarios that will be looped 
    # through to produce plots
    contrasts = np.array([1e-10, 2e-10, 5e-10])
    core_throughputs = np.array([0.08, 0.16, 0.32])
    dark_currents = np.array([1.5e-5, 3e-5, 6e-5])
    iwas = np.array([0.05, 0.07, 0.09])

    # The following two files are required by EXOSIMS
    contrast_filename = os.path.join(".", "inputs", error_budget.contrast_filename)
    throughput_filename = os.path.join(".", "inputs", "example_throughput.csv")
    
    # Specify output directory
    output_dir = os.path.join(".", "output")

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    prompt = "Loop over contrast [c], throughput [t], IWA [i], or dark noise [d]?  "
    sel = input(prompt)

    if sel == 'c':
        sweep = ebs.ParameterSweep(config, parameter='contrast', values=contrasts, error_budget=error_budget, wfe=wfe,
                                   sensitivity=sensitivity, wfsc_factor=wfsc_factor, fixed_contrast=None,
                                   fixed_throughput=core_throughputs[1], contrast_filename=contrast_filename,
                                   throughput_filename=throughput_filename, angles=angles,
                                   output_file_name='example_contrasts')

        result_dict = sweep.run_sweep()
        print(result_dict['int_time'])

        sweep.plot_output(spectral_dict, 'contrast', contrasts, result_dict['int_time'],
                          save_dir=output_dir, save_name= 't-vs-c.pdf')


    elif sel == 't':
        # Loop through core-throughput values while holding contrast at mid-value
        sweep = ebs.ParameterSweep(config, parameter='throughput', values=core_throughputs, error_budget=error_budget,
                                   wfe=wfe, sensitivity=sensitivity, wfsc_factor=wfsc_factor,
                                   fixed_contrast=contrasts[1], fixed_throughput=None,
                                   contrast_filename=contrast_filename, throughput_filename=throughput_filename,
                                   angles=angles, output_file_name='example_core_tputs')
        result_dict = sweep.run_sweep()
        print(result_dict['int_time'])

        sweep.plot_output(spectral_dict, 'throughput', core_throughputs, result_dict['int_time'],
                          save_dir=output_dir, save_name='t-vs-t.pdf')

    elif sel == 'd':
        # Loop through detector dark-noise values.  Note that the calling 
        # routine is different from the ones above because this is an 
        # EXOSIMS parameter (i.e. a paramter specified in the reference JSON 
        # file).

        sweep = ebs.ParameterSweep(config, parameter='dark', values=dark_currents, error_budget=error_budget, wfe=wfe,
                                   sensitivity=sensitivity, wfsc_factor=wfsc_factor, fixed_contrast=contrasts[1],
                                   fixed_throughput=core_throughputs[1], contrast_filename=contrast_filename,
                                   throughput_filename=throughput_filename, angles=angles,
                                   output_file_name='example_dark-currents')

        result_dict = sweep.run_sweep()
        print(result_dict['int_time'])

        sweep.plot_output(spectral_dict, 'dark', dark_currents, result_dict['int_time'],
                          save_dir=output_dir, save_name='t-vs-d.pdf')

    elif sel == 'i':
        # Loop through IWA values.  Note that the calling 
        # routine is different from the ones above because this is an 
        # EXOSIMS parameter (i.e. a parameter specified in the reference JSON
        # file).
        sweep = ebs.ParameterSweep(config, parameter='IWA', values=iwas, error_budget=error_budget, wfe=wfe,
                           sensitivity=sensitivity, wfsc_factor=wfsc_factor, fixed_contrast=contrasts[1],
                           fixed_throughput=core_throughputs[1], contrast_filename=contrast_filename,
                           throughput_filename=throughput_filename, angles=angles, output_file_name='example_IWA')

        result_dict = sweep.run_sweep()
        print(result_dict['int_time'])
        sweep.plot_output(spectral_dict, 'IWA', dark_currents, result_dict['int_time'],
                          save_dir=output_dir, save_name= 't-vs-IWA.pdf')


if __name__ == '__main__':
    nemati2020_vvc6()
