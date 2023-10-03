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
    # Load the config containing relevant parameters for running FREAK
    with open('parameters.yml', 'r') as config:
        config = yaml.load(config, Loader=yaml.FullLoader)

    # load relevant parameters for the ErrorBudget class from the config
    hip_numbers = [config['targets'][star]['HIP'] for star in config['targets']]
    luminosities = [config['targets'][star]['luminosity'] for star in config['targets']]
    eeids = [config['targets'][star]['eeid'] for star in config['targets']]
    eepsrs = [config['targets'][star]['eepsr'] for star in config['targets']]
    exo_zodis = [config['targets'][star]['exo_zodi'] for star in config['targets']]
    # Instantiate the ErrorBudget object
    error_budget = ErrorBudget(input_dir=config['paths']['input'],
                                   ref_json_filename=config['json_files']['ref_json'],
                                   pp_json_filename=config['json_files']['pp_json'],
                                   contrast_filename=config['input_files']['contrast'],
                                   target_list=hip_numbers,
                                   luminosity=luminosities,
                                   eeid=eeids,
                                   eepsr=eepsrs,
                                   exo_zodi=exo_zodis)
    
    # Specify Spectral Type of stars in target_list
    spectral_dict = {}
    for i, star in enumerate(config['targets']):
        spectral_dict[config['targets'][star]['HIP']] = config['targets'][star]['spec_type']

    # Specify  WFE, WFS&C, and sensitivity input data
    num_spatial_modes = config['num_spatial_modes']
    num_temporal_modes = config['num_temporal_modes']
    num_angles = config['angles']['num_angles']
    angles = np.linspace(config['angles']['start'], config['angles']['stop'], num_angles)  # Angular separation [as]
    wfe = (np.sqrt(1.72**2/num_temporal_modes) * np.ones((num_temporal_modes, num_spatial_modes)))  # [pm]
    wfsc_factor = 0.5*np.ones_like(wfe)  # Fractional residual WFE post-WFS&C 
    sensitivity = (np.array(num_angles*[3.21, 4.64, 4.51, 3.78, 5.19, 5.82, 10.6, 8.84, 9.09, 3.68, 9.33, 6.16, 0.745])
                   .reshape(num_angles, num_spatial_modes))  # [ppt/pm]
  
    # Specify multiple contrast and throughput scenarios that will be looped 
    # through to produce plots
    contrasts = np.array(config['iter_parameters']['contrasts'])
    core_throughputs = np.array(config['iter_parameters']['throughputs'])
    dark_currents = np.array(config['iter_parameters']['dark_currents'])
    iwas = np.array(config['iter_parameters']['iwas'])

    # The following two files are required by EXOSIMS
    contrast_filename = os.path.join(config['paths']['input'], error_budget.contrast_filename)
    throughput_filename = os.path.join(config['paths']['input'], config['input_files']['throughput'])
    
    # Specify output directory
    output_dir = config['paths']['output']

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    prompt = "Loop over contrast [c], throughput [t], IWA [i], or dark noise [d]?  "
    sel = input(prompt)

    if sel == 'c':
        sweep = ParameterSweep(config, parameter='contrast', values=contrasts, error_budget=error_budget, wfe=wfe,
                                   sensitivity=sensitivity, wfsc_factor=wfsc_factor, fixed_contrast=None,
                                   fixed_throughput=core_throughputs[1], contrast_filename=contrast_filename,
                                   throughput_filename=throughput_filename, angles=angles,
                                   output_file_name='example_contrasts')

        result_dict = sweep.run_sweep()
        print(result_dict['int_time'])

        sweep.plot_output(spectral_dict, 'contrast', contrasts, result_dict['int_time'],
                          save_dir=output_dir, save_name='t-vs-c.pdf')

    elif sel == 't':
        # Loop through core-throughput values while holding contrast at mid-value
        sweep = ParameterSweep(config, parameter='throughput', values=core_throughputs, error_budget=error_budget,
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

        sweep = ParameterSweep(config, parameter='dark', values=dark_currents, error_budget=error_budget, wfe=wfe,
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
        sweep = ParameterSweep(config, parameter='IWA', values=iwas, error_budget=error_budget, wfe=wfe,
                           sensitivity=sensitivity, wfsc_factor=wfsc_factor, fixed_contrast=contrasts[1],
                           fixed_throughput=core_throughputs[1], contrast_filename=contrast_filename,
                           throughput_filename=throughput_filename, angles=angles, output_file_name='example_IWA')

        result_dict = sweep.run_sweep()
        print(result_dict['int_time'])
        sweep.plot_output(spectral_dict, 'IWA', dark_currents, result_dict['int_time'],
                          save_dir=output_dir, save_name= 't-vs-IWA.pdf')


if __name__ == '__main__':
    nemati2020_vvc6()
