import argparse
import logging
from ebs.utils import read_csv
from ebs.error_budget import ParameterSweep, ErrorBudget
import numpy as np
import yaml
import os
from ebs.visualization import plot_ebs_output


def parse():
    # read in command line arguments
    parser = argparse.ArgumentParser(description='EBS Command Line Interface')
    parser.add_argument('param', type=str, help='parameter to sweep over')
    parser.add_argument('-sub', type=str, default=None, help='sub parameter to sweep over',
                        dest='sub_param')
    parser.add_argument('-c', type=str, help='config of parameter values', default='parameters.yml',
                        dest='config')
    return parser.parse_args()


def main():
    # command line arguments will overwrite the values from the config
    log = logging.getLogger()
    args = parse()
    config = args.config
    with open(config, 'r') as config:
        config = yaml.load(config, Loader=yaml.FullLoader)

    input_path = config['paths']['input']
    output_path = config['paths']['output']

    log.info(f'Running parameter sweep over {args.param}')

    parameter = args.param
    subparameter = args.sub_param
    values = config['iter_values']
    hip_numbers = [config['targets'][star]['HIP'] for star in config['targets']]
    eeids = [config['targets'][star]['eeid'] for star in config['targets']]
    eepsrs = [config['targets'][star]['eepsr'] for star in config['targets']]
    exo_zodis = [config['targets'][star]['exo_zodi'] for star in config['targets']]

    wfe = read_csv(os.path.join(input_path, config['input_files']['wfe']), skiprows=1)
    wfsc_factor = read_csv(os.path.join(input_path, config['input_files']['wfsc']), skiprows=1)
    sensitivity = read_csv(os.path.join(input_path, config['input_files']['sensitivity']), skiprows=1)

    error_budget = ErrorBudget(args.config)

    sweep = ParameterSweep(config, parameter=(parameter, subparameter), values=values, error_budget=error_budget, wfe=wfe,
                           sensitivity=sensitivity, wfsc_factor=wfsc_factor,
                           fixed_contrast=config['fixed_contrast'] if parameter != 'contrast' else None,
                           fixed_throughput=config['fixed_throughput'] if parameter != 'throughput' else None,
                           contrast_filename=os.path.join(input_path, config['input_files']['contrast']),
                           throughput_filename=os.path.join(input_path, config['input_files']['throughput']),
                           output_file_name='out')

    result_dict, error_budget = sweep.run_sweep()
    # Specify Spectral Type of stars in target_list
    spectral_dict = {}
    for i, star in enumerate(config['targets']):
        spectral_dict[config['targets'][star]['HIP']] = config['targets'][star]['spec_type']

    save_name = f'inttime_vs_{subparameter if subparameter else parameter}.pdf'
    plot_ebs_output(error_budget, spectral_dict, parameter if not subparameter else subparameter, values,
                    result_dict['int_time'], force_linear=config['plotting']['force_linear'],
                    plot_stars=config["plotting"]["plot_stars"], fill=config["plotting"]["fill"], save_dir=output_path,
                    save_name=save_name, plot_by_spectype=config["plotting"]["plot_by_spectype"])
