import argparse
import logging
from ebs.utils import read_csv
from ebs.error_budget import ParameterSweep, ErrorBudget
import numpy as np
import yaml
import os


def parse():
    # read in command line arguments
    parser = argparse.ArgumentParser(description='EBS Command Line Interface')
    parser.add_argument('param', help='parameter to sweep over')
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

    values = config['iter_parameters'][str(args.param)]
    hip_numbers = [config['targets'][star]['HIP'] for star in config['targets']]
    luminosities = [config['targets'][star]['luminosity'] for star in config['targets']]
    eeids = [config['targets'][star]['eeid'] for star in config['targets']]
    eepsrs = [config['targets'][star]['eepsr'] for star in config['targets']]
    exo_zodis = [config['targets'][star]['exo_zodi'] for star in config['targets']]

    wfe = read_csv(os.path.join(input_path, config['input_files']['wfe']))
    wfsc_factor = read_csv(os.path.join(input_path, config['input_files']['wfsc']))
    sensitivity = read_csv(os.path.join(input_path, config['input_files']['sensitivity']))
    num_angles = config['angles']['num_angles']
    angles = np.linspace(config['angles']['start'], config['angles']['stop'], num_angles)

    error_budget = ErrorBudget(input_dir=config['paths']['input'],
                               pp_json_filename=config['json_files']['pp_json'],
                               contrast_filename=config['input_files']['contrast'],
                               target_list=hip_numbers, luminosity=luminosities, eeid=eeids, eepsr=eepsrs,
                               exo_zodi=exo_zodis)

    sweep = ParameterSweep(config, parameter=args.param, values=values, error_budget=error_budget, wfe=wfe,
                           sensitivity=sensitivity, wfsc_factor=wfsc_factor,
                           fixed_contrast=config['fixed_contrast'] if args.param != 'contrast' else None,
                           fixed_throughput=config['fixed_throughput'] if args.param != 'throughput' else None,
                           contrast_filename=os.path.join(input_path, config['input_files']['contrast']),
                           throughput_filename=os.path.join(input_path, config['input_files']['throughput']), angles=angles,
                           output_file_name='out')

    result_dict = sweep.run_sweep()
    # Specify Spectral Type of stars in target_list
    spectral_dict = {}
    for i, star in enumerate(config['targets']):
        spectral_dict[config['targets'][star]['HIP']] = config['targets'][star]['spec_type']

    sweep.plot_output(spectral_dict, args.param, values, result_dict['int_time'],
                      save_dir=output_path, save_name='cli_t-vs-c.pdf')