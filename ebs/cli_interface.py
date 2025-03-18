import argparse
from ebs.logger import logger
from ebs.error_budget import ErrorBudget
from ebs.parameter_sweep import ParameterSweep
import yaml
from ebs.visualization import plot_ebs_output


def parse():
    # Read in command line arguments.
    parser = argparse.ArgumentParser(description='EBS Command Line Interface')
    parser.add_argument('param',
                        type=str,
                        help='parameter to sweep over')
    parser.add_argument('-sub',
                        type=str,
                        default=None,
                        help='sub parameter to sweep over',
                        dest='sub_param')
    parser.add_argument('-c',
                        type=str,
                        help='config of parameter values',
                        default='parameters.yml',
                        dest='config')
    return parser.parse_args()


def main():
    # Command line arguments will overwrite the values from the config.
    log = logging.getLogger()
    args = parse()
    config = args.config
    with open(config, 'r') as config:
        config = yaml.load(config, Loader=yaml.FullLoader)

    output_path = config['paths']['output']

    log.info(f'Running parameter sweep over {args.param}')

    parameter = args.param
    subparameter = args.sub_param
    values = config['iter_values']

    error_budget = ErrorBudget(args.config)

    sweep = ParameterSweep(config,
                           parameter=(parameter, subparameter),
                           values=values,
                           error_budget=error_budget)

    result_dict, error_budget = sweep.run_sweep()

    # Specify Spectral Type of stars in target_list.

    spectral_dict = {}
    for i, star in enumerate(config['targets']):
        spectral_dict[config['targets'][star]['HIP']] \
            = config['targets'][star]['spec_type']

    save_name = f'inttime_vs_{subparameter if subparameter else parameter}.pdf'
    use_param = parameter if not subparameter else subparameter

    plot_ebs_output(error_budget,
                    spectral_dict,
                    use_param,
                    values,
                    result_dict['int_time'],
                    force_linear=config['plotting']['force_linear'],
                    plot_stars=config["plotting"]["plot_stars"],
                    fill=config["plotting"]["fill"],
                    save_dir=output_path,
                    save_name=save_name,
                    plot_by_spectype=config["plotting"]["plot_by_spectype"])
