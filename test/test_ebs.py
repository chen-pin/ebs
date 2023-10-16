import os
import math
import json as js
import pytest as pt
import numpy as np
import astropy.units as u
from ebs.error_budget import ErrorBudget
import yaml


@pt.fixture
def obs():
    """
    Instantiate EXOSIMS object.

    """
    t = ErrorBudget(input_dir='./inputs'
                    , pp_json_filename="test_pp.json"
                    , contrast_filename="test_contrast.csv"
                    , target_list=[57443, 15457, 72659]
                    , luminosity=[-0.0737, -0.0669, -0.2572]
                    , eeid=[0.09858, 0.09981, 0.11012]
                    , eepsr=[1.37e-10, 1.35e-10, 2.09e-10]
                    , exo_zodi=3*[3.0])

    num_spatial_modes = 13
    num_temporal_modes = 1
    num_angles = 27

    config_fname = './inputs/test_parameters.yml'
    with open(config_fname, 'r') as config:
        config = yaml.load(config, Loader=yaml.FullLoader)

    config['paths']['input'] = './inputs'
    config['paths']['output'] = './outputs'

    config['num_spatial_modes'] = 13
    config['num_temporal_modes'] = 1
    config['angles']['num_angles'] = 27

    with open(config_fname, 'w') as output_config:
        yaml.dump(config, output_config)

    wfe = 1.72*np.random.rand(num_temporal_modes, num_spatial_modes)
    wfsc_factor = 0.5*np.random.rand(wfe.shape[0], wfe.shape[1])
    sensitivity = (np.array(num_angles*[3.21, 4.64, 4.51, 3.78, 5.19, 5.82, 10.6, 8.84, 9.09, 3.68, 9.33, 15.0, 0.745])
                 .reshape(num_angles, num_spatial_modes))

    t.run_etc(config, wfe, wfsc_factor, sensitivity, 'test_output.json', var_par=False)
    return t


def test_count_rates(obs):
    """
    Test count rates against values computed using formulas in Stark et al. 
    (2014) ApJ.  Stark et al. assumed that `r_sp` was negligible.

    """
    assert obs.C_p[1][1].value == pt.approx(0.0673, 0.2)
    assert obs.C_b[1][1].value == pt.approx(0.275, 0.2)
    assert obs.C_sr[1][1].value == pt.approx(0.0879, 0.2)
    assert obs.C_z[1][1].value == pt.approx(0.022, 0.1)
    assert obs.C_ez[1][1].value == pt.approx(0.166, 0.1)


def test_ppFact(obs):
    """
    Check self-consistency in array computations of ppFact.

    """
    num_temporal_modes = obs.wfe.shape[0]
    num_spatial_modes = obs.wfe.shape[1]
    num_angles = obs.sensitivity.shape[0]
    rss_wfe_residual = np.empty(num_spatial_modes)
    speckle_intensity = np.empty(num_angles)
    for s_mode in range(num_spatial_modes):
        rss_wfe_residual[s_mode] = np.sqrt(
                (obs.post_wfsc_wfe[:,s_mode]**2).sum()
                                          )
    for angle in range(num_angles):
        speckle_intensity[angle] = 1e-12*np.sqrt(
                (np.multiply(rss_wfe_residual, obs.sensitivity[angle,:])**2)
                .sum())
    assert (obs.delta_contrast == speckle_intensity).all()


def test_exposure_time(obs):
    """
    Check exposure-time calculations against values computed using Eq. 28 in 
    Nemati et al. (2020) JATIS.  

    """
    snr = obs.input_dict["observingModes"][0]["SNR"]
    print("snr = {}".format(snr))
    C_b = np.array(obs.C_b)
    C_p = np.array(obs.C_p)
    C_sp = np.array(obs.C_sp)
    int_time = np.array(obs.int_time)
    tau = (np.true_divide(snr**2 * C_b , C_p**2 - snr**2 * C_sp**2))/(24*3600)
    assert tau == pt.approx(int_time, 0.001)


def test_var_pars(obs):

    config = './inputs/test_parameters.yml'
    with open(config, 'r') as config:
        config = yaml.load(config, Loader=yaml.FullLoader)

    qe = [0.7, 0.8, 0.9]

    for value in qe:
        obs.run_etc(config, obs.wfe, obs.wfsc_factor, obs.sensitivity, var_par=True
                    , subsystem='scienceInstruments', name='QE', value=value, remove_temp_jsons=False)

        print(obs.input_dir )
        path = os.path.join(obs.input_dir, 'temp_' + 'QE' + '_' + str(value) + '.json')

        with open(path) as f:
            input_dict = js.load(f)
        assert input_dict['scienceInstruments'][0]['QE'] == value



