import os
import math
import json as js
import pytest as pt
import numpy as np
import astropy.units as u
from ebs.error_budget import ErrorBudget
from ebs.log_pdf import inverse_bounded


@pt.fixture
def obs():
    """
    Instantiate EXOSIMS object.
    """
    t = ErrorBudget(config_file="./inputs/test_parameters.yml")
    t.initialize_for_exosims()
    t.run(clean_files=True)
    return t


@pt.fixture
def obs_mcmc():
    """
    Instantiate EXOSIMS object.
    """
    t = ErrorBudget(config_file="./inputs/mcmc_config.yml")
    t.initialize_for_exosims()
    t.run(clean_files=True)
    return t


def test_count_rates(obs):
    """
    Test count rates against values computed using formulas in Stark et al. 
    (2014) ApJ.  Stark et al. assumed that `r_sp` was negligible.

    """
    assert obs.C_p[1][1] == pt.approx(0.0673, 0.2)
    assert obs.C_b[1][1] == pt.approx(0.275, 0.2)
    assert obs.C_sr[1][1] == pt.approx(0.0879, 0.2)
    assert obs.C_z[1][1] == pt.approx(0.022, 0.1)
    assert obs.C_ez[1][1] == pt.approx(0.166, 0.1)


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
    assert [math.isclose(a, speckle_intensity[i], rel_tol=1.0e-12) for i, a in enumerate(obs.delta_contrast)]


def test_exposure_time(obs):
    """
    Check exposure-time calculations against values computed using Eq. 28 in 
    Nemati et al. (2020) JATIS.  

    """
    snr = obs.exosims_pars_dict["observingModes"][0]["SNR"]
    print("snr = {}".format(snr))
    C_b = np.array(obs.C_b)
    C_p = np.array(obs.C_p)
    C_sp = np.array(obs.C_sp)
    int_time = np.array(obs.int_time)
    tau = (np.true_divide(snr**2 * C_b , C_p**2 - snr**2 * C_sp**2))/(24*3600)
    tau[tau < 0] = 0
    int_time[~np.isfinite(int_time)] = 0
    assert tau == pt.approx(int_time, 0.001)


def test_var_pars(obs):
    qe = (0.7, 0.8, 0.9)
    output_filename = 'test_var_par_output'
    for value in qe:
        path = os.path.join(obs.temp_dir, 'temp.json')
        obs.run(subsystem='scienceInstruments', name='QE', value=value, clean_files=False)
        with open(path) as f:
            input_dict = js.load(f)
        assert input_dict['scienceInstruments'][0]['QE'] == value

def test_mcmc_delta_contrast(obs_mcmc):
    """
    Check self-consistency in array computations of delta_contrast.

    """
    states = obs_mcmc.initialize_walkers()
    for state in states:
        obs_mcmc.update_attributes_mcmc(state)
        num_temporal_modes = obs_mcmc.wfe.shape[0]
        num_spatial_modes = obs_mcmc.wfe.shape[1]
        num_angles = obs_mcmc.sensitivity.shape[0]
        rss_wfe_residual = np.empty(num_spatial_modes)
        speckle_intensity = np.empty(num_angles)
        for s_mode in range(num_spatial_modes):
            rss_wfe_residual[s_mode] = np.sqrt(
                    (obs_mcmc.post_wfsc_wfe[:,s_mode]**2).sum()
                                              )
        for angle in range(num_angles):
            speckle_intensity[angle] = 1e-12*np.sqrt(
                    (np.multiply(rss_wfe_residual
                        , obs_mcmc.sensitivity[angle,:])**2).sum())
        assert obs_mcmc.delta_contrast == pt.approx(speckle_intensity, 1E-12)
        obs_mcmc.clean_files()


def test_mcmc_ppFact(obs_mcmc):
    states = obs_mcmc.initialize_walkers()
    for state in states:
        obs_mcmc.update_attributes_mcmc(state)
        num_temporal_modes = obs_mcmc.wfe.shape[0]
        num_spatial_modes = obs_mcmc.wfe.shape[1]
        num_angles = obs_mcmc.sensitivity.shape[0]
        rss_wfe_residual = np.empty(num_spatial_modes)
        speckle_intensity = np.empty(num_angles)
        for s_mode in range(num_spatial_modes):
            rss_wfe_residual[s_mode] = np.sqrt(
                    (obs_mcmc.post_wfsc_wfe[:,s_mode]**2).sum()
                                              )
        for angle in range(num_angles):
            speckle_intensity[angle] = 1e-12*np.sqrt(
                    (np.multiply(rss_wfe_residual
                        , obs_mcmc.sensitivity[angle,:])**2).sum())
        for i, contrast in enumerate(obs_mcmc.contrast):
            if obs_mcmc.delta_contrast[i]/contrast > 1.0:
                assert obs_mcmc.ppFact[i] == 1.0
            else:
                assert (0.0 <= obs_mcmc.ppFact[i] ==
                        obs_mcmc.delta_contrast[i]/contrast)
        obs_mcmc.clean_files()


def test_mcmc_initialize_walkers(obs_mcmc):
    states = obs_mcmc.initialize_walkers()
    config = obs_mcmc.config
    nwalkers = config['mcmc']['nwalkers']
    ndim = 25
    upper_bounds = np.array([1e-5+0.5e-7, 5.05E-7, 5.05E-3, 5.05E-7, 5.05E-3
                             , 5.05E-7, 5.05E-3, 5.05E-7, 5.05E-3
                             , 5.05E-7, 5.05E-3, 5.05E-7, 5.05E-3] 
                             + 6*[2.01E-10] + 6*[0.15075])
    lower_bounds = np.array([1e-5-0.5e-7, 4.95E-7, 4.95E-3, 4.95E-7, 4.95E-3
                             , 4.95E-7, 4.95E-3, 4.95E-7, 4.95E-3
                             , 4.95E-7, 4.95E-3, 4.95E-7, 4.95E-3] 
                             + 6*[1.99E-10] + 6*[0.14925])
    print(f"upper_bounds:\n{upper_bounds}")
    print(f"lower_bounds:\n{lower_bounds}")

    assert states.shape == (nwalkers, ndim)
    for state in states:
        print(f"state:\n{state}")
        assert (lower_bounds <= state).all() and (state <= upper_bounds).all()
    obs_mcmc.clean_files()


def test_mcmc_update_attributes(obs_mcmc):
    state = obs_mcmc.initialize_walkers()[0]
    print(f"STATE: {state}")
    obs_mcmc.update_attributes_mcmc(state)
    assert obs_mcmc.idark == state[0]
    indices = (np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
               , np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]))
    assert (obs_mcmc.wfsc_factor[indices] == state[1:13]).all()
    print(f"CONTRAST:  {obs_mcmc.contrast}")
    assert (obs_mcmc.contrast == state[13:19]).all()
    assert (obs_mcmc.throughput == state[19:]).all()
    obs_mcmc.clean_files()


def test_mcmc_log_prior(obs_mcmc):
    all_true_val = [2E-6] + 12*[0.5] + 6*[5E-10] + 6*[0.15]
    one_false_val = [2E-4] + 12*[0.5] + 6*[5E-10] + 6*[0.15]
    assert obs_mcmc.log_prior(all_true_val) == 0.0
    assert obs_mcmc.log_prior(one_false_val) == -np.inf
    obs_mcmc.clean_files()


def test_mcmc_log_probability(obs_mcmc):
    obs_mcmc.initialize_for_exosims()
    accept_state = ([1E-5] + [5E-7, 5E-3, 5E-7, 5E-3, 5E-7, 5E-3
                                      , 5E-7, 5E-3, 5E-7, 5E-3, 5E-7, 5E-3] 
                                      + 6*[2E-10] + 6*[0.15])
    prior_reject_state = ([1E-5] + [5E-7, 5E-3, 5E-7, 5E-3, 5E-7, 5E-3
                                      , 5E-7, 5E-3, 5E-7, 5E-3, 5E-7, 5E-3] 
                                      + 6*[2E-10] + 6*[0.31])
    merit_reject_state = ([1E-5] + [0.99, 5E-3, 5E-7, 5E-3, 5E-7, 5E-3
                                      , 5E-7, 5E-3, 5E-7, 5E-3, 5E-7, 5E-3] 
                                      + 6*[2E-10] + 6*[0.15])
    states = [accept_state, prior_reject_state, merit_reject_state]
    for i, state in enumerate(states):
        log_prior = obs_mcmc.log_prior(state)
        log_merit = obs_mcmc.log_merit(state)[0]
        log_probability = log_prior + log_merit
        if i == 0:
            print(f"LOG PRIOR:  {log_prior}")
            print(f"LOG MERIT:  {log_merit}")
            print(f"LOG PROBABILITY:  {log_probability}")
            assert log_prior == 0.0
            assert log_merit == 0.0
            assert log_probability == 0.0
        elif i == 1:
            assert log_prior == -np.inf
            assert log_merit == 0.0
            assert log_probability == -np.inf
        else:
            assert log_prior == 0.0
            assert log_merit == -np.inf
            assert log_probability == -np.inf
        obs_mcmc.clean_files()


def test_mcmc_inverse_bounded():
    assert inverse_bounded(5.0, 4.99, 5.01) == np.log(1/5.0)
    assert inverse_bounded(4.988, 4.99, 5.01)  == -np.inf
    assert inverse_bounded(5.01001, 4.99, 5.01)  == -np.inf




