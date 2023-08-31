import os 
import numpy as np
import matplotlib.pyplot as plt
import json as js
from astropy.io import fits
import astropy.units as u
import EXOSIMS.MissionSim as ems
from freak import freak


def nemati2020_vvc6():
    """
    An example of how to run <freak> using sensitivities for the 
    "6-m off-axis segmented with VVC-6" case in the reference.

    Reference
    ---------
    Nemati et al. (2020) JATIS, Fig 26

    """
    # Instantiate the ErrorBudget object
    error_budget = freak.ErrorBudget(
                        input_dir=os.path.join(".", "inputs")
                      , ref_json_filename="example_ref.json"
                      , pp_json_filename="example_pp.json"
                      , output_json_filename="example_output.json"
                      , contrast_filename="example_contrast.csv"
                      , target_list=[32439, 77052, 79672, 26779, 113283]
                      , luminosity=[0.2615, -0.0788, 0.0391, -0.3209, -0.70]
                      , eeid=[0.07423, 0.06174, 0.07399, 0.05633, 0.05829]
                      , eepsr=[6.34e-11, 1.39e-10, 1.06e-10, 2.42e-10, 5.89e-10]
                          , exo_zodi=5*[3.0]
                          )
    
    # Specify  WFE, WFS&C, and sensitivity input data
    num_spatial_modes = 13
    num_temporal_modes = 6
    num_angles = 5
    angles = np.linspace(0.055, 0.5, num_angles)  # Angular separaton [as]
    wfe = (
            np.sqrt(1.72**2/num_temporal_modes)
            *np.ones((num_temporal_modes, num_spatial_modes))
          )  # [pm]
    wfsc_factor = 0.5*np.ones_like(wfe)  # Fractional residual WFE post-WFS&C 
    sensitivity = (
            np.array(
                      num_angles*[3.21, 4.64, 4.51, 3.78, 5.19, 5.82, 10.6, 8.84
                      , 9.09, 3.68, 9.33, 6.16, 0.745])
                      .reshape(num_angles, num_spatial_modes)
                  )  # [ppt/pm]
  
    # Specify multiple contrast and throughput scenarios that will be looped 
    # through to produce plots
    contrasts = np.array([1e-10, 2e-10, 5e-10])
    core_throughputs = np.array([0.08, 0.16, 0.32])
    # The following two files are required by EXOSIMS
    contrast_filename = os.path.join(".", "inputs", "example_contrast.csv") 
    throughput_filename = os.path.join(".", "inputs", "example_throughput.csv")
    
    # Create arrays to hold output
    C_p = np.empty((3, 5, 3))  # Planet count rate [1/s]
    C_b = np.empty((3, 5, 3))  # Total background count rate 
    C_sp = np.empty((3, 5, 3))  # Post-subtraction residual speckle 
                              # (systematic error)
                        # count rate
    C_star = np.empty((3, 5, 3))  # Non-coronagraphic stellar count rate
    C_sr = np.empty((3, 5, 3))  # Residual starlight count rate
    C_z = np.empty((3, 5, 3))  # Local zodi count rate
    C_ez = np.empty((3, 5, 3))  # Exo-zodi count rate
    C_dc = np.empty((3, 5, 3))  # Dark current count rate
    C_rn = np.empty((3, 5, 3))  # Read noise count rate
    int_time = np.empty((3, 5, 3))  # Required integration time to reach SNR  

    #  Loop through contrast values while holding core throughput at mid-value
    for k, contrast in enumerate(contrasts):
        np.savetxt(contrast_filename
                   , np.column_stack((angles, contrast*np.ones(num_angles)))
                   , delimiter=",", header=('r_as,core_contrast')
                   , comments="")
        np.savetxt(throughput_filename
                   , np.column_stack((angles
                                    , core_throughputs[1]*np.ones(num_angles)))
                   , delimiter=",", header=('r_as,core_thruput')
                   , comments="")
        error_budget.run_etc(wfe, wfsc_factor, sensitivity)
        C_p[k] = np.array(error_budget.C_p)
        C_b[k] = np.array(error_budget.C_b)
        C_sp[k] = np.array(error_budget.C_sp)
        C_star[k] = np.array(error_budget.C_star)
        C_sr[k] = np.array(error_budget.C_sr)
        C_z[k] = np.array(error_budget.C_z)
        C_ez[k] = np.array(error_budget.C_ez)
        C_dc[k] = np.array(error_budget.C_dc)
        C_rn[k] = np.array(error_budget.C_rn)
        int_time[k] = np.array(error_budget.int_time)
    print(int_time)

    #  Loop through core-throughput values while holding contrast at mid-value
    for t, throughput in enumerate(core_throughputs):
        np.savetxt(throughput_filename
                   , np.column_stack((angles
                                      , contrasts[1]*np.ones(num_angles)))
                   , delimiter=",", header=('r_as,contrast_thruput')
                   , comments="")
        np.savetxt(throughput_filename
                   , np.column_stack((angles
                                    , throughput*np.ones(num_angles)))
                   , delimiter=",", header=('r_as,core_thruput')
                   , comments="")
        error_budget.run_etc(wfe, wfsc_factor, sensitivity)
        C_p[t] = np.array(error_budget.C_p)
        C_b[t] = np.array(error_budget.C_b)
        C_sp[t] = np.array(error_budget.C_sp)
        C_star[t] = np.array(error_budget.C_star)
        C_sr[t] = np.array(error_budget.C_sr)
        C_z[t] = np.array(error_budget.C_z)
        C_ez[t] = np.array(error_budget.C_ez)
        C_dc[t] = np.array(error_budget.C_dc)
        C_rn[t] = np.array(error_budget.C_rn)
        int_time[t] = np.array(error_budget.int_time)
    print(int_time)



if __name__ == '__main__':
    nemati2020_vvc6()
