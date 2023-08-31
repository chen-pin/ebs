import os 
import numpy as np
import json as js
from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt
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
    num_spatial_modes = 13
    num_temporal_modes = 6
    num_angles = 27
    wfe = (0.65*np.ones((num_temporal_modes, num_spatial_modes)))
    wfsc_factor = 0.5*np.ones_like(wfe)
    sensitivity = 1.69*np.ones((num_angles, num_spatial_modes))
    # Now instantiate and run the calculation
    x = freak.ErrorBudget()
    x.run_etc(wfe, wfsc_factor, sensitivity)
    # View the results in "./../freak_out/`self.outupt_json_filename`
    return x



if __name__ == '__main__':
    nemati2020_vvc6()
