"""
Input file for running MCMC parameter exploration
"""

import numpy as np
from astropy import units as u


# First define the likelihood function

def likelihood(exposure_time, beta=1.0*u.day):
    """
    Return log likelihood
    
    """
    tau = np.array(exposure_time)
    tau_sum = np.where(np.isfinite(tau), tau, np.inf).sum()
    argument = (tau_sum*tau.unit/beta).value
    return np.exp(-argument**2)


# Define functions that will be used as priors for variables

def log_uniform(x, lower_bound, upper_bound):
    if lower_bound < x < upper_bound: 
        return 0.0
    return -np.inf

def log_gaussian(x, ctr, scale):
    return -((x-ctr)/scale)**2


# If necessary, define priors for WFE variables
wfsc_factor_prior = [
        4*[log_uniform, (1.0/(1.0+1e6), 1.0/(1.0+1000.0))]


# Define the dictionary of input parameters
pars = 
{
"par_id": ['wfsc_factor', 'exo_zodi', 'QE']
"is_exosims_par": [False, False, True]
"prior": {
    'wfsc_factor': [24*[

                   ]
         }

    


        
