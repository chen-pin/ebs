"""Exposure-time calculations based on coronagraphic input parameters

"""
import os, time, shutil
import numpy as np
import yaml 
from multiprocessing import Pool
from astropy.io import fits
import astropy.units as u
import emcee
import EXOSIMS.MissionSim as ems
from copy import deepcopy
from ebs.utils import read_csv
import ebs.log_pdf as pdf


class ExosimsWrapper:
    """
    Takes in a config dict specifying desired targets and their corresponding
    earth equivalent insolation distances (eeid), Earth-equivalent planet-star
    flux ratios (eepsr) and exo-zodi levels. Main function is run_exosims which
    given an input json file returns the necessary integration times and
    various output count rates.

    Parameters
    ----------
    config: dict
        Dictionary of configuration parameters.
    """
    def __init__(self, config):
        # pull relevant values from the config
        # Pull relevant values from the config.
        self.working_angles = config["working_angles"]
        hip_numbers = [str(config['targets'][star]['HIP']) for star in config['targets']]
        self.target_list = [f"HIP {n}" for n in hip_numbers]
        self.eeid = [config['targets'][star]['eeid'] for star in config['targets']]
        self.eepsr = [config['targets'][star]['eepsr'] for star in config['targets']]
        self.exo_zodi = [config['targets'][star]['exo_zodi'] for star in config['targets']]

        # Initialize output arrays.
        self.C_p = np.empty((len(self.target_list), len(self.working_angles)))
        self.C_b = np.empty((len(self.target_list), len(self.working_angles)))
        self.C_sp = np.empty((len(self.target_list), len(self.working_angles)))
        self.C_sr = np.empty((len(self.target_list), len(self.working_angles)))
        self.C_z = np.empty((len(self.target_list), len(self.working_angles)))
        self.C_ez = np.empty((len(self.target_list), len(self.working_angles)))
        self.C_dc = np.empty((len(self.target_list), len(self.working_angles)))
        self.C_rn = np.empty((len(self.target_list), len(self.working_angles)))
        self.C_star = np.empty((len(self.target_list), len(self.working_angles)))
        self.int_time = np.empty((len(self.target_list), len(self.working_angles))) * u.d

    def run_exosims(self):
        """
        Run EXOSIMS to generate results, including exposure times required for
        reaching specified SNR.

        """
        wa_coefs = self.config["working_angles"]
        n_angles = len(wa_coefs)
        target_list = self.target_list
        eeid = self.eeid
        eepsr = self.eepsr
        exo_zodi = self.exo_zodi
        sim = ems.MissionSim(use_core_thruput_for_ez=False
                             , **deepcopy(self.exosims_pars_dict))
        
        print(f"TARGET_LIST:  {target_list}")
        print(f"TARGETLIST NAME: {sim.TargetList.Name}")
        sInds = np.array([np.where(sim.TargetList.Name == t)[0][0] for t 
                         in target_list])
        
        # Assemble information needed for integration time calculation.
        mode = sim.OpticalSystem.observingModes[0]
        
        # Use the nominal local zodi and exozodi values.
        fZ = sim.ZodiacalLight.fZ0
        
        # Loop through the targets of interest and compute integration
        # times for each.
        for j, sInd in enumerate(sInds):
            # Choose angular separation for coronagraph performance
            # this doesn't matter for a flat contrast/throughput, but
            # matters a lot when you have real performance curves.
            WA = np.array(wa_coefs)*eeid[j]
            # Target planet deltaMag (evaluate for a range).
            dMags = 5.0*np.log10(np.array(wa_coefs)) - 2.5*np.log10(eepsr[j])
            self.int_time[j] = sim.OpticalSystem.calc_intTime(
                sim.TargetList,
                [sInd] * n_angles,
                [fZ.value] * n_angles * fZ.unit,
                [exo_zodi[j]*sim.ZodiacalLight.fEZ0.value] * n_angles 
                    * sim.ZodiacalLight.fEZ0.unit,
                dMags,
                WA * u.arcsec,
                mode
            )
            counts = sim.OpticalSystem.Cp_Cb_Csp(
                sim.TargetList,
                [sInd] * n_angles,
                [fZ.value] * n_angles * fZ.unit,
                [exo_zodi[j]*sim.ZodiacalLight.fEZ0.value] * n_angles 
                    * sim.ZodiacalLight.fEZ0.unit,
                dMags,
                WA * u.arcsec,
                mode,
                True
            )

            self.C_p[j] = counts[0].value
            self.C_b[j] = counts[1].value
            self.C_sp[j] = counts[2].value
            self.C_sr[j] = counts[3]["C_sr"].value
            self.C_z = counts[3]["C_z"].value
            self.C_ez = counts[3]["C_ez"].value
            self.C_dc= counts[3]["C_dc"].value
            self.C_rn = counts[3]["C_rn"].value
            self.C_star = counts[3]["C_star"].value

        return self.int_time, self.C_p, self.C_b, self.C_sp, self.C_sr, self.C_z, self.C_ez, self.C_dc, self.C_rn, self.C_star


class ErrorBudget(ExosimsWrapper):
    """
    Exposure time calculator incorporating dynamical wavefront errors and
    WFS&C. Can also implement the Markov-chain-Monte-Carlo exploration of
    coronagraphic parameters.

    Parameters
    ----------
    config_file: str
        Name of the YAML configuration file.
    """
    def __init__(self, config_file):
        self.config_file = config_file
        with open(config_file, 'r') as config:
            self.config = yaml.load(config, Loader=yaml.FullLoader)
        super().__init__(self.config)

        self.input_dir = self.config["paths"]["input"]
        self.temp_dir = self.config["paths"]["temporary"]

        self.ref_contrast = self.config["reference_contrast"]
        self.wfe = None
        self.wfsc_factor = None
        self.sensitivity = None
        self.post_wfsc_wfe = None
        self.angles = None
        self.contrast = None
        self.ppFact_filename = None

        self.sensitivities_filename = self.config["input_files"]["sensitivity"]

        self.throughput_filename = self.config["initial_exosims"]["starlightSuppressionSystems"][0]["core_thruput"]
        self.contrast_filename = self.config["initial_exosims"]["starlightSuppressionSystems"][0]["core_contrast"]

        self.exosims_pars_dict = None
        self.trash_can = []

    @property
    def delta_contrast(self):
        """Compute change in contrast due to residual WFE.

        Assigns the value to the `ppFact` file.

        Reference
        ---------
        - See <Post-Processing Factor> document for mathematical description

        """
        if (self.wfe is not None and self.wfsc_factor is not None 
            and self.sensitivity is not None and self.contrast is not None):
            self.post_wfsc_wfe = np.multiply(self.wfe, self.wfsc_factor)
            delta_contrast = np.empty(self.sensitivity.shape[0])
            for n in range(len(delta_contrast)):
                delta_contrast[n] = np.sqrt((np.multiply(self.sensitivity[n], self.post_wfsc_wfe)**2).sum())
            return 1E-12*delta_contrast
        else: 
            print("Need to assign wfe, wfsc_factor, sensitivity, " + 
                  "and contrast values before determining delta_contrast") 

    @property
    def ppFact(self):
        """Compute the post-processing factor.

        Reference
        ---------
        - See <Post-Processing Factor> document for mathematical description

        """
        ppFact = self.delta_contrast/np.sqrt(self.contrast * self.ref_contrast)
        return np.where(ppFact>1.0, 1.0, ppFact)

    def load_sensitivities(self):
        """
        Load the angles and sensitivities from the sensitivities CSV into an array.
        """
        path = os.path.join(self.input_dir, self.sensitivities_filename)
        angles = np.genfromtxt(path, delimiter=',', skip_header=1)[:, 0]
        sensitivities = np.genfromtxt(path, delimiter=',', skip_header=1)[:, 1:]
        return angles, sensitivities

    def load_contrast(self):
        """
        Load the angles and sensitivities from the sensitivities CSV into an array.
        """
        path = os.path.join(self.input_dir, self.contrast_filename)
        angles = np.genfromtxt(path, delimiter=',', skip_header=1)[:, 0]
        contrasts = np.genfromtxt(path, delimiter=',', skip_header=1)[:, 1]
        return angles, contrasts

    def write_ppFact_fits(self, trash=True):
        """Writes the post-processing factor to a FITS file.

        File is saved in self.temp_dir

        Parameters
        ----------
        trash: bool
            If True, will add the FITS file to self.trash_can to be cleaned.
        """
        if self.angles is not None:

            rng = np.random.default_rng()
            random_string = str(rng.integers(1E9))
            filename = "ppFact_"+random_string+".fits"

            path = os.path.join(self.temp_dir, filename)

            with open(path, 'wb') as f:
                arr = np.vstack((self.angles, self.ppFact)).T
                fits.writeto(f, arr, overwrite=False)
                self.ppFact_filename = path

            if trash:
                self.trash_can.append(path)
        else:  
            print("Need to assign angle values to write ppFact FITS file")

    def write_csv(self, contrast_or_throughput):
        """Write the contrast or throughput values to a CSV file.

        This then allows these files to be saved for future reference or
        application.

        Parameters
        ----------
        contrast_or_throughput: str
            "contrast" or "throughput" - specifies which type of file to write.
        """
        if self.angles is not None:
            rng = np.random.default_rng()
            random_string = str(rng.integers(1E9))
            filename = contrast_or_throughput+'_'+random_string+".csv"
            path = os.path.join(self.temp_dir, filename)
            if contrast_or_throughput == 'contrast':
                arr = np.vstack((self.angles, self.contrast)).T
                self.contrast_filename = path
                np.savetxt(path, arr, delimiter=','
                           , header="r_as,core_contrast", comments='')
            if contrast_or_throughput == 'throughput':
                arr = np.vstack((self.angles, self.throughput)).T
                self.throughput_filename = path
                np.savetxt(path, arr, delimiter=',', header="r_as,core_thruput"
                           , comments='')
            self.trash_can.append(path)
        else:  
            print("Need to assign angle values to write CSV file")

    def initialize_for_exosims(self):
        """Initializes the EXOSIMS parameter dict with the config."""

        config = self.config

        self.wfe = read_csv(filename=os.path.join(self.input_dir, config['input_files']['wfe']), skiprows=1)
        self.wfsc_factor = read_csv(filename=os.path.join(self.input_dir, config['input_files']['wfsc_factor']), skiprows=1)
        self.angles, self.sensitivity = self.load_sensitivities()
        _, self.contrast = self.load_contrast()

        self.exosims_pars_dict = config['initial_exosims']

        if self.throughput_filename:
            throughput_path = os.path.join(self.input_dir, self.throughput_filename)
            self.throughput = read_csv(filename=throughput_path, skiprows=1)[:, 1]
            self.exosims_pars_dict['starlightSuppressionSystems'][0] \
                ['core_thruput'] = throughput_path

        if self.contrast_filename:
            contrast_path = os.path.join(self.input_dir, self.contrast_filename)
            self.contrast = read_csv(filename=contrast_path, skiprows=1)[:, 1]
            self.exosims_pars_dict['starlightSuppressionSystems'][0] \
                ['core_contrast'] = contrast_path

        self.write_ppFact_fits(trash=True)

        self.exosims_pars_dict['ppFact'] = self.ppFact_filename
        self.exosims_pars_dict['cherryPickStars'] = self.target_list

        for key in self.exosims_pars_dict['scienceInstruments'][0].keys():
            if key != 'optics' in dir(self):
                setattr(self, key, self.exosims_pars_dict['scienceInstruments'][0][key])
        for key in self.exosims_pars_dict['starlightSuppressionSystems'][0].keys():
            if key in dir(self):
                setattr(self, key, self.exosims_pars_dict['starlightSuppressionSystems'][0][key])

    def initialize_walkers(self):
        """

        Returns
        -------
        walker_pos:
        """
        center = []
        [center.append(np.array(val).ravel())  
         for var_name in self.config['mcmc']['variables'] 
         for val in self.config['mcmc']['variables'][var_name]['ini_pars']['center']]
        center = np.concatenate(center)
        center = center[np.where(np.isfinite(center))]
        spread = []
        [spread.append(np.array(val).ravel())  
         for var_name in self.config['mcmc']['variables'] 
         for val in self.config['mcmc']['variables'][var_name]['ini_pars']
                               ['spread']]
        spread = np.concatenate(spread)
        spread= spread[np.where(np.isfinite(spread))]
        ndim = center.shape[0]
        nwalkers = self.config['mcmc']['nwalkers']
        walker_pos = center + np.random.uniform(-spread/2.0, spread/2.0
                                                , (nwalkers, ndim))
        return walker_pos

    def update_attributes(self, subsystem=None, name=None, value=None):
        """Updates the EXOSIMS parameter dict for the subsystem and name with the value.

        For example if subsystem = scienceInstruments and name = QE then
        self.exosims_pars_dict["scienceInstruments"]["QE] will take on the value.

        Parameters
        ----------
        subsystem: str
            Name of subsystem to update.
        name: str
            Name of the variable to update.
        value: float or int or str or bool
            Value to update the variable to.
        """
        if name is not None:
            try:
                self.exosims_pars_dict[subsystem][0][name] = value
            except KeyError:
                self.exosims_pars_dict[name][0] = value

    def update_attributes_mcmc(self, values):
        """

        Parameters
        ----------
        values
        """
        for var_name in self.config['mcmc']['variables']:
            if var_name in dir(self):
                template = np.array(self.config['mcmc']['variables'][var_name]
                                               ['ini_pars']['center'])
                indices = np.where(np.isfinite(template))
                use_values, values = np.split(values, [indices[0].size])
                attr_val = getattr(self, var_name)
                if type(attr_val) == float or type(attr_val) == np.float64:
                    attr_val = use_values[0]
                else:
                    attr_val[indices] = use_values
                setattr(self, var_name, attr_val)
                if var_name == 'SNR':
                    self.exosims_pars_dict['observingModes'][0][var_name]\
                            = attr_val
                if var_name in ['QE', 'sread', 'idark', 'Rs', 'lenslSamp', 
                                'pixelNumber', 'pixelSize']:
                    self.exosims_pars_dict['scienceInstruments'][0][var_name]\
                            = attr_val
                if var_name in ['optics', 'BW', 'IWA', 'OWA']:
                    self.exosims_pars_dict['starlightSuppressionSystems'][0]\
                            ['var_name'] = attr_val
                if var_name in ['contrast', 'wfe', 'wfsc_factor'
                                , 'sensitivity']:
                    if var_name == 'contrast':
                        self.write_csv(var_name)
                        self.exosims_pars_dict['starlightSuppressionSystems']\
                                [0]['core_contrast'] = self.contrast_filename
                    self.write_ppFact_fits(trash=True)
                    self.exosims_pars_dict['ppFact'] = self.ppFact_filename
                if var_name == 'throughput':
                    self.write_csv(var_name)
                    self.exosims_pars_dict['starlightSuppressionSystems']\
                            [0]['core_thruput'] = self.throughput_filename
            else:
                print(var_name+" not found in attributes list")

    def log_prior(self, values):
        """

        Parameters
        ----------
        values

        Returns
        -------
        joint_prob:
        """
        joint_prob = 0.0
        counter = 0
        for i, var_name in enumerate(self.config['mcmc']['variables'].keys()):
            prior_ftn = np.array(
                    self.config['mcmc']['variables'][var_name]['prior_ftn']
                                )
            index = np.where(prior_ftn!='nan')
            all_args = [np.array(item) for item in 
                    self.config['mcmc']['variables'][var_name]['prior_args']
                        .values()]
            for m, row_index in enumerate(index[0]):
                if len(index) > 1:
                    col_index = index[1][m]
                    ftn = getattr(pdf, prior_ftn[row_index, col_index])
                    args = [page[row_index][col_index] for page in all_args]
                else:
                    ftn = getattr(pdf, prior_ftn[row_index])
                    args = [page[row_index] for page in all_args]
                prob = ftn(values[counter], *args)
                counter += 1
                if np.isinf(prob):
                    return prob
                else: 
                    joint_prob += prob
        return joint_prob 

    def log_merit(self, values):
        """

        Parameters
        ----------
        values

        Returns
        -------

        """
        self.update_attributes_mcmc(values)
        int_time, C_p, C_b, C_sp, C_sr, C_z, C_ez, C_dc, C_rn, C_star = self.run_exosims()
        if np.isnan(int_time.value).any():
            return -np.inf, int_time, C_p, C_b, C_sp, C_sr, C_z, C_ez, C_dc\
                   , C_rn, C_star
        else:
            ftn_name = self.config['mcmc']['merit']['ftn']
            args = [arg for arg in self.config['mcmc']['merit']['args']\
                    .values()]
            ftn = getattr(pdf, ftn_name)
            mean_int_time = np.array(int_time.value).mean() 
            return ftn(mean_int_time, *args), int_time, C_p, C_b, C_sp, C_sr, \
                    C_z, C_ez, C_dc, C_rn, C_star 

    def run(self, subsystem=None, name=None, value=None, clean_files=True):
        """Main method for running EBS not in MCMC mode.

        Updates the variable defined by the subsystem and name with the given
        value before generating all the necessary files to run EXOSIMS as
        specified by the config and input CSV files. If no subsystem or name
        is given then EXOSIMS is just run with the input CSV files and config.

        Parameters
        ----------
        subsystem: str
            Name of subsystem to update.
        name: str
            Name of the variable to update.
        value: float or int or str or bool
            Value to update the variable to.
        clean_files: bool
            If True will remove temporary intermediate files after they are used.
        """
        self.initialize_for_exosims()
        self.update_attributes(subsystem=subsystem, name=name, value=value)

        self.run_exosims()

        if clean_files:
            self.clean_files()

    def run_mcmc(self):
        """Main method for running EBS in MCMC mode.

        Returns
        -------
        sampler: emcee.EnsembleSampler
        """
        self.initialize_for_exosims()
        pos = self.initialize_walkers()
        nwalkers, ndim = pos.shape
        nsteps = self.config['mcmc']['nsteps']
        ntargets = len(self.config['targets'])
        if self.config['mcmc']['save']:
            dtype = [('int_time', float, (ntargets, 3))
                    , ('C_p', float, (ntargets, 3))
                    , ('C_b', float, (ntargets, 3))
                    , ('C_sp', float, (ntargets, 3))
                    , ('C_sr', float, (ntargets, 3))
                    , ('C_z', float, (ntargets, 3))
                    , ('C_ez', float, (ntargets, 3))
                    , ('C_dc', float, (ntargets, 3))
                    , ('C_rn', float, (ntargets, 3))
                    , ('C_star', float, (ntargets, 3)) ]
            time_stamp = time.strftime('%Y%m%dt%H%M%S')
            save_path = os.path.join(self.config['paths']['output']
                                     , 'saved_run_'+time_stamp)
            os.mkdir(save_path)
            if self.config['mcmc']['new_run']:
                backend = emcee.backends.HDFBackend(os.path.join(save_path
                                                    , 'backend.hdf'))
                backend.reset(nwalkers, ndim)
            elif self.config['mcmc']['new_run'] == False:
                pos = None
                backend = emcee.backends.HDFBackend(
                        self.config['mcmc']['previous_backend_path'])
            shutil.copy2(self.config_file, save_path)
            for key in self.config['input_files']:
                filename = self.config['input_files'][key]
                shutil.copy2(os.path.join(self.config['paths']['input']
                                          , filename), save_path)
        else:
            backend = None
            dtype = None
        if self.config['mcmc']['parallel']:
            os.environ["OMP_NUM_THREADS"] = "1"
            with Pool() as pool:
                sampler = emcee.EnsembleSampler(nwalkers, ndim
                            , log_probability, backend=backend, pool=pool
                            , args=[self], blobs_dtype=dtype)
                sampler.run_mcmc(pos, nsteps, progress=True, store=True)
        else:
            sampler = emcee.EnsembleSampler(nwalkers, ndim
                          , log_probability, backend=backend, args=[self]
                          , blobs_dtype=dtype)
            sampler.run_mcmc(pos, nsteps, progress=True, store=True)
        return sampler

    def clean_files(self):
        """Deletes all files in the trash_can"""
        for path in self.trash_can:
            if os.path.isfile(path):
                os.remove(path)


def log_probability(values, error_budget):
    """

    Parameters
    ----------
    values
    error_budget

    Returns
    -------

    """
    log_prior = error_budget.log_prior(values)
    if np.isinf(log_prior):
        return -np.inf, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0\
                , -1.0
    log_merit, int_time, C_p, C_b, C_sp, C_sr, C_z, C_ez, C_dc, C_rn, C_star \
            = error_budget.log_merit(values)
    if np.isinf(log_merit):
        return -np.inf, int_time, C_p, C_b, C_sp, C_sr, C_z, C_ez, C_dc, C_rn\
                , C_star
    log_probability = log_prior + log_merit
    return log_probability, int_time, C_p, C_b, C_sp, C_sr, C_z, C_ez, C_dc\
            , C_rn, C_star
