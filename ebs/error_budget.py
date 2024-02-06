"""Exposure-time calculations based on coronagraphic input parameters

"""


import os, glob, time, shutil
import numpy as np
import json as js
import yaml 
from multiprocessing import Pool
from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt
import emcee
import EXOSIMS.MissionSim as ems
from copy import deepcopy
from ebs.utils import update_json
from ebs.utils import read_csv
import ebs.log_pdf as pdf


class ExosimsWrapper:
    def __init__(self, config):
        self.C_p = []
        self.C_b = []
        self.C_sp = []
        self.C_sr = []
        self.C_z = []
        self.C_ez = []
        self.C_dc = []
        self.C_rn = []
        self.C_star = []

        self.working_angles = config["working_angles"]
        hip_numbers = [str(config['targets'][star]['HIP']) for star in config['targets']]
        self.target_list = [f"HIP {n}" for n in hip_numbers]
        self.eeid = [config['targets'][star]['eeid'] for star in config['targets']]
        self.eepsr = [config['targets'][star]['eepsr'] for star in config['targets']]
        self.exo_zodi = [config['targets'][star]['exo_zodi'] for star in config['targets']]

        self.int_time = np.empty((len(self.target_list), len(self.working_angles))) * u.d

    def run_exosims(self, json_file):
        """
        Run EXOSIMS to generate results, including exposure times
        required for reaching specified SNR.

        """
        n_angles = len(self.working_angles)
        sim = ems.MissionSim(json_file, use_core_thruput_for_ez=False)
        for j, t in enumerate(self.target_list):
            if t not in sim.TargetList.Name:
                self.target_list[j] += " A"
                assert self.target_list[j] in sim.TargetList.Name
        sInds = np.array([np.where(sim.TargetList.Name == t)[0][0] for t in self.target_list])

        # assemble information needed for integration time calculation:

        # we have only one observing mode defined, so use that
        mode = sim.OpticalSystem.observingModes[0]

        # use the nominal local zodi and exozodi values
        fZ = sim.ZodiacalLight.fZ0

        # now we loop through the targets of interest and compute integration
        # times for each:
        for j, sInd in enumerate(sInds):
            # choose angular separation for coronagraph performance
            # this doesn't matter for a flat contrast/throughput, but
            # matters a lot when you have real performane curves
            # target planet deltaMag (evaluate for a range):
            WA = np.array(self.working_angles) * self.eeid[j]
            dMags = 5.0 * np.log10(np.array(self.working_angles)) - 2.5 * np.log10(self.eepsr[j])
            self.int_time[j] = sim.OpticalSystem.calc_intTime(
                sim.TargetList,
                [sInd] * n_angles,
                [fZ.value] * n_angles * fZ.unit,
                [self.exo_zodi[j] * sim.ZodiacalLight.fEZ0.value] * n_angles
                * sim.ZodiacalLight.fEZ0.unit,
                dMags,
                WA * u.arcsec,
                mode
            )
            counts = sim.OpticalSystem.Cp_Cb_Csp(
                sim.TargetList,
                [sInd] * n_angles,
                [fZ.value] * n_angles * fZ.unit,
                [self.exo_zodi[j] * sim.ZodiacalLight.fEZ0.value] * n_angles
                * sim.ZodiacalLight.fEZ0.unit,
                dMags,
                WA * u.arcsec,
                mode,
                True
            )
            self.C_p.append(counts[0])
            self.C_b.append(counts[1])
            self.C_sp.append(counts[2])
            self.C_sr.append(counts[3]["C_sr"])
            self.C_z.append(counts[3]["C_z"])
            self.C_ez.append(counts[3]["C_ez"])
            self.C_dc.append(counts[3]["C_dc"])
            self.C_rn.append(counts[3]["C_rn"])
            self.C_star.append(counts[3]["C_star"])

        return (self.int_time, self.C_p, self.C_b, self.C_sp, self.C_sr, self.C_z, self.C_ez, self.C_dc, self.C_rn,
                self.C_star)


class ErrorBudget(ExosimsWrapper):
    """Markov-chain-Monte-Carlo exploration of coronagraphic parameters.

    Attributes:
        config_file: Name of the configuration file.
    """

    def __init__(self, config_file):
        self.config_file = config_file
        with open(config_file, 'r') as config:
            self.config = yaml.load(config, Loader=yaml.FullLoader)
        super().__init__(self.config)

        self.input_dir = self.config["paths"]["input"]
        self.temp_dir = self.config["paths"]["temporary"]

        self.wfe = None
        self.wfsc_factor = None
        self.sensitivity = None
        self.post_wfsc_wfe = None
        self.angles = None
        self.contrast = None
        self.QE = None
        self.sread = None
        self.idark = None
        self.Rs = None
        self.lensSamp = None
        self.pixelNumber = None
        self.pixelSize = None
        self.optics = None
        self.BW = None
        self.IWA = None
        self.OWA = None 
        self.throughput = None
        self.SNR = None
        self.ppFact_filename = None

        self.contrast_filename = self.config["input_files"]["contrast"]
        self.throughput_filename = self.config["input_files"]["throughput"]

        self.exosims_pars_dict = None
        self.trash_can = []

    @property
    def delta_contrast(self):
        """
        Compute change in contrast due to residual WFE and assign the array to 
        `self.ppFact`. 

        Reference
        ---------
        - See <Post-Processing Factor> document for mathematical description

        """
        if (self.wfe is not None and self.wfsc_factor is not None 
            and self.sensitivity is not None and self.contrast is not None):
            self.post_wfsc_wfe = np.multiply(self.wfe, self.wfsc_factor)
            delta_contrast = np.empty(self.contrast.shape[0])
            for n in range(len(delta_contrast)):
                delta_contrast[n] = np.sqrt((np.multiply(self.sensitivity[n]
                                         , self.post_wfsc_wfe)**2).sum()
                                           ) 
            return 1E-12*delta_contrast
        else: 
            print("Need to assign wfe, wfsc_factor, sensitivity, " + 
                  "and contrast values before determining delta_contrast") 

    @property
    def ppFact(self):
        """
        Compute the post-processing factor and assign the array to 
        `self.ppFact`. 

        Reference
        ---------
        - See <Post-Processing Factor> document for mathematical description

        """
        ppFact = self.delta_contrast/self.contrast
        return np.where(ppFact>1.0, 1.0, ppFact)

    def load_json(self, json_file):
        """
        Load the JSON input file, which contains reference EXOSIMS parameters
        as well as WFE, sensitivity, and WFS&C parameters.  Assign parameter
        dictionary to `self.input_dict`.

        """
        with open(os.path.join(self.input_dir, json_file)) as input_json:
            input_dict = js.load(input_json)
        self.exosims_pars_dict = input_dict

    def load_csv_contrast(self):
        """
        Load CSV file containing contrast vs. angular separation values into
        ndarray and assign to `self.contrast`.

        """
        path = os.path.join(self.input_dir, self.contrast_filename)
        self.angles = np.genfromtxt(path, delimiter=',', skip_header=1)[:, 0]
        self.contrast = np.genfromtxt(path, delimiter=',', skip_header=1)[:, 1]

    def update_dict(self, thorughput_path, contrast_path, wfe, wfsc, sensitivity):

        self.exosims_pars_dict['starlightSuppressionSystems'][0]['core_thruput'] = thorughput_path
        self.exosims_pars_dict['starlightSuppressionSystems'][0]['core_contrast'] = contrast_path

        self.exosims_pars_dict['wfe'] = wfe.tolist()
        self.exosims_pars_dict['wfsc_factor'] = wfsc.tolist()
        self.exosims_pars_dict['sensitivity'] = sensitivity.tolist()

    def write_temp_json(self, filename='temp.json'):
        """
        Write `self.input_dict` to temporary JSON file for running EXOSIMS.

        """
        self.exosims_pars_dict["ppFact"] = self.ppFact_filename
        path = os.path.join(self.temp_dir, filename)
        with open(path, 'w') as f:
            js.dump(self.exosims_pars_dict, f)
        return path

    def write_ppFact_fits(self, trash=False):
        """
        Create FITS file of ppFact array with randomized filename.  

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
        """
        Create csv file of contrast or throughput array with randomized
        filename.  

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
        config = self.config

        contrast_path = os.path.join(self.input_dir, self.contrast_filename)
        throughput_path = os.path.join(self.input_dir, self.throughput_filename)

        self.angles = read_csv(filename=contrast_path, skiprows=1)[:, 0]
        self.contrast = read_csv(filename=contrast_path, skiprows=1)[:, 1]
        self.throughput = read_csv(filename=throughput_path, skiprows=1)[:, 1]
        self.wfe = read_csv(filename=os.path.join(self.input_dir, config['input_files']['wfe']), skiprows=1)
        self.wfsc_factor = read_csv(filename=os.path.join(self.input_dir, config['input_files']['wfsc']), skiprows=1)
        self.sensitivity = read_csv(filename=os.path.join(self.input_dir, config['input_files']['sensitivity']),
                                    skiprows=1)

        self.load_json(config["json_file"])

        self.update_dict(throughput_path, contrast_path, self.wfe, self.wfsc_factor, self.sensitivity)

        self.write_ppFact_fits(trash=True)

        self.exosims_pars_dict['ppFact'] = self.ppFact_filename

        for key in self.exosims_pars_dict['scienceInstruments'][0].keys():
            if key != 'optics' in dir(self):
                setattr(self, key, self.exosims_pars_dict['scienceInstruments'][0][key])
        for key in self.exosims_pars_dict['starlightSuppressionSystems'][0].keys():
            if key in dir(self):
                setattr(self, key, self.exosims_pars_dict['starlightSuppressionSystems'][0][key])

    def initialize_walkers(self):
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
        if name is not None:
            try:
                self.exosims_pars_dict[subsystem][0][name] = value
            except KeyError:
                self.exosims_pars_dict[name][0] = value

    def update_attributes_mcmc(self, values):
        for var_name in self.config['mcmc']['variables']:
            if var_name in dir(self):
                template = np.array(self.config['mcmc']['variables'][var_name]
                                               ['ini_pars']['center'])
                indices = np.where(np.isfinite(template))
                use_values, values = np.split(values, [indices[0].size])
                attr_val = getattr(self, var_name)
                if type(attr_val) == float or type(attr_val)==np.float64:
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
        self.update_attributes_mcmc(values)
        run_json = self.write_temp_json()
        self.trash_can.append(run_json)
        int_time, C_p, C_b, C_sp, C_sr, C_z, C_ez, C_dc, C_rn, C_star = self.run_exosims(run_json)
        self.clean_files()
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
        self.initialize_for_exosims()
        self.update_attributes(subsystem=subsystem, name=name, value=value)

        run_json = self.write_temp_json()

        self.run_exosims(run_json)

        self.trash_can.append(run_json)

        if clean_files:
            self.clean_files()

    def run_mcmc(self):
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
        for path in self.trash_can:
            if os.path.isfile(path):
                os.remove(path)


def log_probability(values, error_budget):
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


class ParameterSweep:
    def __init__(self, config, parameter, values, error_budget, output_file_name=''):
        '''

        :param config: dict
            The configuration for the ebs run.
        :param parameter: str
            The name of the parameter being swept over.
        :param values: array
            The values of the parameter to be swept over
        :param error_budget: ErrorBudget
            ErrorBudget object to use for the sweep.
        :param output_file_name: str
            name of the output file
        :param is_exosims_param: bool
            If True will feed the parameter into EXOSIMS to be swept over
        '''
        self.config = config
        self.input_dir = self.config["paths"]["input"]

        self.parameter, self.subparameter = parameter
        self.values = values
        self.result_dict = {}
        self.error_budget = error_budget
        self.error_budget.load_csv_contrast()
        self.output_file_name = output_file_name
        self.angles = self.error_budget.angles
        self.result_dict = {
            'C_p': np.empty((len(values), len(config['targets']), 3)),
            'C_b': np.empty((len(values), len(config['targets']), 3)),
            'C_sp': np.empty((len(values), len(config['targets']), 3)),
            'C_star': np.empty((len(values), len(config['targets']), 3)),
            'C_sr': np.empty((len(values), len(config['targets']), 3)),
            'C_z': np.empty((len(values), len(config['targets']), 3)),
            'C_ez': np.empty((len(values), len(config['targets']), 3)),
            'C_dc': np.empty((len(values), len(config['targets']), 3)),
            'C_rn': np.empty((len(values), len(config['targets']), 3)),
            'int_time': np.empty((len(values), len(config['targets']), 3))
        }

    def plot_output(self, spectral_dict, parameter, values, int_times, save_dir, save_name):
        plt.figure(figsize=(16, 9))
        plt.rcParams.update({'font.size': 12})
        plt.suptitle(f"Required Integration time (hr, SNR=7) vs. {parameter}", fontsize=20)

        for i, (k, v) in enumerate(spectral_dict.items()):
            plt.subplot(1, 5, i + 1)
            txt = 'HIP%s\n%s, EEID=%imas' % (k, v, np.round(self.error_budget.eeid[i] * 1000))

            plt.title(txt)
            plt.plot(values, 24 * int_times[:, i, 0], label='inner HZ')
            plt.plot(values, 24 * int_times[:, i, 1], label='mid HZ')
            plt.plot(values, 24 * int_times[:, i, 2], label='outer HZ')
            plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, save_name))
        plt.show()

    def run_sweep(self):
        # 3 contrasts, 5 stars, 3 zones
        for i, value in enumerate(self.values):
            if self.parameter == 'contrast':
                new_file = f"contrast_{value}.csv"
                np.savetxt(self.input_dir + "/" + new_file, np.column_stack((self.angles, value * np.ones(len(self.angles))))
                           , delimiter=",", header=('r_as,core_contrast')
                           , comments="")
                self.error_budget.contrast_filename = new_file
            if self.parameter == 'throughput':
                new_file = f"throughput_{value}.csv"
                np.savetxt(self.input_dir + "/" + new_file
                           , np.column_stack((self.angles, value * np.ones(len(self.angles))))
                           , delimiter=",", header=('r_as,core_thruput')
                           , comments="")
                self.error_budget.throughput_filename = new_file

            # TODO change mutable parameters in ErrorBudget class and remove this deep copy
            error_budget = deepcopy(self.error_budget)

            error_budget.run(subsystem=self.parameter, name=self.subparameter, value=value)

            for key in self.result_dict.keys():
                arr = self.result_dict[key]
                arr[i] = np.array(getattr(error_budget, key))
                self.result_dict[key] = arr

        return self.result_dict, error_budget
