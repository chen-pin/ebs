"""Exposure-time calculations based on coronagraphic input parameters

"""
import os, glob, time, shutil, datetime
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
from ebs.utils import read_csv
import ebs.log_pdf as pdf
import pickle


class ExosimsWrapper:
    """
    Takes in a config dict specifying desired targets and their corresponding earth equivalent
    insolation distances (eeid), Earth-equivalent planet-star flux ratios (eepsr) and exo-zodi levels.
    Main function is run_exosims which given an input json file returns the necessary integration times and various
    output count rates.

    Parameters
    ----------
    config: dict
        dictionary of configuration parameters.
    """
    def __init__(self, config):
        # pull relevant values from the config
        self.working_angles = config["working_angles"]
        hip_numbers = [str(config['targets'][star]['HIP']) for star in config['targets']]
        self.target_list = [f"HIP {n}" for n in hip_numbers]
        self.eeid = [config['targets'][star]['eeid'] for star in config['targets']]
        self.eepsr = [config['targets'][star]['eepsr'] for star in config['targets']]
        self.exo_zodi = [config['targets'][star]['exo_zodi'] for star in config['targets']]

        # intitialize output arrays
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

    def run_exosims(self, file_cleanup=True):
        """
        Run EXOSIMS to generate results, including exposure times 
        required for reaching specified SNR.  

        """
        # C_p = []
        # C_b = []
        # C_sp = []
        # C_sr = []
        # C_z = []
        # C_ez = []
        # C_dc = []
        # C_rn = []
        # C_star = []
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
        
        # assemble information needed for integration time calculation:
        
        # we have only one observing mode defined, so use that
        mode = sim.OpticalSystem.observingModes[0]
        
        # use the nominal local zodi and exozodi values
        fZ = sim.ZodiacalLight.fZ0
        
        # now we loop through the targets of interest and compute integration 
        # times for each:
        int_time = np.empty((len(target_list), n_angles))*u.d
        for j, sInd in enumerate(sInds):
            # choose angular separation for coronagraph performance
            # this doesn't matter for a flat contrast/throughput, but
            # matters a lot when you have real performane curves
            # target planet deltaMag (evaluate for a range):
            WA = np.array(wa_coefs)*eeid[j]
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
#            self.C_p.append(counts[0])
#
#            self.C_b.append(counts[1])
#            self.C_sp.append(counts[2])
#            self.C_sr.append(counts[3]["C_sr"])
#            self.C_z.append(counts[3]["C_z"])
#            self.C_ez.append(counts[3]["C_ez"])
#            self.C_dc.append(counts[3]["C_dc"])
#            self.C_rn.append(counts[3]["C_rn"])
#            self.C_star.append(counts[3]["C_star"])
            self.C_p[j] = counts[0].value
            self.C_b[j] = counts[1].value
            self.C_sp[j] = counts[2].value
            self.C_sr[j] = counts[3]["C_sr"].value
            self.C_z = counts[3]["C_z"].value
            self.C_ez = counts[3]["C_ez"].value
            self.C_dc= counts[3]["C_dc"].value
            self.C_rn = counts[3]["C_rn"].value
            self.C_star = counts[3]["C_star"].value
        if file_cleanup:
            self.clean_files()
        return self.int_time, self.C_p, self.C_b, self.C_sp, self.C_sr, self.C_z, self.C_ez, self.C_dc, self.C_rn, self.C_star

    # def run_exosims(self, json_file):
        # """ runs EXOSIMS using the parameters in the json_file.

        # Parameters
        # ----------
        # json_file: str
            # fully qualified file path to a JSON file containing all EXOSIMS input parameters.

        # Returns
        # -------
        # int_times: ndarray
            # integration times for each object at each working angle.
        # C_p: ndarray
            # Planet signal electron count rate (1/s).
        # C_b: ndarray
            # Background noise electron count rate (1/s).
        # C_sp: ndarray
            # Residual speckle spatial structure (systematic error) (1/s).
        # C_sr: ndarray
            # Starlight residual count rate (1/s).
        # C_z: ndarray
            # Local zodi count rate (1/s).
        # C_ez: ndarray
            # Exozodi count rate (1/s).
        # C_dc: ndarray
            # Dark current count rate (1/s).
        # C_rn: ndarray
            # Readout noise (1/s).
        # C_star: ndarray
            # Non-coronagraphic star count rate (1/s).
        # """
        # n_angles = len(self.working_angles)
        # sim = ems.MissionSim(json_file, use_core_thruput_for_ez=False)
        # for j, t in enumerate(self.target_list):
            # if t not in sim.TargetList.Name:
                # self.target_list[j] += " A"
                # assert self.target_list[j] in sim.TargetList.Name
        # sInds = np.array([np.where(sim.TargetList.Name == t)[0][0] for t in self.target_list])

        # # assemble information needed for integration time calculation:

        # # we have only one observing mode defined, so use that
        # mode = sim.OpticalSystem.observingModes[0]

        # # use the nominal local zodi and exozodi values
        # fZ = sim.ZodiacalLight.fZ0

        # # now we loop through the targets of interest and compute integration
        # # times for each:
        # for j, sInd in enumerate(sInds):
            # # choose angular separation for coronagraph performance
            # # this doesn't matter for a flat contrast/throughput, but
            # # matters a lot when you have real performane curves
            # # target planet deltaMag (evaluate for a range):
            # WA = np.array(self.working_angles) * self.eeid[j]
            # dMags = 5.0 * np.log10(np.array(self.working_angles)) - 2.5 * np.log10(self.eepsr[j])
            # self.int_time[j] = sim.OpticalSystem.calc_intTime(
                # sim.TargetList,
                # [sInd] * n_angles,
                # [fZ.value] * n_angles * fZ.unit,
                # [self.exo_zodi[j] * sim.ZodiacalLight.fEZ0.value] * n_angles
                # * sim.ZodiacalLight.fEZ0.unit,
                # dMags,
                # WA * u.arcsec,
                # mode
            # )
            # counts = sim.OpticalSystem.Cp_Cb_Csp(
                # sim.TargetList,
                # [sInd] * n_angles,
                # [fZ.value] * n_angles * fZ.unit,
                # [self.exo_zodi[j] * sim.ZodiacalLight.fEZ0.value] * n_angles
                # * sim.ZodiacalLight.fEZ0.unit,
                # dMags,
                # WA * u.arcsec,
                # mode,
                # True
            # )
            # self.C_p[j] = counts[0]
            # self.C_b[j] = counts[1]
            # self.C_sp[j] = counts[2]
            # self.C_sr[j] = counts[3]["C_sr"]
            # self.C_z[j] = counts[3]["C_z"]
            # self.C_ez[j] = counts[3]["C_ez"]
            # self.C_dc[j] = counts[3]["C_dc"]
            # self.C_rn[j] = counts[3]["C_rn"]
            # self.C_star[j] = counts[3]["C_star"]

        # return (self.int_time, self.C_p, self.C_b, self.C_sp, self.C_sr, self.C_z, self.C_ez, self.C_dc, self.C_rn,
                # self.C_star)


class ErrorBudget(ExosimsWrapper):
    """
    Exposure time calculator incorporating dynamical wavefront errors and
    WFS&C. Can also implement the Markov-chain-Monte-Carlo exploration of coronagraphic parameters.

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
        """Compute change in contrast due to residual WFE and assign the array to `self.ppFact`.

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
        """Compute the post-processing factor and assign the array to `self.ppFact`.

        Reference
        ---------
        - See <Post-Processing Factor> document for mathematical description

        """
        ppFact = self.delta_contrast/self.contrast
        return np.where(ppFact>1.0, 1.0, ppFact)

#    def load_json(self, json_file):
#        """
#        Load the JSON input file, which contains reference EXOSIMS parameters
#        as well as WFE, sensitivity, and WFS&C parameters.  Assign parameter
#        dictionary to `self.input_dict`.
#        """
#        with open(os.path.join(self.input_dir, json_file)) as input_json:
#            input_dict = js.load(input_json)
#        self.exosims_pars_dict = input_dict

    def load_csv_contrast(self):
        """
        Load CSV file containing contrast vs. angular separation values into
        ndarray and assign to `self.contrast' and `self.angles'.
        """
        path = os.path.join(self.input_dir, self.contrast_filename)
        self.angles = np.genfromtxt(path, delimiter=',', skip_header=1)[:, 0]
        self.contrast = np.genfromtxt(path, delimiter=',', skip_header=1)[:, 1]

#    def update_dict(self, throughput_path, contrast_path, wfe, wfsc, sensitivity):
#        """Updates self.exosims_pars_dict with the appropriate values from the input CSV files.
#
#        Parameters
#        ----------
#        throughput_path: str
#            fully qualified file path to the throughput CSV.
#        contrast_path: str
#            fully qualified file path to the contrast CSV.
#        wfe: list or numpy array
#            wavefront error values.
#        wfsc: list or numpy array
#            wavefront sensing and control values.
#        sensitivity: list or numpy array
#            sensitivity values.
#        """
#
#        self.exosims_pars_dict['starlightSuppressionSystems'][0]['core_thruput'] = throughput_path
#        self.exosims_pars_dict['starlightSuppressionSystems'][0]['core_contrast'] = contrast_path
#
#        self.exosims_pars_dict['wfe'] = wfe.tolist()
#        self.exosims_pars_dict['wfsc_factor'] = wfsc.tolist()
#        self.exosims_pars_dict['sensitivity'] = sensitivity.tolist()

#    def write_temp_json(self, filename='temp.json'):
#        """Writes a temporary JSON file with the current values in self.exosims_pars_dict.
#
#        Intended to be the input JSON for running EXOSIMS.
#
#        Parameters
#        ----------
#        filename: str
#            name of the temporary JSON file
#
#        Returns
#        -------
#        path: str
#            fully qualified path to the temporary JSON file.
#        """
#        self.exosims_pars_dict["ppFact"] = self.ppFact_filename
#        path = os.path.join(self.temp_dir, filename)
#        with open(path, 'w') as f:
#            js.dump(self.exosims_pars_dict, f)
#        return path

    def write_ppFact_fits(self, trash=False):
        """Writes the post-processing factor to a FITS file to be saved in self.temp_dir.

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

        This then allows these files to be saved for future reference or application.

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
        """Intitializes the EXOSIMS parameter dict with values from the config and reference JSON."""

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

#        self.load_json(config["json_file"])
        self.exosims_pars_dict = config['initial_exosims']

#        self.update_dict(throughput_path, contrast_path, self.wfe, self.wfsc_factor, self.sensitivity)

        self.write_ppFact_fits(trash=True)

        self.exosims_pars_dict['ppFact'] = self.ppFact_filename
        self.exosims_pars_dict['cherryPickStars'] = self.target_list
        self.exosims_pars_dict['starlightSuppressionSystems'][0]\
            ['core_contrast'] = contrast_path
        self.exosims_pars_dict['starlightSuppressionSystems'][0]\
            ['core_thruput'] = throughput_path

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
#        run_json = self.write_temp_json()
#        self.trash_can.append(run_json)
        int_time, C_p, C_b, C_sp, C_sr, C_z, C_ez, C_dc, C_rn, C_star = self.run_exosims()
#        self.clean_files()
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

        Updates the variable defined by the subsystem and name with the given value before
        generating all the necessary files to run EXOSIMS as specified by the config and input
        CSV files. If no subsystem or name is given then EXOSIMS is just run with the input CSV files and
        config.

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

#        run_json = self.write_temp_json()

        self.run_exosims()

#        self.trash_can.append(run_json)

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


class ParameterSweep:
    def __init__(self, config, parameter, values, error_budget):
        """

        Parameters
        ----------
        config: dict
            Config dict generated from the YAML.
        parameter: (str, str)
            (subsystem, name) of the parameter to be swept over.
        values: ndarray or list
            Values to sweep over.
        error_budget: ErrorBudget
            Initialized ErrorBudget object to use for the sweep. Only the parameter will be iterated over.
        """
        self.config = config
        self.input_dir = self.config["paths"]["input"]
        self.output_dir = self.config["paths"]["output"]

        self.parameter, self.subparameter = parameter
        self.values = values
        self.result_dict = {}
        self.error_budget = error_budget
        self.error_budget.load_csv_contrast()
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
        """Saves and displays a results plot of intergation times vs. sweep parameter

        Parameters
        ----------
        spectral_dict: dict
        parameter: (str, str)
            (subsystem, name) of the parameter that was swept over.
        values: ndarray or list
            Values that were swept over.
        int_times: ndarray or list
            Resulting integration time.
        save_dir: str
            Directory where to save the final plot.
        save_name: str
            Name of the final plot save file.
        """
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

    def run_sweep(self, save_output_dict=True):
        """Runs the single parameter sweep

        Parameters
        ----------
        save_output_dict: bool
            If True saves the results dictionary to a pickle file.
        """
        # 3 contrasts, 5 stars, 3 zones
        for i, value in enumerate(self.values):
            if self.parameter == 'contrast':
                new_file = f"contrast_{value}.csv"
                np.savetxt(self.input_dir + "/" + new_file, np.column_stack((self.angles, value * np.ones(len(self.angles))))
                           , delimiter=",", header=('r_as,core_contrast')
                           , comments="")
                self.error_budget.contrast_filename = new_file
            elif self.parameter == 'throughput':
                new_file = f"throughput_{value}.csv"
                np.savetxt(self.input_dir + "/" + new_file
                           , np.column_stack((self.angles, value * np.ones(len(self.angles))))
                           , delimiter=",", header=('r_as,core_thruput')
                           , comments="")
                self.error_budget.throughput_filename = new_file

            self.error_budget.run(subsystem=self.parameter, name=self.subparameter, value=value)

            for key in self.result_dict.keys():
                arr = self.result_dict[key]
                arr[i] = np.array(getattr(self.error_budget, key))
                self.result_dict[key] = arr

        if save_output_dict:
            time_stamp = time.time()
            dt_str = datetime.datetime.fromtimestamp(time_stamp).strftime("%Y-%m-%dT%H-%M-%S")
            if self.subparameter is not None:
                save_name = self.output_dir + '/' + f'{self.parameter}_{self.subparameter}_sweep_results_{dt_str}.pkl'
            else:
                save_name = self.output_dir + '/' + f'{self.parameter}_sweep_results_{dt_str}.pkl'
            with open(save_name, 'wb') as f:
                pickle.dump(self.result_dict, f)
        return self.result_dict, self.error_budget
