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
from ebs.utils import update_pp_json
from ebs.utils import read_csv
import ebs.log_pdf as pdf


class ErrorBudget(object):
    """
    Exposure time calculator incorporating dynamical wavefront errors and 
    WFS&C
    
    Parameters
    ----------
    input_dir : `os.path`
        Directory path where the above-listed input files reside
    output_dir : `os.path`
        Directory path where the output files will be saved
    pp_json_filename : str
        Name of JSON file that has WFE, WFS&C factors, and sensitivity 
        coefficients appended to the reference EXOSIMS parameters.
    contrast_filename : str
        Name of CSV file specifying contrast of the reference dark hole
        (i.e. contrast obtained on the reference star)
    target_list : list
        List of target-star HIP IDs
    eeid : list
        Earth-equivalent insolation distance values of the target stars 
        [as]
    eepsr : list
        Earth-equivalent planet-star flux ratio of the target stars
    exo_zodi : list
        Exozodi brightness of the target stars in units of EXOSIMS nominal 
        exo-zodi brightness `fEZ0` 

    Attributes
    ----------
    input_dict : dict
        Dictionary of parameters loaded from `pp_json_filename` file
    wfe : array 
        Wavefront changes specified in spatio-temporal bins, loaded from the 
        input JSON file, in pm units.  
        Dimensions:  (num_temporal_modes, num_spatial_modes).
    wfsc_factor : array
        Wavefront-change-mitigation factors, loaded from the input JSON file.  
        Values should be between 0 and 1.    
        Dimensions:  same as `wfe`
    sensitivity : array
        Coefficients of contrast sensitivity w.r.t. wavefront changes, 
        in ppt/pm units.  
        Dimensions:  (num_angles, num_spatial_modes)
    post_wfsc_wfe : array
        Element-by-element product of `wfe` and `wfsc_factor`.  
        Dimensions:  same as `wfe`
    delta_contrast : array
        Change in contrast due to residual wavefront error after WFS&C. 
        Dimensions:  (num_angles)
    angles : array
        Angular-separation values, loaded from the `contrast_filename' file
        [arcsec].  
        Dimension:  num_angles
    contrast : array
        Reference contrast values at each angular separation, loaded from the 
        `contrast_filename` file
        Dimension:  num_angles
    ppFact : array
        Post-processing factor at each angular separation.  WARNING:  all 
        values capped at 1.0
        Dimension:  num_angles
    working_angles : list
        For each target star, a list of 3 angular values corresponding to 
        the inner HZ edge (equiv. 0.96 AU), EEID, and the outer HZ edge 
        (equiv. 1.67 AU)
    C_p : (~astropy.units.Quantity(~numpy.ndarray(float)))
        Planet signal electron count rate [1/s]
    C_b : (~astropy.units.Quantity(~numpy.ndarray(float)))
        Background noise electron count rate [1/s]
    C_sp : (~astropy.units.Quantity(~numpy.ndarray(float)))
        Residual speckle spatial structure (systematic error) [1/s]
    C_star : (~astropy.units.Quantity(~numpy.ndarray(float)))
        Non-coronagraphic stellar count rate [1/s]
    C_sr : (~astropy.units.Quantity(~numpy.ndarray(float)))
        Starlight residual count rate [1/s]
    C_z : (~astropy.units.Quantity(~numpy.ndarray(float)))
        Local zodi  count rate [1/s]
    C_ez : (~astropy.units.Quantity(~numpy.ndarray(float)))
        Exo-zodi count rate [1/s]
    C_dc : (~astropy.units.Quantity(~numpy.ndarray(float)))
        Dark current  count rate [1/s]
    C_rn : (~astropy.units.Quantity(~numpy.ndarray(float)))
        Read noise count rate [1/s]
    int_time : list
        Required integration times associated with the targets

    References
    ----------
    1. Nemati et al. (2020) JATIS
    2. EXOSIMS documentation:  https://exosims.readthedocs.io/en/latest/
    3. EXOSIMS Github code repository

    """

    def __init__(self
                 , input_dir=os.path.join(".", "inputs")
                 , output_dir=os.path.join(".", "output")
                 , pp_json_filename="test_pp.json"
                 , contrast_filename="contrast.csv"
                 , throughput_filename="throughput.csv"
                 , target_list=[32439, 77052, 79672, 26779, 113283]
                 , eeid=[0.07423, 0.06174, 0.07399, 0.05633, 0.05829]
                 , eepsr=[6.34e-11, 1.39e-10, 1.06e-10, 2.42e-10, 5.89e-10]
                 , exo_zodi=5*[0.0], npoints=3):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.target_list = [f"HIP {n}" for n in target_list]
        self.exo_zodi = exo_zodi
        self.eeid = eeid
        self.eepsr = eepsr
        self.pp_json_filename = pp_json_filename
        self.contrast_filename = contrast_filename
        self.throughput_filename = throughput_filename
        self.input_dict = None
        self.load_json()
        self.wfe = None
        self.wfsc_factor = None
        self.sensitivity = None
        self.post_wfsc_wfe = None
        self.delta_contrast = None
        contrast_path = os.path.join(self.input_dir, contrast_filename)
        self.angles = read_csv(filename=contrast_path, skiprows=1)[:,0]
        self.contrast = None
        self.ppFact = None
        self.working_angles = []
        self.npoints = npoints
        self.C_p = []
        self.C_b = []
        self.C_sp = []
        self.C_star = []
        self.C_sr = []
        self.C_z = []
        self.C_ez = []
        self.C_dc = []
        self.C_rn = []
        self.int_time = np.zeros((len(self.target_list), self.npoints)) * u.d

    def load_json(self):
        """
        Load the JSON input file, which contains reference EXOSIMS parameters 
        as well as WFE, sensitivity, and WFS&C parameters.  Assign parameter 
        dictionary to `self.input_dict`. 

        """
        input_path = os.path.join(self.input_dir, self.pp_json_filename)
        with open(os.path.join(input_path)) as input_json:
            input_dict = js.load(input_json)
        self.input_dict = input_dict

    def load_csv_contrast(self):
        """
        Load CSV file containing contrast vs. angular separation values into 
        ndarray and assign to `self.contrast`.  

        """
        path = os.path.join(self.input_dir, self.contrast_filename)
        self.angles = np.genfromtxt(path, delimiter=',', skip_header=1)[:, 0]
        self.contrast = np.genfromtxt(path, delimiter=',', skip_header=1)[:, 1]

    def compute_ppFact(self):
        """
        Compute the post-processing factor and assign the array to 
        `self.ppFact`.  

        Reference
        ---------
        - See <Post-Processing Factor> document for mathematical description

        """
        self.wfe = np.array(self.input_dict['wfe'])
        self.wfsc_factor = np.array(self.input_dict['wfsc_factor'])
        self.sensitivity = np.array(self.input_dict['sensitivity'])

        self.post_wfsc_wfe = np.multiply(self.wfe, self.wfsc_factor)
        delta_contrast = np.empty(self.sensitivity.shape[0])
        for n in range(len(delta_contrast)):
            delta_contrast[n] = np.sqrt((np.multiply(self.sensitivity[n]
                                     , self.post_wfsc_wfe)**2).sum()
                                       ) 
        self.delta_contrast = 1E-12*delta_contrast
        ppFact = self.delta_contrast/self.contrast
        self.ppFact = np.where(ppFact>1.0, 1.0, ppFact)
        path = os.path.join(self.input_dir, "ppFact.fits")
        with open(path, 'wb') as f:
            arr = np.vstack((self.angles, self.ppFact)).T
            fits.writeto(f, arr, overwrite=True)

    def write_temp_json(self, filename='temp.json'):
        """
        Write `self.input_dict` to temporary JSON file for running EXOSIMS.

        """
        self.input_dict["ppFact"] = os.path.join(self.input_dir, "ppFact.fits")
        path = os.path.join(self.input_dir, filename)
        with open(path, 'w') as f:
            js.dump(self.input_dict, f)
        return filename
        
    def run_exosims(self, temp_json_filename):
        """
        Run EXOSIMS to generate results, including exposure times 
        required for reaching specified SNR.  

        """
        # build sim object:
        input_path = os.path.join(self.input_dir, temp_json_filename)
        sim = ems.MissionSim(str(input_path), use_core_thruput_for_ez=False)
        
        # identify targets of interest
        for j, t in enumerate(self.target_list):
            if t not in sim.TargetList.Name:
                self.target_list[j] += " A"
                assert self.target_list[j] in sim.TargetList.Name
        sInds = np.array([np.where(sim.TargetList.Name == t)[0][0] for t 
                         in self.target_list])
        
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
            WA_inner = 0.95*self.eeid[j]
            WA_outer = 1.67*self.eeid[j]
            WA = [WA_inner, self.eeid[j], WA_outer]
            self.working_angles.append(WA)
            # target planet deltaMag (evaluate for a range):
            dMag0 = -2.5*np.log10(self.eepsr[j])
            dMags = np.array([(dMag0-5.0*np.log10(self.eeid[j]/WA_inner))
                     , dMag0, (dMag0-5.0*np.log10(self.eeid[j]/WA_outer))]) 
            self.int_time[j] = sim.OpticalSystem.calc_intTime(
                sim.TargetList,
                [sInd] * self.npoints,
                [fZ.value] * self.npoints * fZ.unit,
                [self.exo_zodi[j]*sim.ZodiacalLight.fEZ0.value] * self.npoints 
                    * sim.ZodiacalLight.fEZ0.unit,
                dMags,
                WA * u.arcsec,
                mode
            )
            counts = sim.OpticalSystem.Cp_Cb_Csp(
                sim.TargetList,
                [sInd] * self.npoints,
                [fZ.value] * self.npoints * fZ.unit,
                [self.exo_zodi[j]*sim.ZodiacalLight.fEZ0.value] * self.npoints 
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

    def output_to_json(self, output_json_filename):
                 
        """
        Write EXOSIMS results to a JSON file. 

        Parameters
        ----------
        output_json_filename : str
            Name of JSON file containing select attributes, with values, of the
            `instantiated `ErrorBudget` object

        """
        path = os.path.join(self.output_dir, output_json_filename)
        output_dict = {
                "int_time": [x.value.tolist() for x in self.int_time],
                "ppFact": self.ppFact.tolist(),
                "working_angles": self.working_angles,
                "C_p": [x.value.tolist() for x in self.C_p], 
                "C_b": [x.value.tolist() for x in self.C_b], 
                "C_sp": [x.value.tolist() for x in self.C_sp], 
                "C_star": [x.value.tolist() for x in self.C_star], 
                "C_sr": [x.value.tolist() for x in self.C_sr], 
                "C_z": [x.value.tolist() for x in self.C_z], 
                "C_ez": [x.value.tolist() for x in self.C_ez], 
                "C_dc": [x.value.tolist() for x in self.C_dc], 
                "C_rn": [x.value.tolist() for x in self.C_rn]}
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        with open(path, 'w') as f:
            js.dump(output_dict, f, indent=4)

    def run_etc(self, wfe, wfsc_factor, sensitivity, output_filename_prefix='output_',
                var_par=False, subsystem=None, name=None, value=None, remove_temp_jsons=True):
        """
        Run end-to-end sequence of methods to produce results written to
        output JSON file.

        Parameters
        ----------
        wfe : array or list
            Wavefront changes specified in spatio-temporal bins, loaded from
            the input JSON file, in pm units.
            Dimensions:  (num_temporal_modes, num_spatial_modes).
        wfsc_factor : array or list
            Wavefront-change-mitigation factors, loaded from the input JSON
            file.  Values should be between 0 and 1.
            Dimensions:  same as `wfe`
        sensitivity : array or list
            Coefficients of contrast sensitivity w.r.t. wavefront changes,
            in ppt/pm units.
            Dimensions:  (num_angles, num_spatial_modes)
        output_filename_prefix: str
            prefix for the output file name.
        var_par : bool
            Whether the user wants to input a range of values for an
            EXOSIMS 'scienceInstruments', 'starlightSuppressionSystems', or
            'observingModes' parameter.
        subsystem: str
            subsystem of the parameter being swept over e.g. 'scienceInstruments, 'starlightSuppressionSystems', or
            'observingModes'.
        name: str
            Name of the parameter being swept over under the given subsystem e.g. 'QE', 'idark', 'IWA, etc.
        value: float
            Value for the parameter subsystem, name to take for this run of EXOSIMS.
        Note
        ----
        - If `var_par`==True, the EXOSIMS parameter name and the iterated 
        value will be appended to the name of each output JSON file
        - See EXOSIMS documentation for EXOSIMS parameters:  
            - https://exosims.readthedocs.io/en/latest/opticalsystem.html


        """

        if isinstance(wfe, list):
            wfe = np.array(wfe)
        if isinstance(wfsc_factor, list):
            wfsc_factor = np.array(wfsc_factor)
        if isinstance(sensitivity, list):
            sensitivity = np.array(sensitivity)

        update_pp_json(os.path.join(self.input_dir, self.pp_json_filename), wfe=wfe, wfsc=wfsc_factor,
                       sensitivity=sensitivity, throughput_path=os.path.join(self.input_dir, self.throughput_filename),
                       contrast_path=os.path.join(self.input_dir, self.contrast_filename))
        self.load_json()
        self.load_csv_contrast()
        self.compute_ppFact()

        if var_par:
            try:
                self.input_dict[subsystem][0][name] = value
                temp_json_filename = self.write_temp_json('temp_' + name + '_' + str(value) + '.json')
                self.run_exosims(temp_json_filename)
                filename = (output_filename_prefix + '_' + name + '_' + str(value)+ '.json')
                self.output_to_json(filename)
            except KeyError:
                self.input_dict[name][0] = value
                temp_json_filename = self.write_temp_json('temp_' + name + '_' + str(value) + '.json')
                self.run_exosims(temp_json_filename)
                filename = (output_filename_prefix + '_' + name + '_' + str(value)+ '.json')
                self.output_to_json(filename)
        else:
            temp_json_filename = self.write_temp_json()
            self.run_exosims(temp_json_filename)
            self.output_to_json(output_filename_prefix + '.json')

        # Remove temp files to run EXOSIMS to prevent clutter. If these want to be maintained they should be renamed and
        # potentially reformatted
        if remove_temp_jsons:
            for fname in glob.glob(self.input_dir + '/*'):
                if 'temp_' in fname:
                    print(f'Removing {fname}')
                    os.remove(fname)


class ErrorBudgetMcmc(object):
    """Markov-chain-Monte-Carlo exploration of coronagraphic parameters.

    Attributes:
        config_file: Name of the configuration file. 
        target_list: Target stars' HIP names.
        eeid:  Earth-equivalent-insolation distances [arcsec].
        eepsr:  Earth-equivalent planet-star flux ratios.
        wfe:  Wavefront errors [pm].
        wfsc_factor: Wavefront-sensing-and-control factors 
        sensitivity: contrast sensitivity to wavefront errors [1E-12/pm]
        post_wfsc_wfe


    """

    def __init__(self, config_file):
        self.config_file = config_file
        with open(config_file, 'r') as config:
            self.config = yaml.load(config, Loader=yaml.FullLoader)
        self.target_list = None
        self.exo_zodi = None
        self.eeid = None
        self.eepsr = None 
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
        self.contrast_filename = None
        self.throughput_filename = None
        self.ppFact_filename = None
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

    
    def write_ppFact_fits(self, trash):
        """
        Create FITS file of ppFact array with randomized filename.  

        """
        temporary_dir = self.config['paths']['temporary']
        if self.angles is not None:
            rng = np.random.default_rng()
            random_string = str(rng.integers(1E9))
            filename = "ppFact_"+random_string+".fits"
            path = os.path.join(temporary_dir, filename)
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
        Create csv file of contrast or throughput  array with randomized 
        filename.  

        """
        temporary_dir = self.config['paths']['temporary']
        if self.angles is not None:
            rng = np.random.default_rng()
            random_string = str(rng.integers(1E9))
            filename = contrast_or_throughput+'_'+random_string+".csv"
            path = os.path.join(temporary_dir, filename)
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
        input_path = self.config['paths']['input']
        contrast_path = os.path.join(input_path, config['input_files']\
                ['contrast'])
        throughput_path = os.path.join(
                input_path, config['input_files']['throughput'])
        self.angles = read_csv(
                filename=contrast_path
                , skiprows=1
                )[:,0]
        self.contrast = read_csv(
                filename=contrast_path
                , skiprows=1
                )[:,1]
        self.throughput = read_csv(
                filename=throughput_path
                , skiprows=1
                )[:, 1]
        self.wfe = read_csv(
                filename=os.path.join(input_path, config['input_files']['wfe'])
                , skiprows=1
                                )
        self.wfsc_factor = read_csv(
            filename=os.path.join(input_path
            , config['input_files']['wfsc_factor'])
            , skiprows=1
                                )
        self.sensitivity = read_csv(
            filename=os.path.join(input_path
            , config['input_files']['sensitivity'])
            , skiprows=1
                                )
        self.target_list = ['HIP '+ str(config['targets'][star]['HIP'])
                                for star in config['targets']]
        self.eeid = [config['targets'][star]['eeid'] 
                                for star in config['targets']]
        self.eepsr = [config['targets'][star]['eepsr'] 
                                for star in config['targets']]
        self.exo_zodi = [config['targets'][star]['exo_zodi'] 
                                for star in config['targets']]
        self.exosims_pars_dict = config['initial_exosims']
        self.write_ppFact_fits(trash=False)
        self.exosims_pars_dict['ppFact'] = self.ppFact_filename
        self.exosims_pars_dict['cherryPickStars'] = self.target_list
        self.exosims_pars_dict['starlightSuppressionSystems'][0]\
            ['core_contrast'] = contrast_path
        self.exosims_pars_dict['starlightSuppressionSystems'][0]\
            ['core_thruput'] = throughput_path
        for key in self.exosims_pars_dict['scienceInstruments'][0].keys():
            if key != 'optics' in dir(self):
                setattr(self, key
                        , self.exosims_pars_dict['scienceInstruments'][0][key])
        for key \
             in self.exosims_pars_dict['starlightSuppressionSystems'][0] \
                 .keys():
            if key in dir(self):
                setattr(self, key
                        , self.exosims_pars_dict['starlightSuppressionSystems']
                                                [0][key])

    def initialize_walkers(self):
        center = []
        [center.append(np.array(val).ravel())  
         for var_name in self.config['mcmc']['variables'] 
         for val in self.config['mcmc']['variables'][var_name]['ini_pars']
                               ['center']]
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

    def update_attributes(self, values):
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
                print(key+" not found in attributes list")

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
        self.update_attributes(values)
        int_time, C_p, C_b, C_sp, C_sr, C_z, C_ez, C_dc, C_rn, C_star \
                = self.run_exosims()
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

    def run_exosims(self, file_cleanup=True):
        """
        Run EXOSIMS to generate results, including exposure times 
        required for reaching specified SNR.  

        """
        C_p = []
        C_b = []
        C_sp = []
        C_sr = []
        C_z = []
        C_ez = []
        C_dc = []
        C_rn = []
        C_star = []
        wa_coefs = self.config["working_angles"]
        n_angles = len(wa_coefs)
        target_list = self.target_list
        eeid = self.eeid
        eepsr = self.eepsr
        exo_zodi = self.exo_zodi
        sim = ems.MissionSim(use_core_thruput_for_ez=False
                             , **deepcopy(self.exosims_pars_dict))
        
        # identify targets of interest
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
            int_time[j] = sim.OpticalSystem.calc_intTime(
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
            C_p.append(counts[0])

            C_b.append(counts[1])
            C_sp.append(counts[2])
            C_sr.append(counts[3]["C_sr"])
            C_z.append(counts[3]["C_z"])
            C_ez.append(counts[3]["C_ez"])
            C_dc.append(counts[3]["C_dc"])
            C_rn.append(counts[3]["C_rn"])
            C_star.append(counts[3]["C_star"])
        if file_cleanup:
            self.clean_files()
        return int_time, C_p, C_b, C_sp, C_sr, C_z, C_ez, C_dc, C_rn, C_star

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
    def __init__(self, config, parameter, values, error_budget, wfe, wfsc_factor, sensitivity, fixed_contrast,
                 fixed_throughput, contrast_filename, throughput_filename, output_file_name='',
                 is_exosims_param=False):
        '''

        :param config: dict
            The configuration for the ebs run.
        :param parameter: str
            The name of the parameter being swept over.
        :param values: array
            The values of the parameter to be swept over
        :param error_budget: ErrorBudget
            ErrorBudget object to use for the sweep.
        :param wfe: array
            Wavefront changes specified in spatio-temporal bins, loaded from the input JSON file, in pm units.
        :param wfsc_factor: array
            Wavefront-change-mitigation factors, loaded from the input JSON file. Values should be between 0 and 1.
        :param sensitivity: array
            Coefficients of contrast sensitivity w.r.t. wavefront changes, in ppt/pm units.
        :param fixed_contrast: float
            If contrast is not being swept over, the fixed value at which to keep it.
        :param fixed_throughput: float
            If throughput is not being swept over, the fixed value at which to keep it.
        :param contrast_filename: str
            The filename where the contrast information is stored.
        :param throughput_filename: str
            The filename where the throughput information is stored.
        :param output_file_name: str
            name of the output file
        :param is_exosims_param: bool
            If True will feed the parameter into EXOSIMS to be swept over
        '''
        self.config = config
        self.parameter, self.subparameter = parameter
        self.values = values
        self.result_dict = {}
        self.error_budget = error_budget
        self.error_budget.load_csv_contrast()
        self.is_exosims_param = is_exosims_param
        self.wfe = wfe
        self.wfsc_factor = wfsc_factor
        self.sensitivity = sensitivity
        self.output_file_name = output_file_name
        self.fixed_contrast = fixed_contrast
        self.fixed_throughput = fixed_throughput
        self.contrast_filename = contrast_filename
        self.throughput_filename = throughput_filename
        self.angles = self.error_budget.angles
        self.var_par = True
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
                np.savetxt(self.contrast_filename, np.column_stack((self.angles, value * np.ones(len(self.angles))))
                           , delimiter=",", header=('r_as,core_contrast')
                           , comments="")
                self.var_par = False
            else:
                np.savetxt(self.contrast_filename, np.column_stack((self.angles,
                                                                    self.fixed_contrast * np.ones(len(self.angles))))
                           , delimiter=",", header=('r_as,core_contrast')
                           , comments="")
            if self.parameter == 'throughput':
                np.savetxt(self.throughput_filename
                           , np.column_stack((self.angles, value * np.ones(len(self.angles))))
                           , delimiter=",", header=('r_as,core_thruput')
                           , comments="")
                self.var_par = False
            else:
                np.savetxt(self.throughput_filename
                           , np.column_stack((self.angles, self.fixed_throughput * np.ones(len(self.angles))))
                           , delimiter=",", header=('r_as,core_thruput')
                           , comments="")

            # TODO change mutable parameters in ErrorBudget class and remove this deep copy
            error_budget = deepcopy(self.error_budget)

            error_budget.run_etc(self.wfe, self.wfsc_factor, self.sensitivity,
                                 self.output_file_name + f'_{self.parameter}', var_par=self.var_par,
                                 subsystem=self.parameter, name=self.subparameter, value=value)

            for key in self.result_dict.keys():
                arr = self.result_dict[key]
                arr[i] = np.array(getattr(error_budget, key))
                self.result_dict[key] = arr

        return self.result_dict





