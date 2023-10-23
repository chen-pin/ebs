"""<error_budget> module

- Need to first install EXOSIMS, see Refs 2 & 3 below
- Also need 2 CSV files and 1 JSON input file.  The CSV files specify 
reference contrast and throughput values as functions of angular separation.  
The JSON file specifies instrument and observational parameter for the 
reference exposure, followed by  WFE, contrast sensitivity, and WFS&C factors.  
These files need to reside in the path "./inputs".  See doc strings 
for arugments `contrast_filename`, and
`pp_json_filename` below.  One can create arrays of WFE, sensitivity, and 
WFS&C values, and then use the `create_pp_json()` method to create the JSON 
file.  
- See <example.py> in the parent directory for an example on how to use this 
module.
"""


import os 
import numpy as np
import json as js
import yaml 
from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt
import EXOSIMS.MissionSim as ems
from copy import deepcopy
from ebs.utils import generate_pp_json
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
        Name of CSV file specifying contrast of th reference dark hole 
        (i.e. contrast obtained on the reference star)
    target_list : list
        List of target-star HIP IDs
    luminosity : list
        Luminosity values of the target stars [log10(L*/L_sun)]
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
                 , target_list=[32439, 77052, 79672, 26779, 113283]
                 , luminosity=[0.2615, -0.0788, 0.0391, -0.3209, -0.707]
                 , eeid=[0.07423, 0.06174, 0.07399, 0.05633, 0.05829]
                 , eepsr=[6.34e-11, 1.39e-10, 1.06e-10, 2.42e-10, 5.89e-10]
                 , exo_zodi=5*[0.0], npoints=3):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.target_list = [f"HIP {n}" for n in target_list]
        self.luminosity = luminosity
        self.exo_zodi = exo_zodi
        self.eeid = eeid
        self.eepsr = eepsr
        self.pp_json_filename = pp_json_filename
        self.contrast_filename = contrast_filename
        self.input_dict = None
        self.wfe = None
        self.wfsc_factor = None
        self.sensitivity = None
        self.post_wfsc_wfe = None
        self.delta_contrast = None
        self.angles = None
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

    def load_json(self, verbose=False):
        """
        Load the JSON input file, which contains reference EXOSIMS parameters 
        as well as WFE, sensitivity, and WFS&C parameters.  Assign parameter 
        dictionary to `self.input_dict`. 

        """
        input_path = os.path.join(self.input_dir, self.pp_json_filename)
        with open(os.path.join(input_path)) as input_json:
            input_dict = js.load(input_json)
            if verbose:
                print("Top two level dictionary keys\n")
                for key in input_dict.keys():
                    print(key)
                    try:
                        for subkey in input_dict[key].keys():
                            print("\t{}".format(subkey))
                    except:
                        pass
                print("\nStarlightSuppressionSystems:")
                print(input_dict['starlightSuppressionSystems'])
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
            WA_inner = 0.95*np.exp(luminosity[j]/2)*self.eeid[j]
            WA_outer = 1.67*np.exp(luminosity[j]/2)*self.eeid[j]
            WA = [WA_inner, self.eeid[j], WA_outer]
            self.working_angles.append(WA)
            # target planet deltaMag (evaluate for a range):
            dMag0 = -2.5*np.log10(self.eepsr[j])
            dMags = np.array([(dMag0-2.5*np.log10(self.eeid[j]/WA_inner))
                     , dMag0, (dMag0-2.5*np.log10(self.eeid[j]/WA_outer))]) 
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

    def run_etc(self, wfe, wfsc_factor, sensitivity
                , output_filename_prefix
                ,var_par, *args):
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
        var_par : bool
            Whether or not the user wants to input a range of values for an 
            EXOSIMS 'scienceInstruments', 'starlightSuppressionSystems', or 
            'observingModes' parameter.
        *args 
            If `var_par` == True, enter 3 additional arguments:  
                1. String indicating the EXOSIMS subsystem.  Possible values 
                comprise the following 
                    - 'scienceInstruments'
                    - 'starlightSuppressionSystems' 
                    - 'observingModes'
                2. String indicating the EXOSIMS paramter (e.g. 'optics', 
                'QE', or 'SNR')
                3. List_like data providing the range of parameter values

        Note
        ----
        - If `var_par`==True, the EXOSIMS parameter name and the iterated 
        value will be appended to the name of each output JSON file
        - See EXOSIMS documentation for EXOSIMS parameters:  
            - https://exosims.readthedocs.io/en/latest/opticalsystem.html


        """
        generate_pp_json(os.path.join(self.input_dir, self.pp_json_filename), wfe=wfe, wfsc=wfsc_factor, sensitivity=sensitivity)
        self.load_json()
        self.load_csv_contrast()
        self.compute_ppFact()
        if var_par:
            if args[0] == None:
                if args[1] == 'pupilDiam':
                    for value in args[2]:
                        self.input_dict[args[1]] = value
                        temp_json_filename = self.write_temp_json(
                                             'temp_'+args[1]+'_'+str(value)
                                             +'.json'
                                                                 )
                        self.run_exosims(temp_json_filename)
                        filename = (output_filename_prefix+'_'+args[1]+'_'
                                    +str(value))
                        self.output_to_json(filename)
                else: 
                    print("Error in specifying EXOSIMS parameter to be varied")
            else:
                for value in args[2]:
                    self.input_dict[args[0]][0][args[1]] = value
                    temp_json_filename = self.write_temp_json(
                                         'temp_'+args[1]+'_'+str(value)
                                         +'.json')
                    self.run_exosims(temp_json_filename)
                    filename = (output_filename_prefix+'_'+args[1]+'_'
                                +str(value)+'.json')
                    self.output_to_json(filename)
        else:
            temp_json_filename = self.write_temp_json()
            self.run_exosims(temp_json_filename)
            self.output_to_json(output_filename_prefix+'.json')


class ErrorBudget2(object):

    def __init__(self, config_file="config.yml"):
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
        self.working_angles = []
#        self.npoints = None 
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
        self.core_thruput = None
        self.core_mean_intensity = None
        self.SNR = None
        self.C_p = []
        self.C_b = []
        self.C_sp = []
        self.C_star = []
        self.C_sr = []
        self.C_z = []
        self.C_ez = []
        self.C_dc = []
        self.C_rn = []
        self.int_time = None
        self.ppFact_filename = None
        self.contrast_filename = None
        self.throughput_filename = None
        self.ppFact_filename = None
        self.exosims_pars_dict = None
        self.mcmc_vars = self.config['mcmc']['variables'].keys()

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

    
    def write_ppFact_fits(self):
        """
        Create FITS file of ppFact array with randomized filename.  

        """
        input_dir = self.config['paths']['input']
        if self.angles is not None:
            random_string = str(int(1e10*np.random.rand()))
            filename = "ppFact_"+random_string+".fits"
            path = os.path.join(input_dir, filename)
            with open(path, 'wb') as f:
                arr = np.vstack((self.angles, self.ppFact)).T
                fits.writeto(f, arr, overwrite=True)
                self.ppFact_filename = path
        else:  
            print("Need to assign angle values to write ppFact FITS file")
    
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
#        self.load_csv_contrast()
        self.target_list = ['HIP '+ str(config['targets'][star]['HIP'])
                                for star in config['targets']]
        self.eeid = [config['targets'][star]['eeid'] 
                                for star in config['targets']]
        self.eepsr = [config['targets'][star]['eepsr'] 
                                for star in config['targets']]
        self.exo_zodi = [config['targets'][star]['exo_zodi'] 
                                for star in config['targets']]
        self.exosims_pars_dict = config['initial_exosims']
#        self.wfe = np.array(config['wfsc_args']['wfe'])
#        self.wfsc_factor = np.array(config['wfsc_args']['wfsc_factor'])
#        self.sensitivity = np.array(config['wfsc_args']['sensitivity'])
        self.write_ppFact_fits()
        self.exosims_pars_dict['ppFact'] = self.ppFact_filename
        self.exosims_pars_dict['cherryPickStars'] = self.target_list
        self.exosims_pars_dict['starlightSuppressionSystems'][0]\
            ['core_contrast'] = contrast_path
        self.exosims_pars_dict['starlightSuppressionSystems'][0]\
            ['core_thruput'] = throughput_path

    def repack_array(self, par_name, *values):
        print(values)
        template = np.array(self.config['mcmc']['variables'][par_name]
                                       ['ini_pars']['center'])
        print(template)
        indices = np.where(np.isfinite(template))
        print(indices)
        arr = getattr(self, par_name)
        print(arr)
        arr[indices] = values
        print(arr)

    def initialize_walkers(self):
        nwalkers = self.config['mcmc']['nwalkers']
        var_names = self.config['mcmc']['variables'].keys()
        kwargs = [self.config['mcmc']['variables'][par_name]['ini_pars'].keys() 
                  for par_name in var_names]
        center = []
        [center.append(np.array(val).ravel())  
         for var_name in self.config['mcmc']['variables'] 
         for val in self.config['mcmc']['variables'][var_name]['ini_pars']
                               ['center']]
        center = np.concatenate(center)
        center = center[np.where(np.isfinite(center))]
        # print(center)
        spread = []
        [spread.append(np.array(val).ravel())  
         for var_name in self.config['mcmc']['variables'] 
         for val in self.config['mcmc']['variables'][var_name]['ini_pars']
                               ['spread']]
        spread = np.concatenate(spread)
        spread= spread[np.where(np.isfinite(spread))]
        # print(spread)
        ndim = center.shape[0]
        walker_pos = np.random.normal(center, spread, (nwalkers, ndim))
        return walker_pos
        #print(walker_pos)

        #values = [self.config['mcmc']['variables'][kwar]
        #print(values)
        #for i, name in enumerate(var_names):
        #    print(f"MCMC Variable:  {name}")
        #    [print(f"    keyword:  {key}") for key in kwargs[i]]


    def update_attributes(self, value):
        print(self.config['mcmc']['variables'].keys())
        print(dir(self))
        for i, key in enumerate(self.config['mcmc']['variables'].keys()):
            if key in dir(self):
                setattr(self, key, value[i])
                print(f"Updated {key} to {getattr(self, key)}")
            else:
                print(key+" not found in attributes list")


    def run_exosims(self, initial=True, remove_ppFact_file=True):
        """
        Run EXOSIMS to generate results, including exposure times 
        required for reaching specified SNR.  

        """
        if initial:
            self.initialize_for_exosims()
        n_angles = 3
        target_list = self.target_list
        eeid = self.eeid
        eepsr = self.eepsr
        exo_zodi = self.exo_zodi
        # build sim object:
        sim = ems.MissionSim(use_core_thruput_for_ez=False
                             , **self.exosims_pars_dict)
        
        # identify targets of interest
#        for j, t in enumerate(target_list): 
#            if t not in sim.TargetList.Name:
#                target_list[j] += " A"
#                assert target_list[j] in sim.TargetList.Name
        sInds = np.array([np.where(sim.TargetList.Name == t)[0][0] for t 
                         in target_list])
        
        # assemble information needed for integration time calculation:
        
        # we have only one observing mode defined, so use that
        mode = sim.OpticalSystem.observingModes[0]
        
        # use the nominal local zodi and exozodi values
        fZ = sim.ZodiacalLight.fZ0
        
        # now we loop through the targets of interest and compute integration 
        # times for each:
        self.int_time = np.empty((len(target_list), n_angles))*u.d
        for j, sInd in enumerate(sInds):
            # choose angular separation for coronagraph performance
            # this doesn't matter for a flat contrast/throughput, but
            # matters a lot when you have real performane curves
            WA_inner = 0.95*eeid[j]
            WA_outer = 1.67*eeid[j]
            WA = [WA_inner, eeid[j], WA_outer]
            self.working_angles.append(WA)
            # target planet deltaMag (evaluate for a range):
            dMag0 = -2.5*np.log10(eepsr[j])
            dMags = np.array([(dMag0-2.5*np.log10(eeid[j]/WA_inner))
                     , dMag0, (dMag0-2.5*np.log10(eeid[j]/WA_outer))])
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
            self.C_p.append(counts[0])

            self.C_b.append(counts[1])
            self.C_sp.append(counts[2])
            self.C_sr.append(counts[3]["C_sr"])
            self.C_z.append(counts[3]["C_z"])
            self.C_ez.append(counts[3]["C_ez"])
            self.C_dc.append(counts[3]["C_dc"])
            self.C_rn.append(counts[3]["C_rn"])
            self.C_star.append(counts[3]["C_star"])
        if remove_ppFact_file:
            os.remove(self.ppFact_filename)


class ParameterSweep:
    def __init__(self, config, parameter, values, error_budget, wfe, wfsc_factor, sensitivity, fixed_contrast,
                 fixed_throughput, contrast_filename, throughput_filename, angles, output_file_name='',
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
        :param angles: array
            Angular separation values [arcsec]
        :param output_file_name: str
            name of the output file
        :param is_exosims_param: bool
            If True will feed the parameter into EXOSIMS to be swept over
        '''
        self.config = config
        self.parameter = parameter
        self.values = values
        self.result_dict = {}
        self.error_budget = error_budget
        self.is_exosims_param = is_exosims_param
        self.wfe = wfe
        self.wfsc_factor = wfsc_factor
        self.sensitivity = sensitivity
        self.output_file_name = output_file_name
        self.fixed_contrast = fixed_contrast
        self.fixed_throughput = fixed_throughput
        self.contrast_filename = contrast_filename
        self.throughput_filename = throughput_filename
        self.angles = angles
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
        if self.parameter == 'contrast':
            for i, contrast in enumerate(self.values):
                # TODO change mutable parameters in ErrorBudget class and remove this deep copy
                error_budget = deepcopy(self.error_budget)
                np.savetxt(self.contrast_filename, np.column_stack((self.angles, contrast * np.ones(len(self.angles))))
                           , delimiter=",", header=('r_as,core_contrast')
                           , comments="")
                np.savetxt(self.throughput_filename
                           , np.column_stack((self.angles, self.fixed_throughput* np.ones(len(self.angles))))
                           , delimiter=",", header=('r_as,core_thruput')
                           , comments="")

                error_budget.run_etc(self.wfe, self.wfsc_factor, self.sensitivity, 'example_contrasts', False)
                for key in self.result_dict.keys():
                    arr = self.result_dict[key]
                    arr[i] = np.array(getattr(error_budget, key))
                    self.result_dict[key] = arr

        elif self.parameter == 'throughput':
            for i, throughput in enumerate(self.values):
                error_budget = deepcopy(self.error_budget)
                np.savetxt(self.contrast_filename, np.column_stack((self.angles,
                                                                    self.fixed_contrast * np.ones(len(self.angles))))
                           , delimiter=",", header=('r_as,core_contrast')
                           , comments="")
                np.savetxt(self.throughput_filename
                           , np.column_stack((self.angles, throughput * np.ones(len(self.angles))))
                           , delimiter=",", header=('r_as,core_thruput')
                           , comments="")
                error_budget.run_etc(self.wfe, self.wfsc_factor, self.sensitivity
                                     , 'example_core_tputs', False)

                for key in self.result_dict.keys():
                    arr = self.result_dict[key]
                    arr[i] = np.array(getattr(error_budget, key))
                    self.result_dict[key] = arr
        else:
            for i, val in enumerate(self.values):
                error_budget = deepcopy(self.error_budget)
                np.savetxt(self.contrast_filename, np.column_stack((self.angles,
                                                                    self.fixed_contrast * np.ones(len(self.angles))))
                           , delimiter=",", header=('r_as,core_contrast')
                           , comments="")
                np.savetxt(self.throughput_filename
                           , np.column_stack((self.angles, self.fixed_throughput * np.ones(len(self.angles))))
                           , delimiter=",", header=('r_as,core_thruput')
                           , comments="")
                error_budget.run_etc(self.wfe, self.wfsc_factor, self.sensitivity, self.output_file_name, True,
                                          'scienceInstruments', self.parameter, [val])
                for key in self.result_dict.keys():
                    arr = self.result_dict[key]
                    arr[i] = np.array(getattr(error_budget, key))
                    self.result_dict[key] = arr
        return self.result_dict




