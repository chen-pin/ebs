"""<Flux-Ratio Error Allocation Kit (FREAK)> module

- Need to first install EXOSIMS, see Refs 2 & 3 below
- Also need 2 CSV files and 1 JSON input file.  The CSV files specify 
reference contrast and throughput values as functions of angular separation.  
The JSON file specifies instrument and observational parameter for the 
reference exposure, followed by  WFE, contrast sensitivity, and WFS&C factors.  
These files need to reside in the path "./inputs".  See doc strings 
for arugments `contrast_filename`, `ref_json_filename`, and 
`pp_json_filename` below.  One can create arrays of WFE, sensitivity, and 
WFS&C values, and then use the `create_pp_json()` method to create the JSON 
file.  
- See <example.py> in the parent directory for an example on how to use this 
module.
"""


import os 
import numpy as np
import json as js
from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt
import EXOSIMS.MissionSim as ems


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
    ref_json_filename : str
        Name of JSON file specifying the initial EXOSIMS parameters, without 
        considering any wavefront drifts. 
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
                 , ref_json_filename="test_ref.json"
                 , pp_json_filename="test_pp.json"
                 , contrast_filename="contrast.csv"
                 , target_list=[32439, 77052, 79672, 26779, 113283]
                 , luminosity=[0.2615, -0.0788, 0.0391, -0.3209, -0.707]
                 , eeid=[0.07423, 0.06174, 0.07399, 0.05633, 0.05829]
                 , eepsr=[6.34e-11, 1.39e-10, 1.06e-10, 2.42e-10, 5.89e-10]
                 , exo_zodi=5*[0.0]):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.target_list = target_list
        self.luminosity = luminosity
        self.exo_zodi = exo_zodi
        self.eeid = eeid
        self.eepsr = eepsr
        self.ref_json_filename = ref_json_filename
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
        self.working_angles = None
        self.C_p = None
        self.C_b = None
        self.C_sp = None
        self.C_star = None
        self.C_sr = None
        self.C_z = None
        self.C_ez = None
        self.C_dc = None
        self.C_rn = None
        self.int_time = None

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

    def create_pp_json(self, wfe, wfsc_factor, sensitivity):
        """
        Utility to create an input JSON file (named by `self.pp_json_filename`)
        with user-created WFE, sensitivity, and post-processing parameters.  

        """
        path_ref = os.path.join(self.input_dir, self.ref_json_filename)
        with open(path_ref) as f:
            input_dict = js.load(f)
        input_dict['wfe'] = wfe.tolist()
        input_dict['wfsc_factor'] = wfsc_factor.tolist()
        input_dict['sensitivity'] = sensitivity.tolist()
        path_pp = os.path.join(self.input_dir, self.pp_json_filename)
        with open(path_pp, 'w') as f:
            js.dump(input_dict, f, indent=4)
        
    def run_exosims(self, temp_json_filename):
        """
        Run EXOSIMS to generate results, including exposure times 
        required for reaching specified SNR.  

        """
        # build sim object:
        input_path = os.path.join(self.input_dir, temp_json_filename)
        sim = ems.MissionSim(str(input_path), use_core_thruput_for_ez=False)
        
        # identify targets of interest
        targnames = [f"HIP {n}" for n in self.target_list]
        for j, t in enumerate(targnames):
            if t not in sim.TargetList.Name:
                targnames[j] += " A"
                assert targnames[j] in sim.TargetList.Name
        sInds = np.array([np.where(sim.TargetList.Name == t)[0][0] for t 
                         in targnames])
        
        # assemble information needed for integration time calculation:
        
        # we have only one observing mode defined, so use that
        mode = sim.OpticalSystem.observingModes[0]
        
        # use the nominal local zodi and exozodi values
        fZ = sim.ZodiacalLight.fZ0
        
        # now we loop through the targets of interest and compute integration 
        # times for each:
        npoints = 3
        self.int_time = np.zeros((len(targnames), npoints)) * u.d
        self.C_p = []
        self.C_b = []
        self.C_sp = []
        self.C_sr = []
        self.C_z = []
        self.C_ez = []
        self.C_dc = []
        self.C_rn = []
        self.C_star = []
        self.working_angles = []
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
            dMags = np.array([(dMag0-2.5*np.log10(self.eeid[j]/WA_inner))
                     , dMag0, (dMag0-2.5*np.log10(self.eeid[j]/WA_outer))]) 
            self.int_time[j] = sim.OpticalSystem.calc_intTime(
                sim.TargetList,
                [sInd] * npoints,
                [fZ.value] * npoints * fZ.unit,
                [self.exo_zodi[j]*sim.ZodiacalLight.fEZ0.value] * npoints 
                    * sim.ZodiacalLight.fEZ0.unit,
                dMags,
                WA * u.arcsec,
                mode,
            )
            counts = sim.OpticalSystem.Cp_Cb_Csp(
                sim.TargetList,
                [sInd] * npoints,
                [fZ.value] * npoints * fZ.unit,
                [self.exo_zodi[j]*sim.ZodiacalLight.fEZ0.value] * npoints 
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
        wfe : array 
            Wavefront changes specified in spatio-temporal bins, loaded from 
            the input JSON file, in pm units.  
            Dimensions:  (num_temporal_modes, num_spatial_modes).
        wfsc_factor : array
            Wavefront-change-mitigation factors, loaded from the input JSON 
            file.  Values should be between 0 and 1.    
            Dimensions:  same as `wfe`
        sensitivity : array
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
        self.create_pp_json(wfe=wfe, wfsc_factor=wfsc_factor
                            , sensitivity=sensitivity)
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
