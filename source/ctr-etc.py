"""Exposure time calculator module"""

# Contributors:
#    Pin Chen

import os 
import numpy as np
import json as js
from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt
import EXOSIMS.MissionSim as ems


class ErrorBudget(ems.MissionSim):
    """
    Exposure time calculator incorporating dynamical wavefront errors and 
    WFS&C
    
    Parameters
    ----------
    json_filename : str
        Name of input JSON file
    contrast_filename : str
        Name of CSV file specifying contrast of reference dark hole

    Attributes
    ----------
    target_list : list
        List of target-star HIP IDs
    input_dir : `os.path`
        Directory path where the above-listed input files reside
    input_dict : dict
        Dictionary of parameters loaded from the input JSON file
    wfe : array 
        Wavefront error specified in spatio-temporal bins, loaded from the 
        input JSON file, in pm units.  
        Dimensions:  (num_temporal_modes, num_spatial_modes).
    wfsc_factor : array
        Wavefront-error-mitigation factors, loaded from the input JSON file.  
        Values should be between 0 and 1.  
        Dimensions:  same as `wfe`
    sensitivity : array
        Coefficients of contrast sensitivity w.r.t. wavefront changes, 
        in ppt/pm units.  
        Dimensions:  (num_angles, num_temporal_modes, num_spatial_modes)
    post_wfsc_wfe : array
        Element-by-element product of `wfe` and `wfsc_factor`.  
        Dimensions:  same as `wfe`
    delta_contrast : array
        Change in contrast due to residual wavefront error after WFS&C. 
        Dimensions:  (num_angles)
    angles : array
        Angular-separation values, loaded from the input JSON file, 
        in arcsec units.  
        Dimension:  num_angles
    contrast : array
        Reference contrast values at each angular separation
        Dimension:  num_angles
    ppFact : array
        Post-processing factor at each angular separation
        Dimension:  num_angles
    """

    def __init__(self, ref_json_filename="test_ref.json"
                 , pp_json_filename="test_pp.json"
                 , output_json_filename="test_output.json"
                 , contrast_filename="contrast.csv"
                 , target_list=[32439, 77052, 79672, 26779, 113283]
                 , luminosity=[0.2615, -0.0788, 0.0391, -0.3209, -0.707]
                 , eeid=[0.07423, 0.06174, 0.07399, 0.05633, 0.05829]
                 , eepsr=[6.34e-11, 1.39e-10, 1.06e-10, 2.42e-10, 5.89e-10]
                 , exo_zodi=5*[1.0]):
        self.target_list = target_list
        self.luminosity = luminosity
        self.exo_zodi = exo_zodi
        self.eeid = eeid
        self.eepsr = eepsr
        self.ref_json_filename = ref_json_filename
        self.pp_json_filename = pp_json_filename
        self.output_json_filename = output_json_filename
        self.contrast_filename = contrast_filename
        self.input_dir = os.path.join("..", "inputs")
        self.input_dict = None
        self.wfe = None
        self.wfsc_factor = None
        self.sensitivity = None
        self.post_wfsc_wfe = None
        self.delta_contrast = None
        self.angles = None
        self.contrast = None
        self.ppFact = None

#    def xlsx2json(self):
#        """
#        Convert Exel input file into JSON file
#        """
#        sheet_name = ['KPP, no WFE', 'WFE', 'WFSC-PP factor']
#        input_dict = pd.read_excel(io=self.input_path, sheet_name=sheet_name)
#        df = input_dict['KPP, no WFE']
#        print(df.iloc[30])
#        with open(self.input_path.replace('.xlsx', '.json'), 'w') as f:
#            for key in input_dict.keys():
#                json_str = input_dict[key].to_json(f)

    def load_json(self, verbose=False):
        input_path = os.path.join(self.input_dir, self.ref_json_filename)
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
        path = os.path.join(self.input_dir, self.contrast_filename)
        self.angles = np.genfromtxt(path, delimiter=',', skip_header=1)[:, 0]
        self.contrast = np.genfromtxt(path, delimiter=',', skip_header=1)[:, 1]

    def compute_ppFact(self):
        try:
            if self.input_dict == None:
                pars_dict = self.load_json()
        except:
            pass
        self.wfe = np.array(self.input_dict['wfe'])
        self.wfsc_factor = np.array(self.input_dict['wfsc_factor'])
        self.sensitivity = np.array(self.input_dict['sensitivity'])
        self.post_wfsc_wfe = np.multiply(self.wfe, self.wfsc_factor)
        delta_contrast = np.empty(self.sensitivity.shape[0])
        for n in range(len(delta_contrast)):
            delta_contrast[n] = np.sqrt(
                        (np.multiply(self.sensitivity[n]
                                     , self.post_wfsc_wfe)**2).sum()
                                       )
        self.delta_contrast = delta_contrast
        try:
            if self.contrast == None:
                self.load_csv_contrast()
        except:
            pass
        ppFact = self.delta_contrast*1E-12/self.contrast
        self.ppFact = np.where(ppFact>1.0, 1.0, ppFact)
        path = os.path.join(self.input_dir, "ppFact.fits")
        with open(path, 'wb') as f:
            arr = np.vstack((self.angles, self.ppFact)).T
            fits.writeto(f, arr, overwrite=True)

    def write_json(self):
        try:
            if self.ppFact == None:
                self.compute_ppFact()
        except:
            pass
        self.input_dict["ppFact"] = os.path.join(self.input_dir, "ppFact.fits")
        path = os.path.join(self.input_dir, "temp.json")
        with open(path, 'w') as f:
            js.dump(self.input_dict, f)

    def create_pp_json(self, wfe, wfsc_factor, sensitivity
                           , num_spatial_modes=14, num_temporal_modes=6
                           , num_angles=27):
        path = os.path.join(self.input_dir, self.pp_json_filename)
        with open(path) as f:
            input_dict = js.load(f)
        input_dict['wfe'] = wfe.tolist()
        input_dict['wfsc_factor'] = wfsc_factor.tolist()
        input_dict['sensitivity'] = sensitivity.tolist()
        with open(path, 'w') as f:
            js.dump(input_dict, f, indent=4)
        
    def run_exosims(self):
        # build sim object:
        input_path = os.path.join(self.input_dir, 'temp.json')
        sim = ems.MissionSim(str(input_path))
        
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
#        fEZ = self.exo_zodi*(sim.ZodiacalLight.fEZ0)
        
        # target planet deltaMag (evaluate for a range):
#        npoints = 100
#        dMags = np.linspace(20, 25, npoints)
        
        # choose angular separation for coronagraph performance
        # this doesn't matter for a flat contrast/throughput, but
        # matters a lot when you have real performane curves
        # we'll use the default values, which is halfway between IWA/OWA
#        WA = (mode["OWA"] + mode["IWA"]) / 2
#        WA = self.angles[13]*u.arcsec
        
        
        # now we loop through the targets of interest and compute intTimes for 
        # each:
        npoints = 3
        intTimes = np.zeros((len(targnames), npoints)) * u.d
        for j, sInd in enumerate(sInds):
            WA_inner = 0.95*self.eeid[j]*10**(self.luminosity[j]/2.0)
            WA_outer = 1.67*self.eeid[j]*10**(self.luminosity[j]/2.0)
            WA = [WA_inner, self.eeid[j], WA_outer]
            print("Working Angles:  {} ".format(WA*u.arcsec))
            dMag0 = -2.5*np.log10(self.eepsr[j])
            dMags = np.array([(dMag0-2.5*np.log10(self.eeid[j]/WA_inner))
                     , dMag0, (dMag0-2.5*np.log10(self.eeid[j]/WA_outer))]) 
            print("dMags:  {} ".format(dMags))
            intTimes[j] = sim.OpticalSystem.calc_intTime(
                sim.TargetList,
                [sInd] * npoints,
                [fZ.value] * npoints * fZ.unit,
#                [fEZ.value] * npoints * fEZ.unit,
                [self.exo_zodi[j]*sim.ZodiacalLight.fEZ0.value] * npoints 
                    * sim.ZodiacalLight.fEZ0.unit,
                dMags,
#                [WA.value] * npoints * WA.unit,
                WA * u.arcsec,
                mode,
            )
        print("Integration Times:\n{}".format(intTimes))
        
#        plt.figure(1)
#        plt.clf()
#        for j in range(len(targnames)):
#            plt.semilogy(dMags, intTimes[j], label=targnames[j])
#        
#        plt.xlabel(rf"Achievable Planet $\Delta$mag @ {WA :.2f}")
#        plt.ylabel(f"Integration Time ({intTimes.unit})")
#        plt.legend()
#        plt.savefig('../../ctr_out/plot.png')


if __name__ == '__main__':
    x = ErrorBudget()
    num_spatial_modes = 14
    num_temporal_modes = 6
    num_angles = 27
    wfe = (1e-6*np.ones((num_temporal_modes, num_spatial_modes)))
    wfsc_factor = 0.5*np.ones_like(wfe)
    sensitivity = 5.0*np.ones((num_angles, num_spatial_modes))
    x.create_pp_json(wfe=wfe, wfsc_factor=wfsc_factor, sensitivity=sensitivity)
    x.write_json()
    x.run_exosims()
