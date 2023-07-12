"""Exposure time calculator module"""

# Contributors:
#    Pin Chen

import os 
import numpy as np
import json as js
import astropy.units as u
import matplotlib.pyplot as plt
#import pandas as pd
import EXOSIMS.MissionSim as ems


class ErrorBudget(ems.MissionSim):
    """Sub-class of `EXOSIMS.MissionSim.MissionSim` to compute exposure time"""

    def __init__(self, json_filename="test2.json"
                 , contrast_filename="contrast.csv"):
        self.input_dir = os.path.join("..", "inputs")

        self.input_dict = None
        self.json_filename = json_filename
        self.contrast_filename = contrast_filename
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
        input_path = os.path.join(self.input_dir, self.json_filename)
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
        if self.input_dict == None:
            pars_dict = self.load_json()
        self.wfe = np.array(self.input_dict['wfe'])
        self.wfsc_factor = np.array(self.input_dict['wfsc_factor'])
        self.sensitivity = np.array(self.input_dict['sensitivity'])
        print(self.sensitivity.shape)
        self.post_wfsc_wfe = np.multiply(self.wfe, self.wfsc_factor)
        print(self.post_wfsc_wfe.shape)
        delta_contrast = np.empty(self.sensitivity.shape[0])
        for n in range(len(delta_contrast)):
            delta_contrast[n] = np.sqrt(
                        (np.multiply(self.sensitivity[n]
                                     , self.post_wfsc_wfe)**2).sum()
                                       )
        self.delta_contrast = delta_contrast
        if self.contrast == None:
            self.load_csv_contrast()
        ppFact = self.delta_contrast*1E-12/self.contrast
        self.ppFact = np.where(ppFact>1.0, 1.0, ppFact)
        print(self.ppFact)

    def test_json(self):
        # build sim object:
        input_path = os.path.join(self.input_dir, self.json_filename)
        sim = ems.MissionSim(str(input_path))
        
        # identify targets of interest
        hipnums = [32439, 77052, 79672, 26779, 113283]
        targnames = [f"HIP {n}" for n in hipnums]
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
        fEZ = sim.ZodiacalLight.fEZ0
        
        # target planet deltaMag (evaluate for a range):
        npoints = 100
        dMags = np.linspace(20, 25, npoints)
        
        # choose angular separation for coronagraph performance
        # this doesn't matter for a flat contrast/throughput, but
        # matters a lot when you have real performane curves
        # we'll use the default values, which is halfway between IWA/OWA
        WA = (mode["OWA"] + mode["IWA"]) / 2
        
        
        # now we loop through the targets of interest and compute intTimes for 
        # each:
        intTimes = np.zeros((len(targnames), npoints)) * u.d
        for j, sInd in enumerate(sInds):
            intTimes[j] = sim.OpticalSystem.calc_intTime(
                sim.TargetList,
                [sInd] * npoints,
                [fZ.value] * npoints * fZ.unit,
                [fEZ.value] * npoints * fEZ.unit,
                dMags,
                [WA.value] * npoints * WA.unit,
                mode,
            )
        
        plt.figure(1)
        plt.clf()
        for j in range(len(targnames)):
            plt.semilogy(dMags, intTimes[j], label=targnames[j])
        
        plt.xlabel(rf"Achievable Planet $\Delta$mag @ {WA :.2f}")
        plt.ylabel(f"Integration Time ({intTimes.unit})")
        plt.legend()
        plt.savefig('../../ctr_out/plot.png')


if __name__ == '__main__':
    x = ErrorBudget()
    x = x.test_json()
#    x = x.load_json()
#    x = x.load_csv_contrast()
#    x = x.compute_ppFact()
