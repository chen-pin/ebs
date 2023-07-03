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

    def __init__(self, input_filename="test.json"):
        self.input_dir = os.path.join("..", "inputs")
        self.input_path = os.path.join(self.input_dir, input_filename)

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

    def load_json(self):
        with open(os.path.join(self.input_path)) as input_json:
            input_dict = js.load(input_json)
           # print(input_dict)
            print(input_dict['modules'].keys())

    def test_json(self):
        # build sim object:
        sim = ems.MissionSim(str(self.input_path))
        
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
