"""Exposure time calculator module"""

# Contributors:
#    Pin Chen

import os 
import numpy as np
import json as js
#import matplotlib.pyplot as plt
#import pandas as pd
import EXOSIMS.MissionSim as ems


class ErrorBudget(ems.MissionSim):
    """Sub-class of `EXOSIMS.MissionSim.MissionSim` to compute exposure time"""

    def __init__(self, input_filename="sample.json"):
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

if __name__ == '__main__':
    x = ErrorBudget()
    x.load_json()
