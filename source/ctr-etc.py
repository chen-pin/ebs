"""Exposure time calculator module"""

# Contributors:
#    Pin Chen

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def read_excel():
    sheet_name = ['KPP, no WFE', 'WFE', 'WFSC-PP factor']
    input_dict = pd.read_excel("../inputs/input.xlsx", sheet_name=sheet_name)
    for key in input_dict.keys():
        print(key)
        json_str = input_dict[key].to_json()
        print(json_str)


if __name__ == '__main__':
    read_excel()
