import csv
import numpy as np
import os
import json as js


def read_csv(filename):
    data = np.loadtxt(open(filename, "rb"), delimiter=',', skiprows=0)
    return data


def generate_pp_json(json_file, wfe, wfsc, sensitivity):
    """
    overwrites the original json file
    :param json_file:
    :param wfe:
    :param wfsc:
    :param sensitivity:
    :return:
    """
    with open(json_file) as f:
        input_dict = js.load(f)
    input_dict['wfe'] = wfe.tolist()
    input_dict['wfsc_factor'] = wfsc.tolist()
    input_dict['sensitivity'] = sensitivity.tolist()
    with open(json_file, 'w') as f:
        js.dump(input_dict, f, indent=4)
