import csv
import numpy as np
import os
import json as js


def read_csv(filename, skiprows=0):
    """
    returns the data contained in a csv file as a numpy array.
    Values are separated by commas and rows are separated by newlines
    :param filename: str, fully qualified path to the csv file
    :param skiprows: int, number of rows to skip at the top of the csv file
    :return: numpy array, data contained in the csv file
    """
    data = np.loadtxt(open(filename, "rb"), delimiter=',', skiprows=skiprows)
    return data


def update_pp_json(json_file, wfe, wfsc, sensitivity, contrast_path, throughput_path):
    """
    appends the wavefront error, sensitivity, and sensing and control coefficient to an existing JSON file. Also makes
    sure file paths in the JSON file are correct.

    Overwrites the original JSON file.
    :param json_file: str, fully qualified path to a json file
    :param wfe: array or list, wavefront error data
    :param wfsc: array or list, wavefront sensing and control data
    :param sensitivity: array or list, sensitivity coefficient data
    :return: None
    """
    with open(json_file) as f:
        input_dict = js.load(f)

    input_dict['starlightSuppressionSystems'][0]['core_thruput'] = throughput_path
    input_dict['starlightSuppressionSystems'][0]['core_contrast'] = contrast_path
    input_dict['wfe'] = wfe.tolist()
    input_dict['wfsc_factor'] = wfsc.tolist()
    input_dict['sensitivity'] = sensitivity.tolist()
    with open(json_file, 'w') as f:
        js.dump(input_dict, f, indent=4)
