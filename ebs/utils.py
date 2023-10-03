import csv
import numpy as np
import os


def read_csv(filename):
    data = np.loadtxt(open(filename, "rb"), delimiter=',', skiprows=0)
    return data

def generate_pp_json(wfe, wfsc, sensitivity, json):
    # TODO create this
    return None
