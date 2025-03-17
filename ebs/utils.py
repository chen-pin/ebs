import numpy as np


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


def write_csv(savename, data, header):
    np.savetxt(savename, data, delimiter=',', header=header)
