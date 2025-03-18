import numpy as np


def read_csv(filename, skiprows=0):
    """Returns the data contained in a CSV file as a numpy array.

    Values are separated by commas and rows are separated by newlines

    Parameters
    ----------
    filename: str
        Fully qualified path to the CSV file.
    skiprows: int
         number of rows to skip at the top of the CSV file.

    Returns
    -------
    data: np.ndarray
        Data contained in the CSV file.
    """
    data = np.loadtxt(open(filename, "rb"), delimiter=',', skiprows=skiprows)
    return data


def write_csv(savename, data, header):
    """Write CSV header and data to savename"""
    np.savetxt(savename, data, delimiter=',', header=header)
