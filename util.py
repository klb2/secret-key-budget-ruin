import numpy as np
import pandas as pd


def db_to_linear(value):
    return 10**(np.array(value)/10.)

def linear_to_db(value):
    return 10*np.log10(value)

def capacity(snr):
    snr = np.array(snr)
    return np.log2(1+snr)

def find_closest_element_idx(array, value):
    closest_element = min(array, key=lambda x: abs(x-value))
    idx = np.where(array == closest_element)[0]
    return idx


def export_results(results, filename):
    df = pd.DataFrame.from_dict(results)
    df.to_csv(filename, sep='\t', index=False)
