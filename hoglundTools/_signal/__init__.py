# -*- coding: utf-8 -*-
import numpy as np
from scipy.signal import correlate, correlation_lags
from hoglundTools._hyperspy import is_HyperSpy_signal


def nv_correction(data, type='abs', axis=None):
    '''
    Function to correct for negative values.

    Parameters
    ----------
    data: array-like
        Data to be corrected.
    type: str
        abs - correct data by taking its absolute value.
        offset - coorect data by subtracting the minimum value
    axis: None, int, tuple of int
        Axis to perform noramlization with.
        If None then the full signal is used and the normalization is relative to all data.
        If for example -1, the last axis is normalized independently from the next (eg. each spetra).
    '''
    if type=='abs':
         return np.abs(data)
    elif type=='offset':
        min = np.nanmin(data)
        return data - min
    
def normalize_by_ZLP(data, threshold=3., type='area', zero_base=True, axis=-1):
    if is_HyperSpy_signal:
        ZLP_I = normalize(data.isig[:threshold], type='area', zero_base=True, axis=-1)
    else:
        ZLP_I = normalize(data[:threshold], type='area', zero_base=True, axis=-1)

    return data/ZLP_I

def normalize(data, type='area', zero_base=True, axis=-1):
    '''
	Normalize the data.

	Parameters
    ----------
    data: numpy-like
		Data to be normalized.
    type: str
		area - Normalize the area to 1.
		max  - Normalize so the maximum value is 1.
	zero_base: boolean
		Subtract the minimum value before normalization.
	'''
    
    if zero_base:
        min = np.nanmin(data)
        data =  data - min
    if type=='area': 
        return data/np.nansum(data, axis=axis)
    elif type=='max':
         return data/np.nanmax(data, axis=axis)
    raise AttributeError(f"{type!r} is not a valid normalization type.")