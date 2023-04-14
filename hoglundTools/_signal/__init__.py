# -*- coding: utf-8 -*-
import numpy as np

def nn_correction(data, type='abs', axis=None):
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