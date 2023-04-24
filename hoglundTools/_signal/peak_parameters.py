# -*- coding: utf-8 -*-
"""
Created on Sun June 3 18:49:00 2018

@author: Eric Hoglund
"""
import numpy as np

from hoglundTools._hyperspy import is_HyperSpy_signal

def _estimate_FWPM(data, E, percent=0.5, verbose=None):
    '''
    Estimate the full width of a peak at a percent of its height.

    Parameters
    ----------
    data: numpy-array
        Data containing a peak.
    percent: float
        Fractional height of the peak where the width will be measured.
    E: numpy-like, None
        Energy-axis array.
    verbose: None, str
        info - print information about the peak.
    '''
    
    E_max_i = np.nanargmax(data, axis=-1)
    E_max_v = E[E_max_i]
    if verbose in ['info', 'all']: print(f'E_max: {E_max_v:.2f}')

    I_max = np.nanmax(data, axis=-1)
    I_p = I_max * percent
    if verbose in ['info', 'all']: print(f'I_max: {I_max:.2f}')

    data = np.abs(data-I_p[...,None])

    l = np.nanargmin(data[:E_max_i])
    l = E[:E_max_i][l]
    r = np.nanargmin(data[E_max_i:])
    r = E[E_max_i:][r]

    return r-l
    
def _estimate_LSPM(data, E, percent=0.5, E_max_i=None, verbose=None):
    '''
    Estimate the full width of a peak at a percent of its height.

    Parameters
    ----------
    data: array-like
        Data containing a peak.
    percent: float
        Fractional height of the peak where the width will be measured.
    E: numpy-like, None
        Energy-axis array.
'''
    
    # if E_max_i is None:
    #     raise AttributeError('Neither E nor E_max_i where set.')
    

    l = np.nanargmin(data[:E_max_i])
    l = E[:E_max_i][l]

def estimate_FWPM(data, percent=0.5, E=None, verbose=None, hs_kwargs={'inplace':False}):
    '''
    Estimate the full width of a peak at a percent of its height.

    Parameters
    ----------
    data: array-like
        Data containing a peak.
    percent: float
        Fractional height of the peak where the width will be measured.
    E: numpy-like, None
        Energy-axis array.
        If None then the values will be returned in relative pixel values.
        If data is a HyperSpy signal and E is None then E will be infered from the signal.
    verbose: None, str
        info - print information about the peak.

    hs_kwargs
        Key words to pass to map if data is a HyperSpy signal. 
    '''

    if is_HyperSpy_signal(data):
        if verbose in ['all']: print('Running as HS sig')
        fwpm = data.map(_estimate_FWPM, E=data.axes_manager[-1].axis, percent=percent, verbose=verbose, **hs_kwargs)
    else:
        if E is None:
            if verbose in ['all']: print('E is None. The returned values will be in relative indicies.')
            E = np.arange(data.shape[-1])

        shape = np.shape(data)
        data = data.reshape(-1, shape[-1])
        fwpm = np.asarray([_estimate_FWPM(d, E=E, percent=percent, verbose=verbose) for d in data], dtype=float).reshape(shape[:2])
        
    return fwpm
    
def _estimate_FWPM_center(data, E, percent=0.5, verbose=None):
    '''
    Estimate the peak center at a percent of its height.

    Parameters
    ----------
    data: numpy-array
        Data containing a peak.
    percent: float
        Fractional height of the peak where the width will be measured.
    E: numpy-like, None
        Energy-axis array.
    verbose: None, str
        info - print information about the peak.
    '''
    
    E_max_i = np.nanargmax(data, axis=-1)
    E_max_v = E[E_max_i]
    if verbose in ['info', 'all']: print(f'E_max: {E_max_v:.2f}')

    I_max = np.nanmax(data, axis=-1)
    I_p = I_max * percent
    if verbose in ['info', 'all']: print(f'I_max: {I_max:.2f}')

    data = np.abs(data-I_p[...,None])

    l = np.nanargmin(data[:E_max_i])
    l = E[:E_max_i][l]
    r = np.nanargmin(data[E_max_i:])
    r = E[E_max_i:][r]

    return (l+r)/2

# def estimate_FWPM_center(data):
#     _, l, r = estimate_FWPM(data, percent=0.5, return_sides=True)
#     return (l+r)/2

def estimate_FWPM_center(data, percent=0.5, E=None, verbose=None, hs_kwargs={'inplace':False}):
    '''
    Estimate the peak center at a percent of its height.

    Parameters
    ----------
    data: array-like
        Data containing a peak.
    percent: float
        Fractional height of the peak where the width will be measured.
    E: numpy-like, None
        Energy-axis array.
        If None then the values will be returned in relative pixel values.
        If data is a HyperSpy signal and E is None then E will be infered from the signal.
    verbose: None, str
        info - print information about the peak.

    hs_kwargs
        Key words to pass to map if data is a HyperSpy signal. 
    '''

    if is_HyperSpy_signal(data):
        if verbose in ['all']: print('Running as HS sig')
        fwpm = data.map(_estimate_FWPM_center, E=data.axes_manager[-1].axis, percent=percent, verbose=verbose, **hs_kwargs)
    else:
        if E is None:
            if verbose in ['all']: print('E is None. The returned values will be in relative indicies.')
            E = np.arange(data.shape[-1])

        shape = np.shape(data)
        data = data.reshape(-1, shape[-1])
        fwpm = np.asarray([_estimate_FWPM_center(d, E=E, percent=percent, verbose=verbose) for d in data], dtype=float).reshape(shape[:2])
        
    return fwpm

def estimate_skew(data):
    return estimate_FWPM_center(data) - data.valuemax(-1).as_signal1D(0)