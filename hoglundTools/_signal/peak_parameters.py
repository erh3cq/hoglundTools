# -*- coding: utf-8 -*-
"""
Created on Sun June 3 18:49:00 2018

@author: Eric Hoglund
"""
import numpy as np

from hoglundTools._hyperspy import is_HyperSpy_signal


def estimate_FWPM(data, percent=0.5, E=None, return_sides=False, verbose=None):
    '''
    Estimate the full width of a peak at a percent of its height.

    Parameters
    ----------
    data: float
        Data containing a peak.
    percent: float
        Fractional height of the peak where the width will be measured.
    E: numpy-like, None
        Energy-axis array.
        If data is a HyperSpy signal and E is None then E will be infered from the signal.
        If None then the falues will be returned in relative pixel values.
    return_sides: boolean
        Return the bounds of the peak at the fractional intensity.
    verbose: None, str
        info - print information about the peak.
    
    '''
    hs_sig = is_HyperSpy_signal(data)

    I_min = np.nanmin(data, axis=-1)
    I_max = np.nanmax(data, axis=-1)
    I_diff = I_max - I_min
    I_p = I_diff * percent
    if verbose in ['info']:
        print('I_min:  {:.2f}'.format(I_min))
        print('I_max:  {:.2f}'.format(I_max))
        print('I_diff: {:.2f}'.format(I_diff))
    
    if E is None and hs_sig:
        E = data.axes_manager[-1].axis
    elif E is None:
        E = np.arange(data.shape([-1]))
    # if E is None:
    #     raise ValueError('E was not set and data is not a HyperSpy signal.')

    E_max_i = np.nanargmax(E)
    E_max_v = E[E_max_i]
    if verbose in ['info']:
        print('Peak maximum: {}'.format(E_max_v))
    #pos_extremum = s.valuemax(-1).data.mean() #TODO: use pos_extremum = s.valuemax(-1) for nearest above and below

    s = np.abs(data-I_p[...,None])
    
    if hs_sig:
        r = s.isig[E_max_v:].valuemin(-1).as_signal1D(0) #TODO: add capabilits for image
        l = s.isig[:E_max_v].valuemin(-1).as_signal1D(0) #TODO: add capabilits for image

    
    if return_sides:
        return r-l, l ,r
    else:
        return r-l

def estimate_FWHM_center(data):
    _, l, r = estimate_FWPM(data, percent=0.5, return_sides=True)
    return (l+r)/2

def estimate_skew(s):
    return estimate_FWHM_center(s) - s.valuemax(-1).as_signal1D(0)