# -*- coding: utf-8 -*-
"""
Created on Sun June 3 18:49:00 2018

@author: Eric Hoglund
"""
import numpy as np

def estimate_FWHM(s, return_sides=False):
    difference = s.max(-1) - s.min(-1)
    HM = difference / 2
    
    pos_extremum = s.valuemax(-1).data.mean() #TODO: use pos_extremum = s.valuemax(-1) for nearest above and below
    s= np.abs(s-HM)
    
    print('Position used for peak maximum: {}'.format(pos_extremum, s.axes_manager[-1].units))
    
    nearest_above = s.isig[pos_extremum:-1].valuemin(-1).as_signal1D(0) #TODO: add capabilits for image
    nearest_below = s.isig[0:pos_extremum].valuemin(-1).as_signal1D(0) #TODO: add capabilits for image

    
    if return_sides:
        return nearest_above-nearest_below, nearest_below ,nearest_above
    else:
        return nearest_above-nearest_below

def estimate_FWHM_center(s):
    _, left, right = estimate_FWHM(s, return_sides=True)
    return (left+right)/2

def skew(s):
    return estimate_FWHM_center(s) - s.valuemax(-1).as_signal1D(0)