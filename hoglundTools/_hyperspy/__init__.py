# -*- coding: utf-8 -*-
"""
Created on Wed May 30 11:27:51 2018

@author: erhog
"""

import sys
import os
import numpy as np
#import dask.array as da
import json
from numpy.typing import NDArray
from tqdm import tqdm
from scipy.signal import correlate, correlation_lags

#TODO: check if hs is installed
from hyperspy.signals import BaseSignal, Signal2D
from exspy.signals import EELSSpectrum, LazyEELSSpectrum
from hyperspy._lazy_signals import LazySignal2D
#from hoglundTools._signal import correlate_1D_in_2D

def is_HyperSpy_signal(signal, signal_type=None):
    '''
    Is the signal a hyperspy signal.
    
    Parameters
    ----------
    signal : object
        Object to be tested.
    signal_type : str
        Additional signal type testing. No additional testing will occur if signal_type is None (default).
        Allowed signal_type can be found at <a href="https://hyperspy.org/hyperspy-doc/current/user_guide/signal.html#id4">https://hyperspy.org/hyperspy-doc/current/user_guide/signal.html#id4</ a>
    '''
    if 'hyperspy' in sys.modules:
        is_signal = isinstance(signal, BaseSignal)
    else:
        return False
    
    if signal_type is None:
        return is_signal
    else:
         return signal.__class__.__name__ == signal_type or signal.__class__.__base__.__name__ == signal_type
         
#TODO: Move to different module.
def correlate_1D_in_2D(data1:NDArray, data2:NDArray, axis:int=-1, normalize:bool=True, kwargs_correlate:dict={}) -> list[NDArray, NDArray]:
    """Perform cross-correlation across a single axis in a 2D signal by performing line by line cross-correlation.
    Data1 and data2 are such that data2 is a refference. If data1 has shifted positive with respect to data 2 then the correlation will return a positive lag.

    Parameters
    ----------
    data1 : NDArray
        2D array. The last two axes are defined as the 2D signal axes.
    data2 : NDArray
        2D array. The last two axes are defined as the 2D signal axes.
    axis : int, optional
        Axis to correlate along. By default -1
    normalize : bool, optional
        Perfrom normalized cross-correlation. By default True.
    kwargs_correlate : dict, optional
        _description_, by default {}

    Returns
    -------
    List
        A list of NDArray. The first is the lags and the second is the correlation coefficients.

    See Also
    --------
    scipy.signal.correlate
    """    
    lags = correlation_lags(data1.shape[axis], data2.shape[axis])
    
    corr = []#np.stack([correlate(s1, s2) for s1, s2 in zip(sig1,sig2)])
    for i, (s1, s2) in enumerate(zip(data1,data2)):
        if normalize:
            s1 = (s1 - np.nanmean(s1)) / np.nanstd(s1)
            s2 = (s2 - np.nanmean(s2)) / np.nanstd(s2)
        corr.append(correlate(s1, s2, **kwargs_correlate) / min(s1.size,s2.size))    
    corr = np.stack(corr)
    
    return lags, corr

def hs1D_to_np(signal):
    '''
    Convert a hyperspy Signal1D.
    
    Parameters
    ----------
    signal : object
        Object to be tested.
    '''
    assert is_HyperSpy_signal(signal)

    return np.stack([np.full(signal.data.shape, signal.axes_manager[-1].axis),
                      signal.data])


def get_hs_image_extent(signal:object, origin:str='upper') -> NDArray:
    """get_hs_image_extent
    Get an array of the image extents for plotting.

    Parameters
    ----------
    signal : object
        _description_
    origin : str, optional
        _description_, by default 'upper'

    Returns
    -------
    NDArray
        Array of extents

    Raises
    ------
    KeyError
        Origin incorrectly set.
    """

    if is_HyperSpy_signal(signal):
        axX = signal.axes_manager[0].axis
        axY = signal.axes_manager[1].axis
        extent = np.asarray(signal.axes_manager.signal_extent)
        
        if origin == 'upper':
            extent[2:] = extent[2:][::-1]
        elif origin == 'lower':
            extent = extent
        else:
            raise KeyError(f'origin must be upper or lower. A value of {origin} was passed.')
        return extent

def load_swift_to_hs(file:str, signal_type:str=None, **kwargs) -> object:
    from warnings import warn
    warn("The `load_swift_to_hs` function has been moved to hoglundTools.io", DeprecationWarning, 2)


def get_wq_rigid_shifts(signal: Signal2D, cumlative:bool=True, calibrated_units:bool=True, kwargs_correlate:dict={}) -> NDArray:
    """
    Measure the rigid energy shift of an omega-q spectrum image.
    The current implementation uses each pixels prior pixel as a refference.
    The current implementation uses cross-correlation and the result returns the shift and correlation coefficent as an (...,2) numpy array.

    Parameters
    ----------
    signal : HyperSpy signal
        Signal2D to align.
    cumlative : bool, optional
        Return cumlative shifts By default True.
    calibrated_units : bool, optional
        Use the energy-axis calibrated units. By default True.
    kwargs_correlate : dict, optional
        kwargs to supply to correlate_1D_in_2D.
        By default, normalize is set to True.
    Returns
    -------
    NDArray
        _description_
    """    
    if 'normalize' not in kwargs_correlate: kwargs_correlate['normalize'] = True
    
    nav_shape = signal.axes_manager.navigation_shape
    signal.unfold_navigation_space()
    shifts = [[0,1]]
    for i, sig in enumerate(tqdm(signal.inav[1:])):
        l,c = correlate_1D_in_2D(sig.data, signal.inav[i].data, **kwargs_correlate)
        c = c.mean(0)
        shifts.append([l[c.argmax()], c[c.argmax()]])
    shifts = np.stack(shifts)
    if calibrated_units:
        shifts[...,0] *= signal.axes_manager['E'].scale
    if cumlative:
        shifts[...,0] = np.cumsum(shifts[...,0])
    shifts = shifts.reshape((*nav_shape[::-1],-1))
    
    signal.fold()
    return shifts

def shift_Signal2D_along_axis(signal:object, shift_array:NDArray, axis:str|int='E',
                               inplace:bool=True, subpixel:bool=True,
                               kwargs_shit1D:dict={}) -> Signal2D:
    """Rigidly shift a 2D signal along a single axis.

    Parameters
    ----------
    signal : object
        Hyperspy 2D signal to align.
    shift_array : NDArray
        Array containing the shift pixels.
    axis : str | int, optional
        Axis to shift along, by default 'E'
    inplace : bool, optional
        Perform the operation and store in the supplied signal, by default True
    subpixel : bool,optional
        Currently not implemented.
        If true, the `shift1D` opperation is used to shift and will interpolate where necesary.
        If false, ... TODO: shift by the pixel amount and do not interpolate.
    kwargs_shit1D : dict, optional
        kwargs for the hyperspy shift1D function, by default {}

    Returns
    -------
    Signal2D
        Aligned signal if inplace is True.
    """    
    if not inplace:
        signal = signal.deepcopy()
    if 'expand' not in kwargs_shit1D: kwargs_shit1D['expand'] = False
    if 'crop' not in kwargs_shit1D: kwargs_shit1D['crop'] = True
    
    og_sig_axes = np.array([ax.name for ax in signal.axes_manager.signal_axes])
    other_sig_axis = og_sig_axes[og_sig_axes!=axis][0]
    
    shift_array = np.stack([shift_array]*signal.axes_manager[other_sig_axis].size, axis=-1)
    
    signal = signal.as_signal1D(axis)
    if subpixel:
        signal.shift1D(shift_array=-shift_array, **kwargs_shit1D)

    signal = signal.as_signal2D(og_sig_axes)
    if not inplace:
        return signal


def roi_line_info(line, print_info=True, return_dict=False):
    """roi_line_info
    Prints details of a line

    Parameters
    ----------
    line : object
        hyperspy line annotation
    print_info : bool, optional
        by default True
    return_dict : bool, optional
        by default False

    Returns
    -------
    dict
        Information about the line if return_dict is True
    """
    import numpy as np
    
    dx = line.x2-line.x1
    dy = line.y2-line.y1
    angle = np.arctan(dy/dx)
    angle_degree = angle *180/np.pi
	
    if print_info:
        print('Start:  ({},{}) [px]'.format(line.x1, line.y1))
        print('Finish: ({},{} [px])'.format(line.x2, line.y2))
        print('Width:  {} [px]'.format(line.linewidth))
        print('dx:     {}'.format(dx))
        print('dy:     {}'.format(dy))
        print('Length: {} [px]'.format(line.length))
        print('Angle:  {} [rad]'.format(angle))
        print('     :  {} [degree]'.format(angle_degree))
    if return_dict:
        return {'dx': dx,
		'dy': dy,
		'length': line.length,
		'angle': angle,
		'angle_degree': angle_degree}