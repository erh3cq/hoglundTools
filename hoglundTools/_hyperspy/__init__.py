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

#TODO: check if hs is installed
from hyperspy.signals import BaseSignal, EELSSpectrum, Signal2D
from hyperspy._lazy_signals import LazyEELSSpectrum, LazySignal2D


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


def get_hs_image_extent(signal, origin='upper'):
    if is_HyperSpy_signal(signal):
        axX = signal.axes_manager[0].axis
        axY = signal.axes_manager[1].axis
        
        if origin == 'upper':
            return [axX.min(), axX.max(), axY.max(), axY.min()]
        elif origin == 'lower':
            return [axX.min(), axX.max(), axY.min(), axY.max()]
        else:
            raise KeyError(f'origin must be upper of lower. A value of {origin} was passed.')

def load_swift_to_hs(file, signal_type='image', **kwargs):
    f = open(file+'.json'); meta = json.load(f); f.close()
    data = np.load(file+'.npy') #TODO: load as dask when lazy. I couln not figure out how to load a specific npy file as dask, only a directory of npy files.
    
    flag_sig_type = meta['metadata']['hardware_source'].get('signal_type')
    flag_time_series = meta.get('is_sequence')
    
    nav_dim = meta['collection_dimension_count']
    sig_dim = meta['datum_dimension_count']


    axes = meta['spatial_calibrations']
    
    for i, ax in enumerate(axes):
        ax['size'] = data.shape[i]
    axes_order = []

    if flag_time_series:
        axes[0]['name'] = 'time'
        axes[0]['units'] = 'frame'
        axes_order.append(0)

    axes_rspace_i = [i for i, ax in enumerate(axes) if ax['units']=='nm']
    rdim = len(axes_rspace_i)

    for i,j in enumerate(axes_rspace_i):
        axes[j]['name'] = 'zyx'[3-rdim:][i]
        axes_order.append(j)
        
    if flag_sig_type == 'eels':
        axes[-1]['name'] = 'E'
        axes_order.append(-1)
        if meta['datum_dimension_count'] == 2:
            axes[-2]['name'] = 'q'
            axes[-2]['units'] = 'px'
            axes_order.append(-2)
    
    filename = os.path.split(file)[-1]
    metadata = {'General': {'title': filename.split('.')[0], 'original_filename':filename}}

    if flag_sig_type == 'eels' and sig_dim == 1:
        sig = EELSSpectrum(data, axes=[axes[i] for i in axes_order], metadata=metadata, **kwargs)
    if sig_dim == 2:
        sig = Signal2D(data, axes=[axes[i] for i in axes_order], metadata=metadata, **kwargs)

    if 'lazy' in kwargs:
        sig = sig.as_lazy()
        
    return sig

def roi_line_info(line, print_info=True, return_dict=False):
    """
	Prints details of a line
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