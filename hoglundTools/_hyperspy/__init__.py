# -*- coding: utf-8 -*-
"""
Created on Wed May 30 11:27:51 2018

@author: erhog
"""

import sys
import numpy as np

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
        is_signal = isinstance(signal, sys.modules['hyperspy'].signal.BaseSignal)
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