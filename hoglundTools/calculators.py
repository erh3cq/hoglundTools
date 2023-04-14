# -*- coding: utf-8 -*-
"""
Created on Sun May 27 18:09:25 2018

@author: Eric Hoglund
"""
import numpy as np

def bragg_reflection(hkl, lattice, units_out='A', printout=True):
    if isinstance(hkl, int):
        N = hkl
    elif isinstance(hkl, list) and lne(hkl)==3:
        N = hkl[0]**2 + hkl[1]**2 + hkl[2]**2
    else:
        raise Exception('hkl should be an integer N or a list with length three.')
        
    if units_out=='A':
        d = lattice/np.sqrt(N)
    elif units_out=='nm':
        d = lattice/np.sqrt(N)/10
    
    q = 2*np.pi/d
    if printout:
        print('d: ',d,units_out)
        print('q: ',q,units_out,'-')
    return q

def q_to_theta(q, wave_length=0.0197):
	return wave_length*q/2
    
if __name__ == "__main__":
    print(len([0,1,2]))