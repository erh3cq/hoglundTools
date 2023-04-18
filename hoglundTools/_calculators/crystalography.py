import numpy as np

def bragg_reflection(hkl, lattice, units_in='A', units_out='1/A', twoPi = True, printout=True):
    if isinstance(hkl, int):
        N = hkl
    elif isinstance(hkl, list) and len(hkl)==3:
        N = hkl[0]**2 + hkl[1]**2 + hkl[2]**2
    else:
        raise Exception('hkl should be an integer N or a list with length three.')
        
    if units_in=='A':
        r_scale = 1E-10
    elif units_in=='nm':
        r_scale = 1E-9
    if units_out=='1/A':
        q_scale = 1E-10
    elif units_out=='1/nm':
        q_scale = 1E-9

    d = lattice/np.sqrt(N) * r_scale
    q = 1/d * q_scale
    
    if twoPi:
        q = 2*np.pi * q

    if printout:
        print('d: {:.2f} {}'.format(d/r_scale ,units_in))
        print('q: {:.2f} {}'.format(q, units_out))
    return q

