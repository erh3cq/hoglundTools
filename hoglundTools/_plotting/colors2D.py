from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt
import numpy as np


def get_color_wheel(N=301, phi=0, vmax=None, clear=True):
    if vmax is None:
        r=1
    else:
        r=vmax
    x = np.linspace(-r,r,N)
    x,y = np.meshgrid(x,x, indexing='ij')
    phi = np.deg2rad(phi)
    
    comp = x+1j*y
    rgb=np.zeros(x.shape+(4,),dtype=float)
    
    hsv = np.stack([(np.angle(comp)+phi)/(2*np.pi)%1, np.ones_like(x), np.abs(comp)/r], axis=-1)
    rgb = hsv_to_rgb(hsv)
    rgbt = np.concatenate((rgb, np.ones(x.shape+(1,))), axis=-1)
    if clear:
        mask = np.any(rgbt[...,:3]>1, axis=-1)
        rgbt[mask, -1] = 0
    
    return rgbt

def plot_color_wheel(ax, N=301, phi=0, vmax=None, clear=True):
    rgbt = get_color_wheel(N=N, phi=phi, vmax=vmax, clear=clear)
    ax.imshow(rgbt, extent=[-1,1,-1,1], origin='lower')

def get_color_hexagon(N=301, phi=0, clear=True):
    rgbt = get_color_wheel(N=N, phi=phi, clear=True)
    x = np.linspace(-1,1,N)
    x,y = np.meshgrid(x,x, indexing='xy')
    
    mask_side = np.abs(x)>np.cos(np.deg2rad(30))
    rgbt[mask_side,-1] = 0
    mask_top = np.abs(y) > 1-np.abs(x)*np.tan(np.deg2rad(30))
    rgbt[mask_top, -1] = 0
    
    return rgbt

def plot_color_hexagon(ax, N=301, phi=0, clear=True, labels=None, labelpad=0.1, font_kwargs={}):
    rgbt = get_color_hexagon(N=N, phi=phi, clear=clear)
    ax.imshow(rgbt, extent=[-1,1,-1,1], origin='lower')
    if labels is not None:
        if len(labels) == 3:
            deg = np.deg2rad(30)
            ax.text(0, 1-labelpad, labels[0], ha='center', va='center', **font_kwargs)
            ax.text(np.cos(deg)*(1-labelpad), (labelpad-1)*np.sin(deg), labels[1], ha='center', va='center', **font_kwargs)
            ax.text(-np.cos(deg)*(1-labelpad), (labelpad-1)*np.sin(deg), labels[2], ha='center', va='center', **font_kwargs)

def plot_rgb_traingle(ax, N=301, labels=None, labelpad=0.1, font_kwargs={}, scheme='rgb'):
    '''
    Plot a Maxwell tiangle with RGB at the verticies
    '''
    img = np.zeros((N,N,4))
    dx = 2.0/N
    dy = 1.0/N
    x = np.linspace(-1,1,N)
    y = np.linspace(0,1,N)

    x,y = np.meshgrid(x,y, indexing='ij')

    r = y
    g = (x+1-r)/2
    b = 1.0-g-r
    t = np.zeros_like(r)
    rgbt = np.stack((r,g,b,t)).T
    if scheme == 'cmy': rgbt = [1,1,1,0] - rgbt

    mask1 = np.all(rgbt[...,:3]>=0, axis=-1)
    mask2 = np.all(rgbt[...,:3]<=1, axis=-1)
    mask = np.logical_and(mask1, mask2)
    rgbt[mask, -1] = 1
    
    a = 1.0/np.sqrt(3)
    ax.imshow(rgbt, origin='lower',extent=[-a,a,-1/3,2/3])
    ax.set_aspect('equal')
    
    if labels is not None:
        if len(labels) == 3:
            deg = np.deg2rad(30)
            ax.text(0, 2/3-labelpad, labels[0], ha='center', va='center', **font_kwargs)
            ax.text(a - np.cos(deg)*labelpad, -1/3 + np.sin(deg)*labelpad, labels[1], ha='center', va='center', **font_kwargs)
            ax.text(-a + np.cos(deg)*labelpad, -1/3 + np.sin(deg)*labelpad, labels[2], ha='center', va='center', **font_kwargs)