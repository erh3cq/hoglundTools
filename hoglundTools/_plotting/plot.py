# -*- coding: utf-8 -*-
"""
Created on Wed May 30 11:25:15 2018

@author: erhog
"""
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

from hoglundTools._hyperspy import is_HyperSpy_signal, get_hs_image_extent
from hoglundTools._signal import nv_correction

def closest_nice_number(number, round_cuttoff=8):
        oom = 10 ** np.floor(np.log10(number)) #order of magnitude
        if number/oom > round_cuttoff:
            return oom*10
        else:
            return oom * (number // oom)


class add_scale_bar(AnchoredSizeBar):
    def __init__(self, ax, size=None, label=None,
                 color='white', size_vertical='1%', kw_scale={}, kw_font={},
                 pixel_size=None, units=None, max_size_ratio=0.25):
        '''
        Add an axes bar
        
        Parameters
        ----------
        ax: matplotli.axis
            Axis to add scale bar to.
        size: flaot
            Size of the scale bar in data cordinates.
        label: str
            Label to display above scale bar.
        color: str
            Color of the bar and the text.
        kw_scale: dict
            Dictionary of kwargs to supply to AnchoredSizeBar.
        kw_font: dict
            Dictionary of kwargs to supply to FontProperties.
        '''
        self.ax = ax
        self.pixel_size = pixel_size
        self.units = units if units is not None else ''
        self.size = size if size is not None else self.calculate_size(max_size_ratio=max_size_ratio)
        self.size_vertical = self.calculate_vertical_size(size_vertical) if size_vertical is not None else 0
        self.label = label if label is not None else self.generate_label()
        self.color = color
    


        fontprops = {'size': 8}
        for k,i in kw_font.items():
            fontprops[k] = i
        
        scaleprops = {'pad':0.1,
                      'color':self.color,
                      'frameon':False,
                      'label_top':True,
                      'size_vertical':self.size_vertical,
                       'loc':'lower left'}
        for k,i in kw_scale.items():
            scaleprops[k] = i
            
        
        fontprops = fm.FontProperties(**fontprops)
        scalebar = AnchoredSizeBar(
                ax.transData,
                self.size, self.label,
                fontproperties=fontprops,
                **scaleprops)
        #scalebar0.size_bar.get_children()[0].fill = True
        ax.add_artist(scalebar)


    def calculate_size(self, max_size_ratio=0.25):
        '''Calculate the size of the bar.'''
        xlims = self.ax.set_xlim()
        size = closest_nice_number(np.ptp(xlims) * max_size_ratio)
        self.size = size
        return size
    
    def calculate_vertical_size(self, size_vertical):
        '''Calculate the vertical size of the bar.

        Parameters
        ----------
        size_vertical: str, int, flaot
            If str then the value will be calculated based on the y-axis extents.
            If the last character is % then the string will be converted to a percent, otherwise it is assumed the value is a desired fraction.
            If int or float then absolute units are assumed and the input value is returned.
        '''
        yax_size = np.ptp(self.ax.set_ylim())
        if isinstance(size_vertical, str):
            if size_vertical[-1] == '%':
                size_vertical = yax_size * float(size_vertical[:-1]) / 100
            else:
                size_vertical = yax_size * float(size_vertical)
        return size_vertical


        size = closest_nice_number(np.ptp(xlims) * max_size_ratio)
        self.size = size
        return size
    
    def generate_label(self):
        if self.size%1 == 0:
            return f'{int(self.size):d} ' + self.units
        else:
            return f'{round(self.size, 1):.1f} ' + self.units


class   plot_image(object):
    def __init__(self, data, ax=None, norm=None, fix_nv=None,
                 ticks_and_labels='off', scale_bar=True,
                 fix_nv_kwargs=None, scale_bar_kwargs=None , **kwargs):
        """
         A shortcut plotting function for imshow that automatically handles things like imshow kwargs and intenisty bounds.
            
        Parameters
        ----------
        data: array-like or Hyperspy Signal
            Two dimintional data set to plot as an image.
            If a Hyperspy signal is used then extents will be interpreted.
        ax: matplotlib axis
            Axis to plot to. If None then an axis is both created and returned.
        norm: matplotlib normalization
            Normalization to be used. Default is None.
        fix_nv: boolean
            Correct for negative vlaues.
        ticks_and_labels: Str
            What to do with the axes ticks and borders.
            off:    turn off the axes (default).
            empty:  turn the ticks and labels off but leave the borders.
            on:     keep the ticks and borders in on.
        add_scale_bar: boolean
            Add a scalebar object to the image if True. Default if True.
            For brevity, if scale_bar_kwargs is populated then scal_bar is set to True.
        
        fig_nv_kwargs: dict
            kwargs for nn_correction.
        scale_bar_kwargs: dict
            kwargs for scale_bar.
            
        **kwarg:
            kwargs supplied to imshow.
        """
        
        self.ax = ax if ax is not None else plt.gca()
        self._hs_sig = is_HyperSpy_signal(data, signal_type='Signal2D')
        self.norm = norm
        self.ticks_and_labels = ticks_and_labels

        origin = 'upper' if kwargs.get('origin') is None else kwargs['origin']
        if self._hs_sig and kwargs.get('extent') is None: kwargs['extent'] = get_hs_image_extent(data, origin=origin)
        self.extent = kwargs.get('extent')



        #TODO: make dynamic
        if fix_nv is not None:
            data = nv_correction(data, **fix_nv_kwargs)

        self.img = self.ax.imshow(data, norm=self.norm, **kwargs) #TODO: make updatable with show function that is then called in __init__

        # Add scale_bar        
        scale_bar = True if scale_bar or scale_bar_kwargs is not None else False
        if scale_bar_kwargs is None: scale_bar_kwargs = {}
        if scale_bar:
            if scale_bar_kwargs.get('units') is None and self._hs_sig:
                scale_bar_kwargs['units'] = data.axes_manager.signal_axes[0].units
            self.scale_bar = add_scale_bar(self.ax, **scale_bar_kwargs)

        if self.ticks_and_labels is not None: self.set_ticks_and_labels(ticks_and_labels)

    def set_ticks_and_labels(self, state):
        if state == 'off':
            self.ax.axis('off')
        elif state == 'empty':
            self.ax.set_yticks([])
            self.ax.set_xticks([])
        elif state == 'on':
            pass

def save_fig(file_name, fig=None, file_types=['svg','png'], **kwargs):
    if fig is None: fig = plt.gcf()
    for ft in file_types:
        fig.savefig(file_name+'.'+ft, **kwargs)


def pannel_title(axs, pos=[-.2, 1], end='', title=False, **kwargs):
    '''
    Adds an alphabetical label to the figure pannels.
    '''
    alpha = 'abcdefghijklmnopqrstuvwxyz'

    for i, ax in enumerate(axs.flatten()):
        label = alpha[i]+end
        if title:
            if 'loc' not in kwargs.keys():
                kwargs['loc'] = 'left'
            ax.set_title(label, **kwargs)
        else:
            ax.text(pos[0], pos[1], label, transform=ax.transAxes,
                    fontweight='bold', va='top', ha='right', **kwargs)

def legend_2axis(axes, labels='auto', display_axis=0, **kwargs):
    '''
    Plot a legend for a multi-axis subplot.
    
    Parameters
    ----------
    axes : tuple or list
        List of axes containing labels for the legend.
    labels : bool
        If True, the ionization edges with an onset below the lower
        energy limit of the SI will be included.
    kwargs: dict
        kwargs for the matplotlib legend function.
    '''
    lines = []
    labels = []
    for ax in axes:
        li, la = ax.get_legend_handles_labels()
        lines += li
        labels += la
    
    if labels == 'auto':
        axes[0].legend(lines,  labels, **kwargs)
    elif labels is not None:
        axes[0].legend(lines,  labels, **kwargs)