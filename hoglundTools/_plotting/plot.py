# -*- coding: utf-8 -*-
"""
Created on Wed May 30 11:25:15 2018

@author: erhog
"""
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import numpy as np

from hoglundTools._hyperspy import is_HyperSpy_signal
from hoglundTools._signal import nn_correction

def abc(axs, pos=[-.2, 1], end=')', title=False, **kwargs):
    '''
    Adds an alphabetical label to the figures.
    '''
    if title and 'loc' not in kwargs.keys():
        kwargs['loc'] = 'left'
    alpha = 'abcdefghijklmnopqrstuvwxyz'
    axs = axs.flatten()
    for i, ax in enumerate(axs):
        label = alpha[i]+end
        if title:
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


def plot_image(data, ax=None, norm=None, nn_correction=None, nn_correction_kwargs={}, **kwargs):
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
    nn_correction: boolean
        Correct for negative vlaues.
    nn_correction_kwargs: dict
        kwargs for nn_correction.

    **kwarg:
        kwargs supplied to imshow.
    """
    if ax is None:
        ax = plt.gca()

    if is_HyperSpy_signal(data, signal_type='Signal2D'):
        axX = data.axes_manager[-1].axis
        axY = data.axes_manager[0].axis
        
        if 'origin' in kwargs and kwargs['origin']=='lower':
            kwargs['extent'] = [axX.max(), axX.min(), axY.max(), axY.min()]
        else:
            kwargs['extent'] = [axX.min(), axX.max(), axY.max(), axY.min()]

    if nn_correction is not None:
        data = nn_correction(data, **nn_correction_kwargs)
        
    img = ax.imshow(data, **kwargs)

    #TODO: add scale_bar

    if ax is None:
        return img, ax
    else:
        return img