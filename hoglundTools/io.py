# -*- coding: utf-8 -*-

import json
import sys
import os
import numpy as np
import h5py
import json
from numpy.typing import NDArray
from tqdm import tqdm

class swift_json_reader:
    
    def __init__(self, file:str, signal_type:str=None, get_npy_shape:bool=True, **kwargs):
        #File handling
        self.file = file
        self.filename = os.path.split(file)[-1]
        with open(file+'.json') as f: self.meta = json.load(f)

        #General metadata
        self.title = self.meta.get('title')
        self.signal_type = signal_type if signal_type is not None else self.read_signal_type()

        #Axis handling
        self.is_series = x if self.meta.get('is_sequence') is not None else False
        self.nav_dim = self.meta['collection_dimension_count']
        self.sig_dim = self.meta['datum_dimension_count']
        self.data_shape = np.load(file+'.npy', mmap_mode='r').shape

        self.axes = self.read_axes_calibrations()
        if get_npy_shape:
            for ax, d in zip(self.axes, self.data_shape): ax['size'] = d
        if self.is_series and self.axes[0]['units']=='': self.axes[0]['units'] = 'frame'
        self.infer_axes_names()

        #Instrument metadata
        self.beam_energy = self.meta['metadata']['instrument'].get('high_tension')/1E3
        self.scan_rotation = self.meta['metadata']['scan'].get('rotation_deg')
        self.dwell_time = self.meta['metadata']['scan']['scan_device_parameters'].get('pixel_time_us') * 1E-6
        self.exposure = self.meta['metadata']['hardware_source'].get('exposure')
        
        self.aberrations = self.read_abberations()

    def read_signal_type(self):
        signal_type = self.meta['properties'].get('signal_type') or \
            self.meta['metadata']['hardware_source'].get('signal_type')
        if signal_type == 'eels': signal_type = 'EELS'
        return signal_type

    def read_axes_calibrations(self):
        axes = [ax.copy() for ax in self.meta['spatial_calibrations']]
        return axes
    
    def infer_axes_names(self):
        if self.is_series: self.axes[0]['name'] = 'time'
        self.axes_rspace_dims = [i for i, ax in enumerate(self.axes) if ax['units'] in ('um','nm','A','pm')]
        for i,n in zip(self.axes_rspace_dims[::-1], 'xyz'): self.axes[i]['name'] = n #may need to reverse
        if self.signal_type == 'EELS':
            if self.meta['datum_dimension_count'] == 2:
                self.axes_qspace_dims = [-2]
                self.axes[-2]['name'] = 'q'
                self.axes[-2]['units'] = 'px'
            self.axes_sspace_dims = [-1]
            self.axes[-1]['name'] = 'E'
        elif self.signal_type == 'diffraction':
            self.axes[-2]['name'] = 'qx'
            self.axes[-1]['name'] = 'qy'
            self.axes_qspace_dims = [-2, -1]
            for i in self.axes_qspace_dims:
                self.axes[i]['scale'] = 1
                self.axes[i]['offset'] = data.shape[i]/2
                self.axes[i]['units'] = 'px'

    def read_abberations(self):
        aber = {k:v for k,v in self.meta['metadata']['instrument']['ImageScanned'].items() if k[0]=='C' and k[1:3].isdigit()}
        mags = {}
        angs = {}
        for k,v in aber.items():
            ab = k[:3]
            if k[0]=='C' and k[-1]=='a':
                c = aber[ab+'.a']+1j*aber[ab+'.b']
                mags[ab] = np.abs(c)
                angs['phi'+ab[1:]] = np.angle(c)
            else:
                mags[ab] = v
        return {**mags, **angs}

def swift_meta_to_hs_dict(swift_metadata:object, signal_type:str=None) -> dict:
    """swift_meta_to_hs_dict _summary_

    Parameters
    ----------
    swift_metadata : object
        swift_json_reader class containing the swift metadata.
    signal_type : str, optional
        Type of hyperspy signal. Some signals require specific metadata trees.
        Common types are EELS, electron_diffraction. For a full list use `hs.print_known_signal_types()`
        by default None

    Returns
    -------
    dict
        Dictionary in the format of hyperspy metadata.
    """
    if signal_type is None: signal_type = swift_metadata.signal_type

    meta_dict = {}
    meta_dict['General'] = dict(title=swift_metadata.title)
    meta_dict['Signal'] = dict(signal_type=swift_metadata.signal_type)
    meta_dict['Acquisition_instrument'] = {'TEM':{
        'beam_energy': swift_metadata.beam_energy,
        'scan_rotation': swift_metadata.scan_rotation,
        'Detector':{
            'EELS':{
                'dwell_time': swift_metadata.dwell_time,
                'exposure':  swift_metadata.exposure
            }
        },
        'Aberrations': swift_metadata.aberrations
    }}
    return meta_dict

def load_swift_to_hs(file:str, signal_type:str=None, lazy:bool=False, **kwargs) -> object:
    """load_swift_to_hs _summary_

    Parameters
    ----------
    file : str
        Filename for the npy json pair to be read into hyperspy.
    signal_type : str, optional
        Type of hyperspy signal. Some signals require specific metadata trees.
        Common types are EELS, electron_diffraction. For a full list use `hs.print_known_signal_types()`
        by default None
    lazy : bool, optional
        Keep the data on disk, by default False

    Returns
    -------
    object
        _description_
    """
    from hyperspy.signals import BaseSignal, Signal1D, Signal2D

    meta = swift_json_reader(file, signal_type=signal_type)

    mmap = 'c' if kwargs.get('lazy') else None
    data = np.load(file+'.npy', mmap_mode=mmap)

    if meta.sig_dim == 1:
        Signal = Signal1D
    elif meta.sig_dim == 2:
        Signal = Signal2D
    else:
        Signal = BaseSignal

    sig = Signal(data, axes=meta.axes,
                 metadata=swift_meta_to_hs_dict(meta),
                 original_metadata=meta.meta)
    return sig

def load_swift_to_hdf5(file:str, signal_type:str=None, lazy:bool=False, **kwargs) -> object:
    meta = swift_json_reader(file, signal_type=signal_type)
    
    mmap = 'c' if kwargs.get('lazy') else None
    data = np.load(file+'.npy', mmap_mode=mmap)

    f = h5py.File(file+".hdf5", "w")
    ds = f.create_group("Experiments").create_group(meta.title)
    ds.create_dataset("data", data=data)

    grp_cal = ds.create_group("calibration")
    for i, ax in enumerate(meta.axes):
        dim = grp_cal.create_dataset(f'dim{i}', data=np.arange(ax['size'])*ax['scale']+ax['offset'])
        for k,v in ax.items():
            dim.attrs[k] = v

    grp_meta = ds.create_group("metadata")
    grp_ins = grp_meta.create_group("instrument")
    grp_aber = grp_ins.create_group('aberrations')
    for k,v in meta.aberrations.items(): grp_aber.create_dataset(k, data=v*1E9)

    grp_ins.create_dataset('energy', data=meta.beam_energy)
    grp_ins.create_dataset('scan_rotation', data=meta.scan_rotation)
    grp_ins.create_dataset('dwell_time', data=meta.dwell_time)
    grp_ins.create_dataset('exposure_time', data=meta.exposure)

    return f


def convert_swift_to_py4DSTEM(file:str, lazy:bool=False, **kwargs) -> object:
    meta = swift_json_reader(file, signal_type='diffraction')
    
    mmap = 'c' if kwargs.get('lazy') else None
    data = np.load(file+'.npy', mmap_mode=mmap)
    if meta.is_series:
        raise('Reading of series are not currently implimented. The data must be a four-dimenstional 4D-STEM scan.')

    f = h5py.File(file+".hdf5", "w")
    ds = f.create_group("Experiments").create_group(meta.title)
    dc = ds.create_dataset("datacube", data=data)
    
    
    axis_names = ['Rx', 'Ry', 'Qx', 'Qy']
    for i,ax in enumerate(meta.axes):
        dim = dc.create_dataset(f"dim{i}", data=np.arange(ax['size']))
        dim.attrs['name'] = axis_names[i]
        dim.attrs['units'] = ax['units']
    cal = ds.create_group("calibration")
    cal.create_dataset('R_pixel_size', data=meta.axes[0]['scale'])
    cal.create_dataset('R_pixel_units', data=meta.axes[0]['units'])
    cal.create_dataset('Q_pixel_size', data=meta.axes[-1]['scale'])
    cal.create_dataset('Q_pixel_units', data=meta.axes[-1]['units'])
    cal.create_dataset('qx0_mean', data=-meta.axes[-2]['offset'])
    cal.create_dataset('R_pixel_units', data=-meta.axes[-1]['offset'])

    f.close()