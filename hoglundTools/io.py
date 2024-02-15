# -*- coding: utf-8 -*-

import json
import sys
import os
from glob import glob

import numpy as np
from numpy.typing import NDArray
import dask.array as da

import h5py
import json
from tqdm import tqdm

class swift_json_reader:
    
    def __init__(self, file_path:str, file_type=None, signal_type:str=None, get_npy_shape:bool=True, **kwargs):
        #File handling
        self.file_path = file_path
        self.file_directory, self.file_name = os.path.split(file_path)[-1]
        self.file_extension = os.path.split(self.file_name)[-1]
        if self.file_extension=='' and file_type is None:
            self.file_extension = os.path.split(glob(file+'.n*')[0])[-1]
        
        if self.file_extension == '.npy':
            with open(file_path+'.json') as f: self.meta = json.load(f)
            self.data_shape = np.load(file_path+'.npy', mmap_mode='r').shape
        elif self.file_extension == '.ndata1':
            file = np.load(file_path, mmap_mode='r')
            self.meta = json.loads(file['metadata.json'].decode())
            self.data_shape = file['data'].shape

        #General metadata
        self.title = self.meta.get('title')
        self.signal_type = signal_type if signal_type is not None else self.read_signal_type()

        #Axis handling
        self.is_series = True if self.meta.get('is_sequence') is not None else False
        self.is_scan = True if self.meta['metadata'].get('scan') is not None else False
        self.nav_dim = self.meta['collection_dimension_count']
        self.sig_dim = self.meta['datum_dimension_count']

        self.axes = self.read_axes_calibrations()
        if get_npy_shape:
            for ax, d in zip(self.axes, self.data_shape): ax['size'] = d
        if self.is_series and self.axes[0]['units']=='': self.axes[0]['units'] = 'frame'
        self.infer_axes_names()

        #Instrument metadata
        self.beam_energy = self.meta['metadata']['instrument'].get('high_tension')/1E3
        self.aberrations = self.read_abberations()

        #Scan metadata
        if self.is_scan:
            self.scan_rotation = self.meta['metadata']['scan'].get('rotation_deg')
            self.dwell_time = self.meta['metadata']['scan']['scan_device_parameters'].get('pixel_time_us') * 1E-6
        
        #Detector metadata
        self.exposure = self.meta['metadata']['hardware_source'].get('exposure')
        self.binning = self.meta['properties'].get('binning')
        self.flip_x = self.meta['properties'].get('is_flipped_horizontally')
        

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
                self.axes[i]['offset'] = -self.data_shape[i]/2
                self.axes[i]['units'] = 'px'

    def read_abberations(self):
        if self.meta['metadata']['instrument'].get('ImageScanned') is None:
            return None
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
    if signal_type is None:
        if swift_metadata.sig_dim == 1:
            signal_type = swift_metadata.signal_type = 'Signal1D'
        elif swift_metadata.sig_dim == 2:
            signal_type = swift_metadata.signal_type = 'Signal2D'
        else:
            print('HyperSpy signal type could not be infered. Please set it manually.')

    meta_dict = {}
    meta_dict['General'] = dict(title=swift_metadata.title)
    meta_dict['Signal'] = dict(signal_type=swift_metadata.signal_type)
    meta_dict['Acquisition_instrument'] = {'TEM':{
        'beam_energy': swift_metadata.beam_energy,
        'Aberrations': swift_metadata.aberrations,
        'Detector':{}
        }
    }

    if swift_metadata.is_scan:
        meta_dict['Acquisition_instrument']['TEM']['scan_rotation'] = swift_metadata.scan_rotation
        meta_dict['Acquisition_instrument']['TEM']['dwell_time'] = swift_metadata.dwell_time
        if swift_metadata.signal_type == 'EELS' or swift_metadata.signal_type == 'difraction':
            meta_dict['Acquisition_instrument']['TEM']['Detector']['EELS'] = {
                'exposure':  swift_metadata.exposure
            }

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

    #mmap = 'c' if kwargs.get('lazy') else None
    #data = np.load(file+'.npy', mmap_mode=mmap)
    if lazy:
        #data = da.from_array(np.memmap(file+'.npy', mode='r'))
        data = da.from_array(np.load(file+'.npy', mmap_mode='c'))
    else:
        data = np.load(file+'.npy')

    if meta.sig_dim == 1:
        Signal = Signal1D
    elif meta.sig_dim == 2:
        Signal = Signal2D
    else:
        Signal = BaseSignal

    sig = Signal(data, axes=meta.axes,
                 metadata=swift_meta_to_hs_dict(meta),
                 original_metadata=meta.meta)
    if lazy:
        sig = sig.as_lazy()
        
    if signal_type == 'diffraction':
        sig.set_signal_type('electron_diffraction')
    elif meta.signal_type == 'EELS' or signal_type == 'EELS':
        sig.set_signal_type('EELS')
    
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


# def convert_swift_to_py4DSTEM(file:str, lazy:bool=False, **kwargs) -> object:
#     meta = swift_json_reader(file, signal_type='diffraction')
    
#     mmap = 'c' if kwargs.get('lazy') else None
#     data = np.load(file+'.npy', mmap_mode=mmap)
#     if meta.is_series:
#         raise('Reading of series are not currently implimented. The data must be a four-dimenstional 4D-STEM scan.')
#     print(f'Creating {file}.hdf5', end='...')
#     f = h5py.File(file+".hdf5", "w")
    
#     ds = f.create_group("Experiments")
#     ds.attrs['emd_group_type'] = "root"
#     ds.attrs['version_major'] = "0"
#     ds.attrs['version_minor'] = "13"

#     ds = ds.create_group("datacube")
#     ds.attrs['emd_group_type'] = 1
#     if len(meta.axes) == 2:
#         ds.attrs['py4dstem_class'] = "DiffractionSlice"
#     else:
#         ds.attrs['py4dstem_class'] = "DataCube"

#     dc = ds.create_dataset("data", data=data)
#     dc.attrs['units'] = "pixel intensity"
    
#     axis_names = {'x':'Rx', 'y':'Ry', 'qx':'Qx', 'qy':'Qy'}
#     for i,ax in enumerate(meta.axes):
#         dim = ds.create_dataset(f"dim{i}", data=np.arange(ax['size']))
#         dim.attrs['name'] = axis_names[ax['name']]
#         dim.attrs['units'] = ax['units']
    
#     cal = ds.create_group("calibration")
#     cal.attrs['emd_group_type'] = 0
#     cal.attrs['py4dstem_class'] = "Calibration"
#     if meta.axes[0]['name'] in 'xyz':
#         cal.create_dataset('R_pixel_size', data=meta.axes[0]['scale'])
#         cal.create_dataset('R_pixel_units', data=meta.axes[0]['units'])
#     cal.create_dataset('Q_pixel_size', data=meta.axes[-1]['scale'])
#     cal.create_dataset('Q_pixel_units', data=meta.axes[-1]['units'])
#     cal.create_dataset('qx0_mean', data=-meta.axes[-2]['offset'])
#     cal.create_dataset('qy0_mean', data=-meta.axes[-1]['offset'])
#     for v in cal.values(): 
#         if h5py.check_string_dtype(v.dtype):
#             #v.attrs["type"] = 'string'.encode("latin-1")
#             v.attrs.create("type", 'string'.encode("latin-1"))
#         else:
#             #v.attrs["type"] = "number".encode("latin-1")
#             v.attrs.create("type", 'number'.encode("latin-1"))
    
#     print('Created.')
#     f.close()

def convert_swift_to_py4DSTEM(file:str, lazy:bool=False, verbose=False, **kwargs) -> object:
    def add_dataset_wAttrs(add_to, name:str, data, attributes:dict):
        set = add_to.create_dataset(name, data=data)
        for k,v in attributes.items():
            set.attrs.create(k, v)
        return set
    def add_group_wAttrs(add_to, name:str, attributes:dict):
        grp = add_to.create_group(name)
        for k,v in attributes.items():
            grp.attrs.create(k, v)
        return grp

    meta = swift_json_reader(file, signal_type='diffraction')
    
    mmap = 'c' if kwargs.get('lazy') else None
    data = np.load(file+'.npy', mmap_mode=mmap)
    if meta.is_series: raise('Reading of series are not currently implimented. The data must be a four-dimenstional 4D-STEM scan.')
    end = '...' if verbose else '\n'
    print(f'Creating {file}.hdf5', end='...')
    
    f = h5py.File(file+".hdf5", "w")
    f.attrs['authoring_program'] = 'hoglundTools'
    f.attrs['authoring_user'] = ""
    f.attrs['emd_group_type'] = "file"
    f.attrs['version_major'] = 1
    f.attrs['version_minor'] = 0
    
    root = add_group_wAttrs(f, 'Experiments',
                           {'emd_group_type': "root", 'python_class': "Root"})
    ds = add_group_wAttrs(root, "aquisition", {'emd_group_type': "array"})
    if len(meta.axes) == 2:
        ds.attrs['python_class'] = "DiffractionSlice"
    else:
        ds.attrs['python_class'] = "DataCube"

    if meta.flip_x:
        if verbose: print('Detector flip_x flagged True. Reversing the qx axis.')
    else:
        if verbose: print('Detector flip_x flagged False. Not reversing the qx axis.')
        data = data[...,::-1]
        #add_dataset_wAttrs(cal, 'QR_flip', -meta.axes[-1]['offset'], {"type": 'bool'})
    add_dataset_wAttrs(ds, "data", data,
                       {'units': "pixel intensity"})
    
    axis_names = {'x':'Rx', 'y':'Ry', 'qx':'Qx', 'qy':'Qy'}
    for i,ax in enumerate(meta.axes):
        add_dataset_wAttrs(ds, f"dim{i}", np.arange(ax['size']),
                            {'name':axis_names[ax['name']], 'units':ax['units']})
    
    met = add_group_wAttrs(root, "metadatabundle", {'emd_group_type': "metadatabundle"})

    cal = add_group_wAttrs(met, "calibration",
                            {'emd_group_type': "metadata", 'python_class': "Calibration"})

    #add_dataset_wAttrs(cal, '_root_treepath', b'', {"type":'string'})
    grp = add_group_wAttrs(cal, '_target_paths', {"type":'list_of_strings', 'length':1})
    grp.create_dataset('0', data=b'/aquisition')

    if meta.axes[0]['name'] in 'xyz':
        add_dataset_wAttrs(cal, 'R_pixel_size', meta.axes[0]['scale'], {"type":'number'})
        add_dataset_wAttrs(cal, 'R_pixel_units', meta.axes[0]['units'], {"type": 'string'})
    add_dataset_wAttrs(cal, 'Q_pixel_size', meta.axes[-1]['scale'], {"type": 'number'})
    add_dataset_wAttrs(cal, 'Q_pixel_units', meta.axes[-1]['units'], {"type": 'string'})
    add_dataset_wAttrs(cal, 'qx0_mean', -meta.axes[-2]['offset'], {"type": 'number'})
    add_dataset_wAttrs(cal, 'qy0_mean', -meta.axes[-1]['offset'], {"type": 'number'})
    if meta.is_scan:
        add_dataset_wAttrs(cal, 'QR_rotation', np.deg2rad(meta.scan_rotation), {"type": 'number'})
        add_dataset_wAttrs(cal, 'QR_rotation_degrees', meta.scan_rotation, {"type": 'number'})
    
    
    print('Created.')
    f.close()