# -*- coding: utf-8 -*-

import json
import sys
import os
from glob import glob
from warnings import warn

import numpy as np
import dask.array as da

from typing import List
from numpy.typing import NDArray

import h5py
import json
#from tqdm import tqdm
import zipfile

def h5_tree(file:str, pre:str='') -> None:
    """Print an h5 file tree

    Parameters
    ----------
    file : str
        File path
    pre : str, optional
        Entry to each line, by default ''
    """
    items = len(file)
    for key, file in file.items():
        items -= 1
        if items == 0:
            # the last item
            if type(file) == h5py._hl.group.Group:
                print(pre + '└── ' + key)
                h5_tree(file, pre+'    ')
            else:
                print(pre + '└── ' + key)
        else:
            if type(file) == h5py._hl.group.Group:
                print(pre + '├── ' + key)
                h5_tree(file, pre+'│   ')
            else:
                print(pre + '├── ' + key)

def load_memmap_from_npz(path, name):
    '''
    Can't load npy files from npz files as memap.
    Temporary work around from https://github.com/numpy/numpy/issues/5976.
    '''
    zf = zipfile.ZipFile(path)
    info = zf.NameToInfo[name + '.npy']
    assert info.compress_type == 0
    offset = zf.open(name + '.npy')._orig_compress_start

    fp = open(path, 'rb')
    fp.seek(offset)
    version = np.lib.format.read_magic(fp)
    assert version in [(1,0), (2,0)]
    if version == (1,0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(fp)
    elif version == (2,0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(fp)
    data_offset = fp.tell() # file position will be left at beginning of data
    return np.memmap(path, dtype=dtype, shape=shape,
                     order='F' if fortran_order else 'C', mode='r',
                     offset=data_offset)

def parse_file_path(file_path:str):
    directory, name = os.path.split(file_path)
    name, extension = os.path.splitext(name)
    if extension=='':
        extension = os.path.splitext(glob(file_path+'.n*')[0])[-1]
    return directory, name, extension

def collect_swift_file(file_path:str):
    _, _, file_extension = parse_file_path(file_path)
    file_path = os.path.splitext(file_path)[0]

    if file_extension == '.npy':
        with open(file_path+'.json') as f: meta = json.load(f)
        data = np.load(file_path+'.npy', mmap_mode='r')
    elif file_extension == '.ndata1':
        file = np.load(file_path+'.ndata1', mmap_mode='r')
        meta = json.loads(file['metadata.json'].decode())
        data = load_memmap_from_npz(file_path+'.ndata1', 'data')
    elif file_extension == '.h5':
        opened=h5py.File(file_path+'.h5')
        data=np.asarray(opened['data'])
        meta=json.loads(opened['data'].attrs['properties'])
        #printNestedDict(meta)
        meta=maph5meta( meta )
    elif file_extension == '.ndata':
        opened=np.load(file_path+'.ndata')
        data=opened["data"]
        meta=json.loads(opened["metadata.json"].decode())
        #printNestedDict(meta)
        meta=maph5meta(meta)
    else:
        raise Exception(f'The Swift files could not be collected.\nA file extension should with `.npy` or `.ndata1` were not found or provided.\n{file_path}')
    return meta, data

def printNestedDict(dict1,path=""):
    if isinstance(dict1,dict):
        keys=dict1.keys()
    else:
        keys=np.arange(len(dict1))
    for k in keys:
        if isinstance(dict1[k],(dict,list)):
            printNestedDict(dict1[k],path=path+" > "+str(k))
        else:
            print(path+" > "+str(k)+" > ",dict1[k])
def maph5meta(meta):
    new=meta
    new['metadata']['hardware_source']['source']=meta['metadata']['hardware_source']['hardware_source_name']
    #new['metadata']['signal_type']=new['metadata']['hardware_source']['signal_type'].upper().strip()
    new['properties']={}
    new['spatial_calibrations']=meta['dimensional_calibrations']
    return new

class swift_json_reader:
    
    def __init__(self, file_path:str, signal_type:str=None, get_npy_shape:bool=True, verbose=False):
        #File handling
        self.file_path = file_path
        self.file_directory, self.file_name, self.file_extension = parse_file_path(file_path)
        self.meta, data = collect_swift_file(self.file_path)
        self.data_shape = data.shape
        self.detector = self.meta['metadata']['hardware_source'].get('source')

        #General metadata
        self.title = self.meta.get('title')

        #Axis handling
        self.is_series = True if self.meta.get('is_sequence') is not None else False
        self.is_scan = True if self.meta['metadata'].get('scan') is not None else False
        self.is_preprocessed = len(self.meta.get('metadata')) + len(self.meta.get('properties')) == 0
        self.nav_dim = self.meta['collection_dimension_count']
        self.sig_dim = self.meta['datum_dimension_count']
        self.signal_type = signal_type if signal_type is not None else self.read_signal_type()
        if verbose: print(f'Read signal type: {self.signal_type}')

        self.axes = self.read_axes_calibrations()
        if verbose: print(f'Read axes: {self.axes}')
        if get_npy_shape:
            for ax, d in zip(self.axes, self.data_shape): ax['size'] = d
        if self.is_series and self.axes[0]['units']=='': self.axes[0]['units'] = 'frame'
        
        self.infer_axes_names(verbose=verbose)
        if verbose: print(f'Built axes: {self.axes}')

        if not self.is_preprocessed:
            #Instrument metadata
            self.beam_energy = self.meta['metadata']['instrument'].get('high_tension')/1E3
            self.aberrations = self.read_abberations()

            #Scan metadata
            if self.is_scan:
                self.scan_rotation = self.meta['metadata']['scan'].get('rotation_deg')
                self.dwell_time = self.meta['metadata']['scan']['scan_device_parameters'].get('pixel_time_us') * 1E-6
            
            #Detector metadata
            self.exposure = self.meta['metadata']['hardware_source'].get('exposure')
            self.readout_area = self.meta['metadata']['hardware_source'].get('interpolated_area_tlbr')
            self.binning = self.meta['properties'].get('binning')
            self.flip_x = self.meta['properties'].get('is_flipped_horizontally') or \
                self.meta.get('camera_processing_parameters').get('flip_l_r') if self.meta.get('camera_processing_parameters') is not None else None
        

    def read_signal_type(self):
        #signal_type = self.meta['properties'].get('signal_type') or \
        #    self.meta['metadata']['hardware_source'].get('signal_type') if self.meta.get('hardware_source')is not None else None
        #if signal_type == 'eels': signal_type = 'EELS'
        #print("read_signal_type:",self.sig_dim)
        #if signal_type is None:
        if self.sig_dim == 1:
            #sig_unit = self.meta['spatial_calibrations'][-1]
            #if sig_unit[-1] == 'eV':
            signal_type = 'EELS'
        if self.sig_dim == 2:
            sig_unit = np.asanyarray([ax['units'] for ax in self.meta['spatial_calibrations'][-2:]])
            if np.all(sig_unit == 'nm'):
                signal_type = 'Image'
            elif np.all(['rad' in u for u in sig_unit]):
                signal_type = 'diffraction'
            elif sig_unit[-1] == 'eV':
                if np.diff(self.data_shape[-2:])==0:
                    signal_type = 'diffraction'
                else:
                    signal_type = '2D-EELS'
        return signal_type

    def read_axes_calibrations(self):
        axes = [ax.copy() for ax in self.meta['spatial_calibrations']]
        return axes
    
    def infer_axes_names(self, verbose=False):
        if self.is_series: self.axes[0]['name'] = 'time'
        self.axes_rspace_dims = [i for i, ax in enumerate(self.axes) if ax['units'] in ('um','nm','A','pm')]
        for i,n in zip(self.axes_rspace_dims[::-1], 'xyz'): self.axes[i]['name'] = n #may need to reverse
        if self.signal_type == 'EELS':                
            self.axes_sspace_dims = [-1]
            self.axes[-1]['name'] = 'E'
        elif self.signal_type == '2D-EELS':
            self.axes_qspace_dims = [-2]
            self.axes[-2]['name'] = 'q'
            self.axes[-2]['units'] = 'px'
            self.axes_sspace_dims = [-1]
            self.axes[-1]['name'] = 'E'
        elif self.signal_type == 'diffraction':
            self.axes[-2]['name'] = 'qy'
            self.axes[-1]['name'] = 'qx'
            self.axes_qspace_dims = [-2, -1]
            for i in self.axes_qspace_dims:
                if self.detector != 'Ronchigram':
                    self.axes[i]['scale'] = 1
                    self.axes[i]['units'] = 'px'
                self.axes[i]['offset'] = -self.data_shape[i]/2 * self.axes[i]['scale']

    def read_abberations(self):
        if self.meta['metadata']['instrument'].get('ImageScanned') is None:
            return None
        aber = {k:v for k,v in self.meta['metadata']['instrument']['ImageScanned'].items() if k[0]=='C' and k[1:3].isdigit()}
        aber_c = {}
        #mags = {}
        #angs = {}
        for k,v in aber.items():
            ab = k[:3]
            if k[0]=='C' and k[-1]!='b':
                if k[-1] == 'a':
                    aber_c[ab] = aber[ab+'.a']+1j*aber[ab+'.b']
                else:
                    aber_c[ab] = v
        return aber_c

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
    if swift_metadata.is_preprocessed: return meta_dict
    
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

def load_swift_to_hs(file_path:str, signal_type:str=None, lazy:bool=False, verbose=False, **kwargs) -> object:
    """load_swift_to_hs _summary_

    Parameters
    ----------
    file_path : str
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
    from exspy.signals import EELSSpectrum

    meta = swift_json_reader(file_path, signal_type=signal_type, verbose=verbose)
    _, data = collect_swift_file(file_path)
    if meta.flip_x and np.logical_or(meta.signal_type=='diffraction', signal_type=='diffraction'): data[...,::-1]

    #mmap = 'c' if kwargs.get('lazy') else None
    #data = np.load(file_path+'.npy', mmap_mode=mmap)
    if lazy:
        #data = da.from_array(np.memmap(file_path+'.npy', mode='r'))
        data = da.from_array(data)
    else:
        data = data.copy()

    if meta.signal_type == 'EELS' or signal_type == 'EELS':
        Signal=EELSSpectrum
    elif meta.sig_dim == 1:
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
        
    if meta.signal_type == 'diffraction' or signal_type == 'diffraction':
        sig.set_signal_type('electron_diffraction')
    elif meta.signal_type == 'EELS' or signal_type == 'EELS':
        sig.set_signal_type('EELS')
    
    return sig

def load_swift_to_hdf5(file_path:str, signal_type:str=None, lazy:bool=False, **kwargs) -> object:
    meta = swift_json_reader(file_path, signal_type=signal_type)
    
    mmap = 'c' if kwargs.get('lazy') else None
    data = np.load(file_path+'.npy', mmap_mode=mmap)

    f = h5py.File(file_path+".hdf5", "w")
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


#TODO: flip_x is bakwards.
def convert_swift_to_py4DSTEM(file_path:str, lazy:bool=False, verbose=False, **kwargs) -> object:
    warn('Convert_swift_to_py4DSTEM is depreciated. Use load_swift_to_py4DSTEM.', DeprecationWarning)
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

    meta = swift_json_reader(file_path, signal_type='diffraction')
    _, data = collect_swift_file(file_path)

    if not kwargs.get('lazy'): data = data.copy()
    if meta.is_series: raise('Reading of series are not currently implimented. The data must be a four-dimenstional 4D-STEM scan.')
    end = '...' if verbose else '\n'
    print(f'{meta.file_directory}/{meta.file_name}.hdf5', end='...')
    
    #Create the file and initiate front matter
    f = h5py.File(f"{meta.file_directory}/{meta.file_name}.hdf5", "w")
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

def load_swift_to_py4DSTEM(file_path:str, lazy:bool=False, verbose=False, 
                           crop_r:List[int]=None, skip_r:int=None,
                           **kwargs) -> object:
    """Read swift ndata1 or npy+json file into the py4DSTEM format.

    Parameters
    ----------
    file_path : str
        File path to the file to be read. If '.ndata' or '.npy' are provided then the files will be searched for explicitly. If the extension is not provided then the reader will try to find the appropriate '.n*' extension.
    lazy : bool, optional
        If the initial data should be imported in memmap prior to preproccessing, by default False.
        This can be useful is cropping or sparsification is intended.
    verbose : bool or str, optional
        Knowledge==Power!, by default False

    Returns
    -------
    object
        If the data is larger than two-dimensions a DataCube is returned.
        If the data is two-dimensional a DiffractionSLice is retruned.
    """
    from py4DSTEM.data import DiffractionSlice, RealSlice
    from py4DSTEM.datacube import DataCube

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

    # Read metadata and data
    meta = swift_json_reader(file_path, signal_type='diffraction', verbose=verbose)
    _, data = collect_swift_file(file_path)
    if crop_r is not None:
        assert len(crop_r)==meta.nav_dim
        for ax in crop_r: assert len(ax)==2
        data = data[crop_r[0][0]:crop_r[0][1], crop_r[1][0]:crop_r[1][1]]
    if skip_r is not None:
        data = data[::skip_r, ::skip_r]

    if not kwargs.get('lazy'): data = data.copy() #TODO: don't copy out of memmap before crop or sparse

    if meta.flip_x:
        if verbose: print('Detector flip_x flagged True. Reversing the qx axis.')
        data = data[...,::-1]
    else:
        if verbose: print('Detector flip_x flagged False. Not reversing the qx axis.')

    if kwargs.get('lazy'): data = data.copy() # Data up to now is memmap if lazy, which is not supported by py4DSTEM.
        
    # Determine the py4DSTEM class type
    if meta.is_series: raise('Reading of series are not currently implimented. The data must be a four-dimenstional 4D-STEM scan.')
    assert len(data.shape) in (2, 4)
    if len(data.shape) == 4:
        f = DataCube(data=data)
    else:
        if meta.axes[0]['units'] == 'nm':
            f = RealSlice(data=data)
        else:
            f = DiffractionSlice(data=data)

    # Set the calibrations.
    # Note that axes are reordered because Py4DSTEM does not like Qy offset being set beofre Qx.
    axes = {ax['name']: ax for ax in meta.axes}
    axes = {i:axes[i] for i, k in zip(['x','y','qx','qy'], axes.keys()) if i in list(axes.keys())}
    for k,v in axes.items():
        if verbose: print(f'Storing {k} axis', v)
        if v['units'] == 'px': v['units'] = 'pixels'
        if k == 'x':
            if skip_r is None:
                xscale = v['scale']
            else:
                xscale = v['scale'] * skip_r
            f.calibration.set_R_pixel_size(xscale)
            f.calibration.set_R_pixel_units(v['units'])
        elif k == 'y':
            if v['scale'] != axes['x']['scale']:
                print("Warning: py4DSTEM currently only handles uniform x,y sampling. Setting sampling with x calibration")
        elif k == 'qx':
            f.calibration.set_Q_pixel_size(v['scale'])
            f.calibration.set_Q_pixel_units(v['units'])
            f.calibration.set_qx0_mean(-v['offset'])
        elif k == 'qy':
            f.calibration.set_qy0_mean(-v['offset'])
            if v['scale'] != axes['qx']['scale']:
                print("Warning: py4DSTEM currently only handles uniform qx,qy sampling. Setting sampling with qx calibration")
        else:
            print(f'Axes {k} is not supported and will be ignored.')
    if meta.is_scan:
        f.calibration.set_QR_rotation_degrees(meta.scan_rotation)
    
    return f