# from hoglundTools.calculators import *

# from hoglundTools import plot
# from hoglundTools import signal

import importlib


__all__ = [
    'plot',
    'signal',
    'calculators']

def __dir__():
    return sorted(__all__)

# mapping following the pattern: from value import key
_import_mapping = {}

def __getattr__(name):
    if name in __all__:
        if name in _import_mapping.keys():
            import_path = 'hoglundTools' + _import_mapping.get(name)
            return getattr(importlib.import_module(import_path), name)
        else:
            return importlib.import_module("." + name, 'hoglundTools')
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")