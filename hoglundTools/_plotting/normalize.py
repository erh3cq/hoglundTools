# -*- coding: utf-8 -*-
"""
Created on Wed May 30 11:24:38 2018

@author: erhog
"""

def normalize_and_floor(signal):
    return (signal-signal.min())/(signal-signal.min()).max()