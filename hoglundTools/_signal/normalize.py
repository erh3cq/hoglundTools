# -*- coding: utf-8 -*-
"""
Created on Sun June 10 14:50:00 2018

@author: Eric Hoglund
"""
import numpy as np

def normalize_by_ZLP(s, threshold=3.):
	ZLP_I = s.isig[:threshold].max(-1)
	return s/ZLP_I

def normalize_by_Total(s):
	s.change_dtype('int64')
	return s/s.sum(-1)
