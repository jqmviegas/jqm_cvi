# -*- coding: utf-8 -*-
__author__ = "Joaquim Viegas"
"""
jqm CVI
"""

import numpy as np

def delta(ck, cl):
    values = np.ones([len(ck), len(cl)])*10000
    
    for i in range(0, len(ck)):
        for j in range(0, len(cl)):
            values[i, j] = np.linalg.norm(ck[i]-cl[j])
            
    return np.min(values)
    
def big_delta(ci):
    values = np.zeros([len(ci), len(ci)])
    
    for i in range(0, len(ci)):
        for j in range(0, len(ci)):
            values[i, j] = np.linalg.norm(ci[i]-ci[j])
            
    return np.max(values)
    
def dunn(c):
    deltas = np.ones([len(c), len(c)])*10000
    big_deltas = np.zeros([len(c), 1])
    l_range = list(range(0, len(c)))
    
    for k in l_range:
        for l in (l_range[0:k]+l_range[k+1:]):
            deltas[k, l] = delta(c[k], c[l])
        
        big_deltas[k] = big_delta(c[k])

    di = np.min(deltas)/np.max(big_deltas)
    return di