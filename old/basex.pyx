# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np

cdef double d_e(np.ndarray[np.float64_t, ndim=1] c):
    cdef double s = 0
    cdef Py_ssize_t i
    for i in range(0, len(c)):
        s += c[i]*c[i]
        
    return s**(0.5)
    
cdef double delta(np.ndarray[np.float64_t, ndim=2] ck, np.ndarray[np.float64_t, ndim=2] cl):
    cdef np.ndarray[np.float64_t, ndim=2] values = np.ones([len(ck), len(cl)], dtype=np.float64)*10000000
    
    cdef Py_ssize_t i, j
    for i in range(0, len(ck)):
        for j in range(0, len(cl)):
            values[i, j] = d_e(ck[i]-cl[j])
    
    cdef double res = np.min(values)   
    return res

cdef double big_delta(np.ndarray[np.float64_t, ndim=2] ci):
    cdef np.ndarray[np.float64_t, ndim=2] values = np.zeros([len(ci), len(ci)], dtype=np.float64)
    
    cdef Py_ssize_t i, j
    for i in range(0, len(ci)):
        for j in range(0, len(ci)):
            values[i, j] = d_e(ci[i]-ci[j])
    
    cdef double res = np.max(values)
    return res

def dunn(c):
    cdef Py_ssize_t len_c = len(c)
    cdef np.ndarray[np.float64_t, ndim=2] deltas = np.ones([len_c, len_c], dtype=np.float64)*10000000
    cdef np.ndarray[np.float64_t, ndim=2] big_deltas = np.zeros([len_c, 1], dtype=np.float64)
    cdef Py_ssize_t k, l
        
    for k in range(0, len_c):
        for l in range(0, k):
            deltas[k, l] = delta(c[k], c[l])
        for l in range(k+1, len_c):
            deltas[k, l] = delta(c[k], c[l])
        
        big_deltas[k] = big_delta(c[k])
    
    cdef double di = np.min(deltas)/np.max(big_deltas)
    return di