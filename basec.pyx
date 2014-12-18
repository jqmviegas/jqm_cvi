# -*- coding: utf-8 -*-
#cython: language_level=3, boundscheck=False
__author__ = "Joaquim Viegas"

""" JQM_CV - Cython implementations of Dunn and Davis Bouldin clustering validity indices

dunn(k_list):
    Slow implementation of Dunn index that depends on numpy
    -- basec.pyx Cython implementation is much faster but flower than dunn_fast()
dunn_fast(points, labels):
    Fast implementation of Dunn index that depends on numpy and sklearn.pairwise
    -- No Cython implementation
davisbouldin(k_list, k_centers):
    Implementation of Davis Boulding index that depends on numpy
    -- basec.pyx Cython implementation is much faster
"""

import numpy as np
cimport numpy as np
from cython.view cimport array as cvarray
from sklearn.metrics.pairwise import euclidean_distances

cdef double d_euc(double[:] arr1, double[:] arr2):
    """ Euclidean distance
        ...
    """
    cdef double total = 0
    
    for i in range(arr1.shape[0]):
        total += (arr1[i] - arr2[i])*(arr1[i] - arr2[i])
        
    return total**.5

cdef double dunn_delta(double [:, :] ck, double [:, :] cl):
    cdef: 
        double [:, :] values = np.zeros([ck.shape[0], cl.shape[0]], dtype=np.float64)
        Py_ssize_t i, j
        
    for i in range(ck.shape[0]):
        for j in range(cl.shape[0]):
            values[i, j] = d_euc(ck[i], cl[j])
            
    return np.min(values)

cdef double dunn_big_delta(double [:, :] ci):
    cdef: 
        double [:,:] values = np.zeros([ci.shape[0], ci.shape[0]], dtype=np.float64)
        Py_ssize_t i, j
        
    for i in range(ci.shape[0]):
        for j in range(ci.shape[0]):
            values[i, j] = d_euc(ci[i], ci[j])
    
    return np.max(values)

def dunn(k_list):
    """ Dunn index [CVI]
    
    Parameters
    ----------
    k_list : list of np.arrays
        A list containing a numpy array for each cluster |c| = number of clusters
        c[K] is np.array([N, p]) (N : number of samples in cluster K, p : sample dimension)
    """
    cdef: 
        Py_ssize_t len_k_list = len(k_list)
        double [:,:] deltas = np.ones([len_k_list, len_k_list], dtype=np.float64)*100000
        double [:,:] big_deltas = np.zeros([len_k_list, 1], dtype=np.float64)
        Py_ssize_t k, l
        
    for k in range(0, len_k_list):
        for l in range(0, k):
            deltas[k, l] = dunn_delta(k_list[k], k_list[l])
        for l in range(k+1, len_k_list):
            deltas[k, l] = dunn_delta(k_list[k], k_list[l])
        
        big_deltas[k] = dunn_big_delta(k_list[k])
    res = np.min(deltas)/np.max(big_deltas)*1
    return res
    
cdef double big_s(double [:, :] x, double [:] center):
    cdef:
        Py_ssize_t len_x = x.shape[0]
        double total = 0
        
    for i in range(len_x):
        total += d_euc(x[i], center)    
    
    return total/len_x

def davisbouldin(k_list, k_centers):
    """ Davis Bouldin Index
    k_list : list of np.arrays
        A list containing a numpy array for each cluster |c| = number of clusters
        c[K] is np.array([N, p]) (N : number of samples in cluster K, p : sample dimension)
    k_centers : np.array
        The array of the cluster centers (prototypes) of type np.array([K, p])
    Parameters
    ----------
    """
    cdef: 
        Py_ssize_t len_k_list = len(k_list)
        Py_ssize_t k, j
        double [:] big_ss = np.zeros([len_k_list], dtype=np.float64)
        double [:, :] d_eucs = np.zeros([len_k_list, len_k_list], dtype=np.float64)
        double db = 0
        double [:] values = np.zeros([len_k_list-1], dtype=np.float64)

    for k in range(len_k_list):
        big_ss[k] = big_s(k_list[k], k_centers[k])

    for k in range(len_k_list):
        for l in range(0, len_k_list):
            d_eucs[k, l] = d_euc(k_centers[k], k_centers[l])

    for k in range(len_k_list):
        for l in range(0, k):
            values[l] = (big_ss[k] + big_ss[l])/d_eucs[k, l]
        for l in range(k+1, len_k_list):
            values[l-1] = (big_ss[k] + big_ss[l])/d_eucs[k, l]

        db += np.max(values)
    res = db/len_k_list
    return res
    