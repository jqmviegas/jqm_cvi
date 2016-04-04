#! D:\Anaconda3
# -*- coding: utf-8 -*-

__author__ = "Joaquim Viegas"

#==============================================================================
# Description
#==============================================================================

import jqmcvi.basec as jqmcvi
import jqmcvi.base as jqmcvin

import pandas as pd
import numpy as np
import pickle
from timeit import timeit
from sklearn.metrics import silhouette_score as Sil

if __name__ == "__main__":  
    ccs = pickle.load(open('ccs.pkl', 'rb'))
    ps = pickle.load(open('ps.pkl', 'rb'))
    lbls = pickle.load(open('lbls.pkl', 'rb'))
    
    cps = []
    for i in range(0, len(ccs)):
        cps.append([])
    
    i = 0
    for lbl in lbls:
        cps[lbl].append(ps[i])
        i +=1
        
#    cps[0] = cps[0][0:500]
#    cps[1] = cps[1][0:500]
    cps[0] = np.array(cps[0])
    cps[1] = np.array(cps[1])
    
    print(timeit("Sil(ps, lbls, metric='euclidean')", setup="from __main__ import Sil, ps, lbls", number=1))
    print(timeit("jqmcvi.dunn(cps)", setup="from __main__ import jqmcvi, cps", number=1))
    print(timeit("jqmcvin.dunn_fast(ps, lbls)", setup="from __main__ import jqmcvin, ps, lbls", number=1))
    print(timeit("jqmcvi.davisbouldin(cps, ccs)", setup="from __main__ import jqmcvi, cps, ccs", number=1))       
    print(timeit("jqmcvin.davisbouldin(cps, ccs)", setup="from __main__ import jqmcvin, cps, ccs", number=1))       

    print(Sil(ps, lbls, metric='euclidean'))
    print(jqmcvi.dunn(cps))
    print(jqmcvin.dunn_fast(ps, lbls))
    print(jqmcvi.davisbouldin(cps, ccs))
    print(jqmcvin.davisbouldin(cps, ccs))
    