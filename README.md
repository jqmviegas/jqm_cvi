jqm_cvi
=======

Small module with Cluster Validity Indexes (CVI)
------------------------------------------------

Dunn and Davius Bouldin indices are implemented. It follows the equations presented in theory.pdf.

> base.py : Python + NumPy
>
> basec.pyx : Python + NumPy optimized with Cython
>
> basec.pyx tested in Windows 8.1 x64 and compiled with VS2010

Functions:
----------

dunn(k_list):
> Slow implementation of Dunn index that depends on numpy
>
> -- basec.pyx Cython implementation is much faster but flower than dunn_fast()

dunn_fast(points, labels):
> Fast implementation of Dunn index that depends on numpy and sklearn.pairwise
>
> -- No Cython implementation

davisbouldin(k_list, k_centers):
> Implementation of Davis Boulding index that depends on numpy
> 
> -- basec.pyx Cython implementation is much faster
