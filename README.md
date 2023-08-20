jqmcvi
=======

Small module with Cluster Validity Indices (CVI)
------------------------------------------------

Dunn and Davius Bouldin indices are implemented. It follows the equations presented in theory.pdf.

> base.py : Python + NumPy
>
> basec.pyx : Python + NumPy optimized with Cython
>
> basec.pyx tested in Windows 8.1 x64, Python 3.4 and compiled with VS2010 (python setup.py build_ext -i)

Functions:
----------

**dunn(k_list)**:
> Slow implementation of Dunn index that depends on numpy
>
> -- basec.pyx Cython implementation is much faster but slower than dunn_fast()

```python
	""" Dunn index [CVI]
    
    Parameters
    ----------
    k_list : list of np.arrays
        A list containing a numpy array for each cluster |c| = number of clusters
        c[K] is np.array([N, p]) (N : number of samples in cluster K, p : sample dimension)
    """
```

**dunn_fast(points, labels)**:
> Fast implementation of Dunn index that depends on numpy and sklearn.pairwise
>
> -- No Cython implementation

```python
	""" Dunn index - FAST (using sklearn pairwise euclidean_distance function)
    
    Parameters
    ----------
    points : np.array
        np.array([N, p]) of all points
    labels: np.array
        np.array([N]) labels of all points
    """
```

**davisbouldin(k_list, k_centers)**:
> Implementation of Davis Boulding index that depends on numpy
> 
> -- basec.pyx Cython implementation is much faster

```python
	""" Davis Bouldin Index
	
	Parameters
    ----------
    k_list : list of np.arrays
        A list containing a numpy array for each cluster |c| = number of clusters
        c[K] is np.array([N, p]) (N : number of samples in cluster K, p : sample dimension)
    k_centers : np.array
        The array of the cluster centers (prototypes) of type np.array([K, p])
    """
```

## Installation 

```bash
pip install git+https://github.com/jqmviegas/jqm_cvi.git
```

If you face this error
```bash
 error: subprocess-exited-with-error
  
  × python setup.py egg_info did not run successfully.
  │ exit code: 1
  ╰─> [6 lines of output]
      Traceback (most recent call last):
        File "<string>", line 2, in <module>
        File "<pip-setuptools-caller>", line 34, in <module>
        File "/tmp/pip-req-build-utf7v9ji/setup.py", line 9, in <module>
          from Cython.Distutils import build_ext
      ModuleNotFoundError: No module named 'Cython'
      [end of output]
```

you can install Cython first, then try again to install this module
