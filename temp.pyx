#cdef double big_s(double [:, :] x, double [:] center):
#    cdef:
#        Py_ssize_t len_x = x.shape[0]
#        double total = 0
#        
#    for i in range(len_x):
#        total += d_euc(x[i], center)
#
#    return total/len_x
#
#def davisbouldin(k_list, k_centers):
#    """ Davis Bouldin Index
#    
#    Parameters
#    ----------
#    """
#    cdef: 
#        Py_ssize_t len_k_list = len(k_list)
#        double [:] big_ss = np.zeros([len_k_list], dtype=np.float64)
#        double [:, :] d_eucs = np.ones([len_k_list, len_k_list], dtype=np.float64)*100000
#        double db = 0
#        double [:] values = np.zeros([len_k_list-1], dtype=np.float64)
#
#    for k in range(len_k_list):
#        big_ss[k] = big_s(k_list[k], k_centers[k])
#
#    for k in range(len_k_list):
#        for l in range(0, len_k_list):
#            d_eucs[k, l] = d_euc(k_centers[k], k_centers[l])
#
#    for k in range(len_k_list):
#        for l in range(0, k):
#            values[k] = (big_ss[k] + big_ss[l])/d_eucs[k, l]
#        for l in range(k+1, len_k_list):
#            values[k-1] = (big_ss[k] + big_ss[l])/d_eucs[k, l]
#
#        db += np.max(values)
#
#    return db/len_k_list