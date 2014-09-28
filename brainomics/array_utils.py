# -*- coding: utf-8 -*-
"""
Created on Sun Sep 28 17:39:34 2014

@author: edouard.duchesnay@cea.fr
"""

import numpy as np

def arr_get_threshold_from_norm2_ratio(v, ratio=.99):
    """Get threshold to apply to a 1d array such
    ||v[np.abs(v) >= t]|| / ||v|| == ratio
    return the threshold.

    Example
    -------
    >>> import numpy as np
    >>> v = np.random.randn(1e6)
    >>> t = arr_get_threshold_from_norm2_ratio(v, ratio=.5)
    >>> v_t = v.copy()
    >>> v_t[np.abs(v) < t] = 0
    >>> ratio = np.sqrt(np.sum(v[np.abs(v) >= t] ** 2)) / np.sqrt(np.sum(v ** 2))
    >>> print np.allclose(ratio, 0.5)
    True
    """
    #shape = v.shape
    import numpy as np
    v = v.copy().ravel()
    v2 = (v ** 2)
    v2.sort()
    v2 = v2[::-1]
    v_n2 = np.sqrt(np.sum(v2))
    #(v_n2 * ratio) ** 2
    cumsum2 = np.cumsum(v2)  #np.sqrt(np.cumsum(v2))
    select = cumsum2 <= ((v_n2 * ratio) ** 2)
    thres = np.sqrt(v2[select][-1])
    return thres

def arr_threshold_from_norm2_ratio(v, ratio=.99):
    """Threshold input array such
    ||v[np.abs(v) >= t]|| / ||v|| == ratio
    return the thresholded vector and the threshold

    Example
    -------
    >>> import numpy as np
    >>> v = np.random.randn(1e6)
    >>> v_t, t = arr_threshold_from_norm2_ratio(v, ratio=.5)
    >>> ratio = np.sqrt(np.sum(v_t ** 2)) / np.sqrt(np.sum(v ** 2))
    >>> print np.allclose(ratio, 0.5)
    """
    t = arr_get_threshold_from_norm2_ratio(v, ratio=ratio)
    v_t = v.copy()
    v_t[np.abs(v) < t] = 0
    return v_t, t

