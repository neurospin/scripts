# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 16:02:42 2015

@author: mathieu

Utility to generate the data.

"""

import numpy as np

SEED = 42

SHAPE = (100, 100, 1)

N_SAMPLES = 500

# Object model (modulated by SNR): STDEV[0] is for l12, STDEV[1] is for l3,
# STDEV[2] is for l45
STDEV = np.asarray([1, 0.5, 0.8])


def create_model(snr):
    """
    Create a data model for a given value of SNR.
    """
    model = dict(
        # All points has an independant latent
        l1=0., l2=0., l3=STDEV[1] * snr, l4=0., l5=0.,
        # No shared variance
        l12=STDEV[0] * snr, l45=STDEV[2] * snr, l12345=0.,
        # Five dots contribute equally
        b1=1., b2=1., b3=1., b4=1., b5=1.)
    return model
