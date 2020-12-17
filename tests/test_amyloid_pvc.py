import sys
import logging
from os import path
from textwrap import dedent

import numpy as np

from niftypet import nimpa
from niftypet import nipet

mMRpars = nipet.get_mmrparams()
mMRpars['Cnt']['VERBOSE'] = True
mMRpars['Cnt']['LOG'] = logging.INFO

pvcroi = []  # from amyroi_def import pvcroi
# expected %error for static (SUVr) and PVC reconstructions
emape_basic = 0.1
emape_algnd = {
    "pet": 3.0,
    "pos": 0.1,
    "trm": 3.0,
    "pvc": 3.0,
    "hmu": 0.01,
    "omu": 3.0,
}


def test_histogramming(folder_in, folder_ref):
    datain = nipet.classify_input(folder_in, mMRpars)
    refpaths, testext = nipet.resources.get_refimg(folder_ref)
    hst = nipet.mmrhist(datain, mMRpars)

    # prompt counts: head curve & sinogram
    assert np.sum(hst['phc']) == np.sum(hst['psino'])

    # delayed counts: head curve & sinogram
    assert np.sum(hst['dhc']) == np.sum(hst['dsino'])

    # prompt counts: head curve & reference
    assert np.sum(hst['phc']) == refpaths['histo']['p']

    # delayed counts: head curve & reference
    assert np.sum(hst['dhc']) == refpaths['histo']['d']
