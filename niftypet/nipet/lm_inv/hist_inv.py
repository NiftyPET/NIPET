"""hist.py: processing of PET list-mode data: histogramming and randoms estimation."""
import logging
import os
from collections.abc import Collection
from numbers import Integral

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi

from niftypet import nimpa

from .. import invaux

# CUDA extension module
from . import lmproc_inv

log = logging.getLogger(__name__)

# ===============================================================================
# HISTOGRAM THE LIST-MODE DATA
# -------------------------------------------------------------------------------


def hist_inv(datain, scanner_params, t0=0, t1=0, outpath='', frms=None, use_stored=False,
            store=False, cmass_sig=5):
    '''
    Process the list-mode data and return histogram, head curves,
    and centre of mass for motion detection.

    optional bootstrap of LM events can be selected by setting the following:
    Cnt['BTP'] = 0: no bootstrapping [default];
    Cnt['BTP'] = 1: non-parametric bootstrapping;
    Cnt['BTP'] = 2: parametric bootstrapping (using Poisson distribution with mean = 1)

    '''

    # > extract constants, transaxial and axial LUTs
    Cnt = scanner_params['Cnt']
    txLUT = scanner_params['txLUT']
    axLUT = scanner_params['axLUT']

    if Cnt['SPN'] == 1: nsinos = Cnt['NSN1']
    elif Cnt['SPN'] == 0: nsinos = Cnt['NSEG0']

    log.debug('histogramming with span {}.'.format(Cnt['SPN']))

    if (use_stored is True and 'sinos' in datain
            and os.path.basename(datain['sinos']) == f"sinos_s{Cnt['SPN']}_frm-{t0}-{t1}.npz"):
        hstout = dict(np.load(datain['sinos'], allow_pickle=True))
        nitag = len(hstout['phc'])
        log.debug('acquisition duration by integrating time tags is {} sec.'.format(nitag))

    elif os.path.isfile(datain['lm_bf']):
        
        # > gather info about the LM time tags
        nele, ttags, tpos = lmproc_inv.lminfo(datain['lm_bf'], Cnt)

        # > multiply time tags by 200 microsecond - the reported time increments in list data
        nitag = int((ttags[1] - ttags[0] + 999) / 1000)
        log.debug('acquisition duration by integrating time tags is {} sec.'.format(nitag))

        # adjust frame time if outside the limit
        if t1 > nitag: t1 = nitag
        # check if the time point is allowed
        if t0 >= nitag:
            raise ValueError(
                'e> the time frame definition is not allowed! (outside acquisition time)')

        # ---------------------------------------
        # preallocate all the output arrays
        VTIME = 2
        MXNITAG = 5400 # limit to 1hr and 30mins
        if (nitag > MXNITAG):
            tn = int(MXNITAG / (1 << VTIME))
        else:
            tn = int((nitag + (1 << VTIME) - 1) / (1 << VTIME))

        pvs = np.zeros((tn, Cnt['NSEG0'], Cnt['NSBINS']), dtype=np.uint32)
        phc = np.zeros((nitag), dtype=np.uint32)
        dhc = np.zeros((nitag), dtype=np.uint32)
        mss = np.zeros((nitag, Cnt['NSEG0']), dtype=np.uint32)

        bck = np.zeros((2, nitag, Cnt['NBCKT']), dtype=np.uint32)
        fan = np.zeros((Cnt['NRNG'], Cnt['NCRS']), dtype=np.uint32)

        # > prompt and delayed sinograms
        psino = np.zeros((nsinos, Cnt['NSBINS'], Cnt['NSANGLES']), dtype=np.uint16)
        dsino = np.zeros((nsinos, Cnt['NSBINS'], Cnt['NSANGLES']), dtype=np.uint16)

        # > single slice rebinned prompts
        ssr = np.zeros((Cnt['NSEG0'], Cnt['NSBINS'], Cnt['NSANGLES']), dtype=np.uint32)

        hstout = {
            'phc': phc, 'dhc': dhc, 'mss': mss, 'pvs': pvs, 'bck': bck, 'fan': fan, 'psn': psino,
            'dsn': dsino, 'ssr': ssr}
        # ---------------------------------------

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # do the histogramming and processing
        lmproc_inv.hist(hstout, datain['lm_bf'], t0, t1, txLUT, axLUT, Cnt)
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        if store:
            if outpath == '':
                fsino = os.path.dirname(datain['lm_bf'])
            else:
                fsino = os.path.join(outpath, 'sino')
                nimpa.create_dir(fsino)
            # complete the path with the file name
            fsino = os.path.join(fsino, f"sinos_s{Cnt['SPN']}_frm-{t0}-{t1}.npz")
            # store to the above path
            np.savez(fsino, **hstout)

    else:
        log.error('input list-mode data is not defined.')
        return


    activ = hstout['mss']>0
    ssrb_idxs = np.arange(Cnt['NSEG0'])
    cmass = np.zeros(nitag, dtype=np.float32)
    for i in range(nitag):
        cmass[i] = np.sum((hstout['mss'][i,activ[i]] * ssrb_idxs[activ[i]]) / np.sum(hstout['mss'][i,:]))
    cmass = Cnt['AXR']*Cnt['NRNG']/Cnt['NSEG0'] * ndi.filters.gaussian_filter(cmass, cmass_sig, mode='mirror')

    # account for the fact that when t0==t1 that means that full dataset is processed
    if t0 == t1: t1 = t0 + nitag

    return {
        't0': t0,
        't1': t1,
        'dur': t1 - t0,           # duration
        'phc': hstout['phc'],     # prompts head curve
        'dhc': hstout['dhc'],     # delayeds head curve
        'cmass': cmass,           # centre of mass of the radiodistribution in axial direction
        #'pvs_sgtl': pvs_sgtl,     # sagittal projection views in short intervals
        #'pvs_crnl': pvs_crnl,     # coronal projection views in short intervals
        #'fansums': hstout['fan'], # fan sums of delayeds for variance reduction of randoms
        #'sngl_rate': single_rate, # bucket singles over time
        #'tsngl': t,               # time points of singles measurements in list-mode data
        #'buckets': buckets,       # average bucket singles
        'psino': hstout['psn'],    # prompt sinogram
        'dsino': hstout['dsn'],    # delayeds sinogram
        'pssr': hstout['ssr'],     # single-slice rebinned sinogram of prompts
        'mss':hstout['mss']
    }

    return hstout

