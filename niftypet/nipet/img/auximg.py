"""auxilary imaging functions for PET image reconstruction and analysis."""
import logging
import os
from collections.abc import Collection
from numbers import Integral

import numpy as np

from niftypet import nimpa

log = logging.getLogger(__name__)


def obtain_image(img, Cnt=None, imtype=''):
    '''
    Obtain the image (hardware or object mu-map) from file,
    numpy array, dictionary or empty list (assuming blank then).
    The image has to have the dimensions of the PET image used as in Cnt['SO_IM[X-Z]'].
    '''
    # > establishing what and if the image object has been provided
    # > all findings go to the output dictionary
    output = {}
    if isinstance(img, dict):
        if Cnt is not None and img['im'].shape != (Cnt['SO_IMZ'], Cnt['SO_IMY'], Cnt['SO_IMX']):
            log.error('provided ' + imtype +
                      ' via the dictionary has inconsistent dimensions compared to Cnt.')
            raise ValueError('Wrong dimensions of the mu-map')
        else:
            output['im'] = img['im']
            output['exists'] = True
            if 'fim' in img: output['fim'] = img['fim']
            if 'faff' in img: output['faff'] = img['faff']
            if 'fmuref' in img: output['fmuref'] = img['fmuref']
            if 'affine' in img: output['affine'] = img['affine']
            log.info('using ' + imtype + ' from dictionary')

    elif isinstance(img, (np.ndarray, np.generic)):
        if Cnt is not None and img.shape != (Cnt['SO_IMZ'], Cnt['SO_IMY'], Cnt['SO_IMX']):
            log.error('provided ' + imtype +
                      ' via the numpy array has inconsistent dimensions compared to Cnt.')
            raise ValueError('Wrong dimensions of the mu-map')
        else:
            output['im'] = img
            output['exists'] = True
            output['fim'] = ''
            log.info('using hardware mu-map from numpy array.')

    elif isinstance(img, str):
        if os.path.isfile(img):
            imdct = nimpa.getnii(img, output='all')
            output['im'] = imdct['im']
            output['affine'] = imdct['affine']
            if Cnt and output['im'].shape != (Cnt['SO_IMZ'], Cnt['SO_IMY'], Cnt['SO_IMX']):
                log.error('provided ' + imtype +
                          ' via file has inconsistent dimensions compared to Cnt.')
                raise ValueError('Wrong dimensions of the mu-map')
            else:
                output['exists'] = True
                output['fim'] = img
                log.info('using ' + imtype + ' from NIfTI file.')
        else:
            log.error('provided ' + imtype + ' path is invalid.')
            return None
    elif isinstance(img, list):
        output['im'] = np.zeros((Cnt['SO_IMZ'], Cnt['SO_IMY'], Cnt['SO_IMX']), dtype=np.float32)
        log.info(imtype + ' has not been provided -> using blank.')
        output['fim'] = ''
        output['exists'] = False
    # ------------------------------------------------------------------------
    return output


def dynamic_timings(flist, offset=0):
    '''
    Get start and end frame timings from a list of dynamic PET frame definitions.
    Arguments:
      flist: can be 1D list of time duration for each dynamic frame, e.g.:
            flist = [15, 15, 15, 15, 30, 30, 30, ...]
        or a 2D list of lists having 2 entries per definition:
        first for the number of repetitions and the other for the frame duration, e.g.:
            flist = [[4, 15], [8, 30], ...],
        meaning 4x15s, then 8x30s, etc.
      offset: adjusts for the start time (usually when prompts are strong enough over randoms)
    Returns (dict):
      'timings': [[0, 15], [15, 30], [30, 45], [45, 60], [60, 90], [90, 120], [120, 150], ...]
      'total': total time
      'frames': array([ 15,  15,  15,  15,  30,  30,  30,  30, ...])
    '''
    if not isinstance(flist, Collection) or isinstance(flist, str):
        raise TypeError('Wrong type of frame data input')
    if all(isinstance(t, Integral) for t in flist):
        tsum = offset
        # list of frame timings
        if offset > 0:
            t_frames = [[0, offset]]
        else:
            t_frames = []
        for i in range(len(flist)):
            # frame start time
            t0 = tsum
            tsum += flist[i]
            # frame end time
            t1 = tsum
            # append the timings to the list
            t_frames.append([t0, t1])
        frms = np.uint16(flist)
    elif all(isinstance(t, Collection) and len(t) == 2 for t in flist):
        if offset > 0:
            flist.insert(0, [1, offset])
            farray = np.asarray(flist, dtype=np.uint16)
        else:
            farray = np.array(flist)
        # number of dynamic frames
        nfrm = np.sum(farray[:, 0])
        # list of frame duration
        frms = np.zeros(nfrm, dtype=np.uint16)
        # frame iterator
        fi = 0
        # time sum of frames
        tsum = 0
        # list of frame timings
        t_frames = []
        for i in range(farray.shape[0]):
            for _ in range(farray[i, 0]):
                # frame start time
                t0 = tsum
                tsum += farray[i, 1]
                # frame end time
                t1 = tsum
                # append the timings to the list
                t_frames.append([t0, t1])
                frms[fi] = farray[i, 1]
                fi += 1
    else:
        raise TypeError('Unrecognised time frame definitions.')
    return {'total': tsum, 'frames': frms, 'timings': t_frames}
