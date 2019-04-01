"""auxilary imaging functions for PET image reconstruction and analysis."""

__author__      = "Pawel Markiewicz"
__copyright__   = "Copyright 2018"


import numpy as np
import os
from niftypet import nimpa

#==============================================================================

def obtain_image(img, Cnt=[], imtype='', verbose=False):
    ''' 
    Obtain the image (hardware or object mu-map) from file,
    numpy array, dictionary or empty list (assuming blank then).
    The image has to have the dimensions of the PET image used as in Cnt['SO_IM[X-Z]'].
    '''

    if Cnt: verbose = Cnt['VERBOSE']
    # establishing what and if the image object has been provided
    # all findings go to the output dictionary
    output = {}
    if isinstance(img, dict):
        if Cnt and img['im'].shape!=(Cnt['SO_IMZ'], Cnt['SO_IMY'], Cnt['SO_IMX']):
            print 'e> provided '+imtype+' via the dictionary has inconsistent dimensions compared to Cnt.'
            raise ValueError('Wrong dimensions of the mu-map')
        else:
            output['im'] = img['im']
            output['exists'] = True
            output['fim'] = img['fim']
            if 'faff' in img: output['faff'] = img['faff']
            if 'fmuref' in img: output['fmuref'] = img['fmuref']
            if 'affine' in img: output['affine'] = img['affine']
            if verbose: print 'i> using '+imtype+' from dictionary.'

    elif isinstance(img, (np.ndarray, np.generic) ):
        if Cnt and img.shape!=(Cnt['SO_IMZ'], Cnt['SO_IMY'], Cnt['SO_IMX']):
            print 'e> provided '+imtype+' via the numpy array has inconsistent dimensions compared to Cnt.'
            raise ValueError('Wrong dimensions of the mu-map')
        else:
            output['im'] = img
            output['exists'] = True
            output['fim'] = ''
            if verbose: print 'i> using hardware mu-map from numpy array.'

    elif isinstance(img, basestring):
        if os.path.isfile(img):
            imdct = nimpa.getnii(img, output='all')
            output['im'] = imdct['im']
            output['affine'] = imdct['affine']
            if Cnt and output['im'].shape!=(Cnt['SO_IMZ'], Cnt['SO_IMY'], Cnt['SO_IMX']):
                print 'e> provided '+imtype+' via file has inconsistent dimensions compared to Cnt.'
                raise ValueError('Wrong dimensions of the mu-map')
            else:
                output['exists'] = True
                output['fim'] = img
                if verbose: print 'i> using '+imtype+' from NIfTI file.'
        else:
            print 'e> provided '+imtype+' path is invalid.'
            return None
    elif isinstance(img, list):
        output['im'] = np.zeros((Cnt['SO_IMZ'], Cnt['SO_IMY'], Cnt['SO_IMX']), dtype=np.float32)
        if verbose: print 'w> '+imtype+' has not been provided -> using blank.'
        output['fim'] = ''
        output['exists'] = False
    #------------------------------------------------------------------------
    return output


#==============================================================================

def dynamic_timings(flist, offset=0):
    '''
    Get start and end frame timings from a list of dynamic PET frame definitions.
    flist can be 1D list of time duration for each dynamic frame, e.g.: flist = [15, 15, 15, 15, 30, 30, 30, ...]
    or a 2D list of lists having 2 entries: first for the number of repetitions and the other for the frame duration,
    e.g.: flist = [[4,15], [3,15], ...].
    offset adjusts for the start time (usually when prompts are strong enough over randoms)
    The output is a dictionary:
    out['timings'] = [[0, 15], [15, 30], [30, 45], [45, 60], [60, 90], [90, 120], [120, 150], ...]
    out['total'] = total time
    out['frames'] = array([ 15,  15,  15,  15,  30,  30,  30,  30, ...])

    '''
    if not isinstance(flist, list):
        raise TypeError('Wrong type of frame data input')
    if all([isinstance(t,(int, np.int32, np.int16, np.int8, np.uint8, np.uint16, np.uint32)) for t in flist]):
        tsum = offset
        # list of frame timings
        if offset>0:
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
    elif all([isinstance(t,list) and len(t)==2 for t in flist]):
        if offset>0:
            flist.insert(0,[1,offset])
            farray = np.asarray(flist, dtype=np.uint16)
        else:
            farray = np.array(flist)
        # number of dynamic frames
        nfrm = np.sum(farray[:,0])
        # list of frame duration
        frms = np.zeros(nfrm,dtype=np.uint16)
        #frame iterator
        fi = 0
        #time sum of frames
        tsum = 0
        # list of frame timings
        t_frames = []
        for i in range(0, farray.shape[0]):
            for t in range(0, farray[i,0]):
                # frame start time
                t0 = tsum
                tsum += farray[i,1]
                # frame end time
                t1 = tsum
                # append the timings to the list
                t_frames.append([t0, t1])
                frms[fi] = farray[i,1]
                fi += 1
    else:
        raise TypeError('Unrecognised data input.')
    # prepare the output dictionary
    out = {'total':tsum, 'frames':frms, 'timings':t_frames}
    return out
#=================================================================================================