"""hist.py: processing of PET list-mode data: histogramming and randoms estimation."""
import logging
import os
from collections.abc import Collection
from numbers import Integral

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi

from niftypet import nimpa

from .. import mmraux

# CUDA extension module
from . import mmr_lmproc

log = logging.getLogger(__name__)

# ===============================================================================
# HISTOGRAM THE LIST-MODE DATA
# -------------------------------------------------------------------------------


def mmrhist(datain, scanner_params, t0=0, t1=0, outpath='', frms=None, use_stored=False,
            store=False, cmass_sig=5):
    '''
    Process the list-mode data and return histogram, head curves,
    and centre of mass for motion detection.
    '''
    # constants, transaxial and axial LUTs are extracted
    Cnt = scanner_params['Cnt']
    txLUT = scanner_params['txLUT']
    axLUT = scanner_params['axLUT']

    return hist(datain, txLUT, axLUT, Cnt, frms=frms, use_stored=use_stored, store=store,
                outpath=outpath, t0=t0, t1=t1, cmass_sig=cmass_sig)


def hist(
        datain,
        txLUT,
        axLUT,
        Cnt,
        t0=0,
        t1=0,
        cmass_sig=5,
        frms=None,        # np.array([0], dtype=np.uint16),
        use_stored=False,
        store=False,
        outpath=''):
    '''
    Process list mode data with histogramming and optional bootstrapping:
    Cnt['BTP'] = 0: no bootstrapping [default];
    Cnt['BTP'] = 1: non-parametric bootstrapping;
    Cnt['BTP'] = 2: parametric bootstrapping (using Poisson distribution with mean = 1)
    '''

    if Cnt['SPN'] == 1: nsinos = Cnt['NSN1']
    elif Cnt['SPN'] == 11: nsinos = Cnt['NSN11']
    elif Cnt['SPN'] == 0: nsinos = Cnt['NSEG0']

    log.debug('histogramming with span {}.'.format(Cnt['SPN']))

    if (use_stored is True and 'sinos' in datain
            and os.path.basename(datain['sinos']) == f"sinos_s{Cnt['SPN']}_frm-{t0}-{t1}.npz"):
        hstout = dict(np.load(datain['sinos'], allow_pickle=True))
        nitag = len(hstout['phc'])
        log.debug('acquisition duration by integrating time tags is {} sec.'.format(nitag))

    elif os.path.isfile(datain['lm_bf']):
        # gather info about the LM time tags
        nele, ttags, tpos = mmr_lmproc.lminfo(datain['lm_bf'])
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
        mss = np.zeros((nitag), dtype=np.float32)

        bck = np.zeros((2, nitag, Cnt['NBCKT']), dtype=np.uint32)
        fan = np.zeros((Cnt['NRNG'], Cnt['NCRS']), dtype=np.uint32)

        # > prompt and delayed sinograms
        psino = np.zeros((nsinos, Cnt['NSANGLES'], Cnt['NSBINS']), dtype=np.uint16)
        dsino = np.zeros((nsinos, Cnt['NSANGLES'], Cnt['NSBINS']), dtype=np.uint16)

        # > single slice rebinned prompots
        ssr = np.zeros((Cnt['NSEG0'], Cnt['NSANGLES'], Cnt['NSBINS']), dtype=np.uint32)

        hstout = {
            'phc': phc, 'dhc': dhc, 'mss': mss, 'pvs': pvs, 'bck': bck, 'fan': fan, 'psn': psino,
            'dsn': dsino, 'ssr': ssr}
        # ---------------------------------------

        # do the histogramming and processing
        mmr_lmproc.hist(hstout, datain['lm_bf'], t0, t1, txLUT, axLUT, Cnt)

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

    # short (interval) projection views
    pvs_sgtl = np.right_shift(hstout['pvs'], 8).astype(np.float32)
    pvs_crnl = np.bitwise_and(hstout['pvs'], 255).astype(np.float32)

    cmass = Cnt['SO_VXZ'] * ndi.gaussian_filter(hstout['mss'], cmass_sig, mode='mirror')
    log.debug(
        'centre of mass of axial radiodistribution (filtered with Gaussian of SD ={}):  COMPLETED.'
        .format(cmass_sig))

    # ========================= BUCKET SINGLES =========================
    # > number of single rates reported for the given second
    # > the last two bits are used for the number of reports
    nsr = (hstout['bck'][1, :, :] >> 30)

    # > average in a second period
    hstout['bck'][0, nsr > 0] = hstout['bck'][0, nsr > 0] / nsr[nsr > 0]

    # > time indeces when single rates given
    tmsk = np.sum(nsr, axis=1) > 0
    single_rate = np.copy(hstout['bck'][0, tmsk, :])

    # > time
    t = np.arange(nitag)
    t = t[tmsk]

    # > get the average bucket singles:
    buckets = np.int32(np.sum(single_rate, axis=0) / single_rate.shape[0])
    log.debug('dynamic and static buckets single rates:  COMPLETED.')
    # ==================================================================

    # account for the fact that when t0==t1 that means that full dataset is processed
    if t0 == t1: t1 = t0 + nitag

    return {
        't0': t0,
        't1': t1,
        'dur': t1 - t0,           # duration
        'phc': hstout['phc'],     # prompts head curve
        'dhc': hstout['dhc'],     # delayeds head curve
        'cmass': cmass,           # centre of mass of the radiodistribution in axial direction
        'pvs_sgtl': pvs_sgtl,     # sagittal projection views in short intervals
        'pvs_crnl': pvs_crnl,     # coronal projection views in short intervals
        'fansums': hstout['fan'], # fan sums of delayeds for variance reduction of randoms
        'sngl_rate': single_rate, # bucket singles over time
        'tsngl': t,               # time points of singles measurements in list-mode data
        'buckets': buckets,       # average bucket singles
        'psino': hstout['psn'].astype(np.uint16), # prompt sinogram
        'dsino': hstout['dsn'].astype(np.uint16), # delayeds sinogram
        'pssr': hstout['ssr']     # single-slice rebinned sinogram of prompts
    }  # yapf: disable


# ==============================================================================
# GET REDUCED VARIANCE RANDOMS
# ------------------------------------------------------------------------------


def randoms(hst, scanner_params, gpu_dim=False):
    '''
    Get the estimated sinogram of random events using the delayed event
    measurement.  The delayed sinogram is in the histogram dictionary
    obtained from the processing of the list-mode data.
    '''

    # constants, transaxial and axial LUTs are extracted
    Cnt = scanner_params['Cnt']
    txLUT = scanner_params['txLUT']
    axLUT = scanner_params['axLUT']

    rndsino, singles = rand(hst['fansums'], txLUT, axLUT, Cnt)

    if gpu_dim:
        rsng = mmraux.remgaps(rndsino, txLUT, Cnt)
        return rsng, singles
    else:
        return rndsino, singles


def rand(fansums, txLUT, axLUT, Cnt):
    if Cnt['SPN'] == 1: nsinos = Cnt['NSN1']
    elif Cnt['SPN'] == 11: nsinos = Cnt['NSN11']
    elif Cnt['SPN'] == 0: nsinos = Cnt['NSEG0']

    # random sino and estimated crystal map of singles put into a dictionary
    rsn = np.zeros((nsinos, Cnt['NSANGLES'], Cnt['NSBINS']), dtype=np.float32)
    cmap = np.zeros((Cnt['NCRS'], Cnt['NRNG']), dtype=np.float32)
    rndout = {'rsn': rsn, 'cmap': cmap}

    mmr_lmproc.rand(rndout, fansums, txLUT, axLUT, Cnt)

    return rndout['rsn'], rndout['cmap']


# ===============================================================================
# NEW!! GET REDUCED VARIANCE RANDOMS (BASED ON PROMPTS)
# -------------------------------------------------------------------------------


def prand(fansums, pmsk, txLUT, axLUT, Cnt):
    if Cnt['SPN'] == 1: nsinos = Cnt['NSN1']
    elif Cnt['SPN'] == 11: nsinos = Cnt['NSN11']
    elif Cnt['SPN'] == 0: nsinos = Cnt['NSEG0']

    # number of frames
    nfrm = fansums.shape[0]
    log.debug('# of dynamic frames: {}.'.format(nfrm))

    # random sino and estimated crystal map of singles put into a dictionary
    rsn = np.zeros((nsinos, Cnt['NSANGLES'], Cnt['NSBINS']), dtype=np.float32)
    cmap = np.zeros((Cnt['NCRS'], Cnt['NRNG']), dtype=np.float32)
    rndout = {'rsn': rsn, 'cmap': cmap}

    # save results for each frame

    rsino = np.zeros((nfrm, nsinos, Cnt['NSANGLES'], Cnt['NSBINS']), dtype=np.float32)
    crmap = np.zeros((nfrm, Cnt['NCRS'], Cnt['NRNG']), dtype=np.float32)

    for i in range(nfrm):
        rndout['rsn'][:, :, :] = 0
        rndout['cmap'][:, :] = 0
        mmr_lmproc.prand(rndout, pmsk, fansums[i, :, :], txLUT, axLUT, Cnt)
        rsino[i, :, :, :] = rndout['rsn']
        crmap[i, :, :] = rndout['cmap']

    if nfrm == 1:
        rsino = rsino[0, :, :, :]
        crmap = crmap[0, :, :]

    return rsino, crmap


def sino2nii(sino, Cnt, fpth):
    '''save sinogram in span-11 into NIfTI file'''
    # number of segments
    segn = len(Cnt['SEG'])
    cumseg = np.cumsum(Cnt['SEG'])
    cumseg = np.append([0], cumseg)

    # plane offset (relative to 127 planes of seg 0) for each segment
    OFF = np.min(abs(np.append([Cnt['MNRD']], [Cnt['MXRD']], axis=0)), axis=0)
    niisn = np.zeros((Cnt['SEG'][0], Cnt['NSANGLES'], Cnt['NSBINS'], segn), dtype=sino.dtype)

    # first segment (with direct planes)
    # tmp =
    niisn[:, :, :, 0] = sino[Cnt['SEG'][0] - 1::-1, ::-1, ::-1]

    for iseg in range(1, segn):
        niisn[OFF[iseg]:OFF[iseg] + Cnt['SEG'][iseg], :, :,
              iseg] = sino[cumseg[iseg] + Cnt['SEG'][iseg] - 1:cumseg[iseg] - 1:-1, ::-1, ::-1]

    niisn = np.transpose(niisn, (2, 1, 0, 3))

    nim = nib.Nifti1Image(niisn, np.eye(4))
    nib.save(nim, fpth)


# ================================================================================
# create michelogram map for emission data, only when the input sino in in span-1
def get_michem(sino, axLUT, Cnt):
    if Cnt['SPN'] == 1:
        slut = np.arange(Cnt['NSN1']) # for span 1, one-to-one mapping
    elif Cnt['SPN'] == 11:
        slut = axLUT['sn1_sn11']
    else:
        raise ValueError('sino is neither in span-1 or span-11')

    # acitivity michelogram
    Mem = np.zeros((Cnt['NRNG'], Cnt['NRNG']), dtype=np.float32)
    # sino to ring number & sino-1 to sino-11 index:
    sn1_rno = axLUT['sn1_rno']
    # sum all the sinograms inside
    ssm = np.sum(sino, axis=(1, 2))

    for sni in range(len(sn1_rno)):
        r0 = sn1_rno[sni, 0]
        r1 = sn1_rno[sni, 1]
        Mem[r1, r0] = ssm[slut[sni]]

    return Mem


# ================================================================================
# --------------------------------------------------------------------------------
# ================================================================================


def draw_frames(hst, tfrms, plot_diff=True):
    import matplotlib.pyplot as plt
    diff = np.int64(hst['phc']) - np.int64(hst['dhc'])
    plt.figure()
    plt.plot(hst['phc'], label='prompts')
    plt.plot(hst['dhc'], label='randoms')
    if plot_diff:
        plt.plot(diff, label='difference')

    K = [f[0] for f in tfrms if isinstance(f, list)]
    for k in K:
        yval = hst['phc'][k]
        if yval < 0.2 * np.max(hst['phc']):
            yval = 0.2 * np.max(hst['phc'])
        plt.plot([k, k], [0, yval], 'k--', lw=.75)
    plt.legend()
    plt.xlabel('time [sec]')
    plt.ylabel('counts/sec')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))


def get_time_offset(hst):
    '''
    Detects when the signal is stronger than the randoms (noise) in the list-mode data stream.
    '''
    # detect when the signal (here prompt data) is almost as strong as randoms
    s = hst['dhc'] > 0.98 * hst['phc']
    # return index, which will constitute time in seconds, for this offset
    return len(s) - np.argmax(s[::-1]) - 1


def split_frames(hst, Tref=0, t0=0, t1=0):
    '''
    Splits the whole acquisition data into approximately statistically
    equivalent frames relative to the reference frame whose duration is
    Tref or t1-t0.  The next frames will have a similar count level.
    hst: histogram dictionary
    Tref: reference duration in seconds
    t0: start time of the reference frame
    t1: end time of the reference frame
    '''
    # get the offset
    toff = get_time_offset(hst)
    # difference between prompts and randoms
    diff = np.int64(hst['phc']) - np.int64(hst['dhc'])

    # follow up index
    i = t0 + (toff) * (t0 <= 0)
    if Tref > 0:
        j = i + Tref
    elif t1 > 0:
        j = t1 + (toff) * (t0 <= 0)
    else:
        raise ValueError('e> could not figure out the reference frame.')

    # reference count level
    cref = np.sum(diff[i:j])
    # cumulative sum of the difference
    csum = np.cumsum(diff)

    i = 0
    j = toff
    # threshold to be achieved
    thrsh = csum[j - 1] + cref
    fdur = [toff]
    frms = ['timings', [0, toff]]
    clvl = [0]
    print('frame counts t(%d,%d) = %d. diff=%d' % (i, j, clvl[-1], np.sum(diff[i:j]) - cref))
    while thrsh < csum[-1]:
        i = j
        j = np.argmax(csum > thrsh)
        fdur.append(j - i)
        frms.append([i, j])
        clvl.append(np.sum(diff[i:j]))
        print('frame counts t(%d,%d) = %d. diff=%d' % (i, j, clvl[-1], np.sum(diff[i:j]) - cref))
        thrsh += cref
    # last remianing frame
    i = j
    j = hst['dur']
    # if last frame is short, include it in the last one.
    if np.sum(diff[i:]) > .5 * cref:
        fdur.append(j - i)
        frms.append([i, j])
        clvl.append(np.sum(diff[i:]))
    else:
        fdur[-1] += j - i
        frms[-1][-1] += j - i
        clvl[-1] += np.sum(diff[i:])
        i = frms[-1][0]
    print('frame counts t(%d,%d) = %d. diff=%d' % (i, j, clvl[-1], np.sum(diff[i:j]) - cref))
    return {'timings': frms, 'fdur': fdur, 'fcnts': clvl, 'offset': toff, 'csum': csum}


def frame_position(hst, tposition, Cref=0, tr0=0, tr1=15, verbose=True):
    '''
    Inputs:
        hst: histogram data
        tposition: time position (middle point) of the frame to be defined
        Cref: reference count level to be in the frame (prompts - delays)
        tr0, tr1: time definitions of reference frame whose count level
        will be used as the reference Cref.  If Cref is not defined (i.e. = 0)
        then the tr0 and tr1 will be used.
    '''
    # claculate the difference between the prompts and delays (more realistic count level)
    diff = np.int64(hst['phc']) - np.int64(hst['dhc'])
    # cumulative sum for calculating count levels in arbitrary time windows
    cumdiff = np.cumsum(diff)

    if Cref == 0:
        Cref = cumdiff[tr1] - cumdiff[tr0 - 1]

    if Cref < 0:
        raise ValueError('The reference count level has to be non-negative')

    (log.info if verbose else log.debug)('reference count level: {}.'.format(Cref))

    stp0 = 0
    stp1 = 0
    Cw = 0
    while Cw < Cref:
        # check if it is possible to widen the sampling window both ways
        if (tposition - stp0 - 1) > 0: stp0 += 1
        if (tposition + stp1 + 1) <= len(cumdiff) - 1: stp1 += 1
        Cw = cumdiff[tposition + stp1] - cumdiff[tposition - stp0 - 1]

    tw0 = tposition - stp0
    tw1 = tposition + stp1
    Tw = tw1 - tw0
    (log.info if verbose else log.debug)(
        'time window t[{}, {}] of duration T={} and count level Cw={}'.format(tw0, tw1, Tw, Cw))

    return (tw0, tw1)


def auxilary_frames(hst, t_frms, Cref=0, tr0=0, tr1=15, verbose=True):
    '''
    Get auxiliary time frames with equal count levels for constant precision in
    the estimation of subject motion based on PET data.
    '''

    # calculate the difference between the prompts and delays (more realistic count level)
    diff = np.int64(hst['phc']) - np.int64(hst['dhc'])

    # previous frame (time tuple)
    prev_frm = (0, 0)
    # previous frame index
    prev_i = -1
    # look up table to the auxilary frames from the regular ones
    timings = []
    fi2afi = []
    for i in range(len(t_frms)):
        # time point as an average between the frame end points
        tp = int(np.mean([t_frms[i][0], t_frms[i][1]]))
        # alternative (more accurate) average through centre of mass
        t0 = t_frms[i][0]
        t1 = t_frms[i][1]
        if t1 >= hst['dur']: t1 = hst['dur'] - 1
        t = np.arange(t0, t1)
        tcm = np.sum(diff[t] * t) / np.sum(diff[t])
        # get the tuple of the equivalent count level frame
        frm = frame_position(hst, tcm, tr0=tr0, tr1=tr1, verbose=False)
        # form the LUT
        if frm != prev_frm:
            prev_frm = frm
            prev_i += 1
            timings.append(list(frm))
        fi2afi.append(prev_i)
        if verbose:
            print('t[{}, {}]; tp={}, tcm={} => frm id:{}, timings:{}'.format(
                t_frms[i][0], t_frms[i][1], tp, tcm, fi2afi[-1], timings[-1]))
    # form the list of auxilary dynamic frames of equivalent count level (as in Cref)
    mfrm = ['fluid'] + timings
    return {'timings': mfrm, 'frame_idx': fi2afi}


def dynamic_timings(flist, offset=0):
    '''
    Get start and end frame timings from a list of dynamic PET frame definitions.
    Arguments:
      flist: can be 1D list of time duration for each dynamic frame, e.g.:
            flist = [15, 15, 15, 15, 30, 30, 30, ...]
        or a 2D list of lists having 2 entries per definition:
        first for the number of repetitions and the other for the frame duration, e.g.:
            flist = ['def', [4, 15], [8, 30], ...],
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
            t_frames = ['timings', [0, offset]]
        else:
            t_frames = ['timings']
        for i in range(len(flist)):
            # frame start time
            t0 = tsum
            tsum += flist[i]
            # frame end time
            t1 = tsum
            # append the timings to the list
            t_frames.append([t0, t1])
        frms = np.uint16(flist)
    elif flist[0] == 'def' and all(isinstance(t, Collection) and len(t) == 2 for t in flist[1:]):
        flist = flist[1:]
        if offset > 0:
            flist.insert(0, [0, offset])
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
        t_frames = ['timings']
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
