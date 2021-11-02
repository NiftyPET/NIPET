"""auxilary functions for raw PET data processing."""
import glob
import logging
import os
import re
from collections.abc import Collection
from math import pi
from numbers import Integral
from os import fspath
from pathlib import Path
from textwrap import dedent

import cuvec as cu
import numpy as np
import pydicom as dcm
from miutil.fdio import hasext

from niftypet import nimpa
from niftypet.ninst import install_tools as tls

from . import mmr_auxe, resources

log = logging.getLogger(__name__)


def create_dir(pth):
    if not os.path.exists(pth):
        os.makedirs(pth)


def fwhm2sig(fwhm):
    Cnt = resources.get_mmr_constants()
    return (fwhm / Cnt['SZ_VOXY']) / (2 * (2 * np.log(2))**.5)


def lm_pos(datain, Cnt):
    '''get the position of table and gantry offset from the DICOM list-mode file'''
    if 'lm_dcm' in datain and os.path.isfile(datain['lm_dcm']):
        dhdr = dcm.read_file(datain['lm_dcm'])
    elif 'lm_ima' in datain and os.path.isfile(datain['lm_ima']):
        dhdr = dcm.read_file(datain['lm_ima'])
    else:
        log.error('DICOM list-mode data not found!')
        return None

    # ---find GantryOffset
    if dhdr[0x0018, 0x1020].value == 'syngo MR B20P' or dhdr[0x0018,
                                                             0x1020].value == 'syngo MR E11':
        flip = 1
        if [0x29, 0x1120] in dhdr:
            csainfo = dhdr[0x29, 0x1120].value
        else:
            log.error('DICOM field [0x29,0x1120] not found!')
            return None
    # this is for older scanner software
    elif dhdr[0x0018, 0x1020].value == 'syngo MR B18P':
        flip = -1
        if [0x29, 0x1020] in dhdr:
            csainfo = dhdr[0x29, 0x1020].value
        elif [0x29, 0x1120] in dhdr:
            csainfo = dhdr[0x29, 0x1120].value
        else:
            log.error(' DICOM field [0x29,0x1020] not found!')
            return None
    else:
        raise ValueError('unknown scanner software version!')

    # csainfo.find('GantryOffset')
    fi = re.search(b'GantryOffset(?!_)', csainfo).start()
    # regular expression for the needed three numbers
    xyz = re.findall(b'-?\\d.\\d{4,10}', csainfo[fi:fi + 200])
    # offset in cm
    # > hack to avoid other numbers (counting from the back)
    xoff = float(xyz[-3]) / 10
    yoff = float(xyz[-2]) / 10
    zoff = float(xyz[-1]) / 10

    goff = flip * np.array([xoff, yoff, zoff])
    log.info('gantry offset from DICOM:\n{}'.format(goff))

    fi = csainfo.find(b'TablePositionOrigin')
    # regular expression for the needed three numbers
    tpostr = csainfo[fi:fi + 200]
    tpo = re.sub(b'[^a-zA-Z0-9\\-]', b'', tpostr).split(b'M')
    tpozyx = np.array([float(tpo[-1]), float(tpo[-2]), float(tpo[-3])])
    log.info('table position origin from DICOM:\n{}'.format(tpozyx))

    return goff, tpozyx


def hdr_lm(datain, Cnt):
    '''Get the headers from DICOM list-mode data file'''
    if 'lm_dcm' in datain and os.path.isfile(datain['lm_dcm']):
        dhdr = dcm.read_file(datain['lm_dcm'])
    elif 'lm_ima' in datain and os.path.isfile(datain['lm_ima']):
        dhdr = dcm.read_file(datain['lm_ima'])
    else:
        log.error('DICOM list-mode data not found!')
        return None

    # list possible DICOM locations for list-mode interfile header
    lmhdr_locations = [[0x29, 0x1010], [0x29, 0x1110]]

    # for newer scanner software
    if dhdr[0x0018, 0x1020].value == 'syngo MR B20P' or dhdr[0x0018,
                                                             0x1020].value == 'syngo MR E11':
        # interfile header
        if [0x29, 0x1010] in dhdr:
            lmhdr = dhdr[0x29, 0x1010].value
            log.info('got LM interfile.')
        else:
            log.warning('DICOM field [0x29,0x1010] not found!')
            lmhdr = None

        # CSA Series Header Info
        if [0x29, 0x1120] in dhdr:
            csahdr = dhdr[0x29, 0x1120].value
            log.info('got CSA info.')
        else:
            log.error('DICOM field [0x29,0x1120] not found!')
            return lmhdr, None

    # for older scanner software
    elif dhdr[0x0018, 0x1020].value == 'syngo MR B18P':

        # find interfile header
        found_lmhdr = False
        for loc in lmhdr_locations:
            if loc in dhdr:
                lmhdr = dhdr[loc].value
                if '!INTERFILE' in lmhdr and 'start horizontal bed position' in lmhdr:
                    log.info(
                        dedent('''\
                        obtained list-mode interfile header from:
                        [{}, {}]''').format(hex(loc[0]), hex(loc[1])))
                    found_lmhdr = True
                    break
        if not found_lmhdr:
            log.warning('DICOM field with LM interfile header has not been found!')
            lmhdr = None

        # CSA Series Header Info
        if [0x29, 0x1020] in dhdr:
            csahdr = dhdr[0x29, 0x1020].value
            log.info('got CSA info.')
        elif [0x29, 0x1120] in dhdr:
            csahdr = dhdr[0x29, 0x1120].value
            log.info('got CSA info (may not be accurate, please check).')
        else:
            log.error('DICOM field [0x29,0x1020] not found!')
            return lmhdr, None

    return lmhdr.decode('ascii'), csahdr


def vh_bedpos(datain, Cnt):
    ihdr, csainfo = hdr_lm(datain, Cnt)

    # start horizontal bed position
    m = re.search(r'start horizontal bed position.*\d{1,3}\.*\d*', ihdr)
    fi = ihdr[m.start():m.end()].find('=')
    hbedpos = 0.1 * float(ihdr[m.start() + fi + 1:m.end()])

    # start vertical bed position
    m = re.search(r'start vertical bed position.*\d{1,3}\.*\d*', ihdr)
    fi = ihdr[m.start():m.end()].find('=')
    vbedpos = 0.1 * float(ihdr[m.start() + fi + 1:m.end()])

    return vbedpos, hbedpos


def hmu_resample0(hmupos, parts, Cnt):

    # output image sampling
    Cim = {
        'VXSRx': Cnt['SO_VXX'], 'VXSRy': Cnt['SO_VXY'], 'VXSRz': Cnt['SO_VXZ'],
        'VXNRx': Cnt['SO_IMX'], 'VXNRy': Cnt['SO_IMY'], 'VXNRz': Cnt['SO_IMZ']}
    # voxel position/offset
    Cim['OFFRx'] = -0.5 * Cim['VXNRx'] * Cim['VXSRx'] # -0.5*Cim['VXSRx']
    Cim['OFFRy'] = -0.5 * Cim['VXNRy'] * Cim['VXSRy'] # -0.5*Cim['VXSRy']
    Cim['OFFRz'] = -0.5 * Cim['VXNRz'] * Cim['VXSRz'] - hmupos[0]['HBedPos']

    Trnsl = (0.0, 0.0, 0.0)
    # transformation matrix
    A = np.array(
        [[1., 0., 0., Trnsl[0]], [0., 1., 0., Trnsl[1]], [0., 0., 1., Trnsl[2]], [0., 0., 0., 1.]],
        dtype=np.float32)

    imr = np.zeros((Cnt['SO_IMZ'], Cnt['SO_IMY'], Cnt['SO_IMX']), dtype=np.float32)
    # ===== Go through the hardware mu-map parts =====
    for i in parts:
        Cim['VXSOx'] = hmupos[i]['ivs'][2]
        Cim['VXSOy'] = hmupos[i]['ivs'][1]
        Cim['VXSOz'] = hmupos[i]['ivs'][0]
        Cim['VXNOx'] = hmupos[i]['shape'][2]
        Cim['VXNOy'] = hmupos[i]['shape'][1]
        Cim['VXNOz'] = hmupos[i]['shape'][0]
        # original image offset
        Cim['OFFOx'] = hmupos[i]['vpos'][2]
        Cim['OFFOy'] = hmupos[i]['vpos'][1]
        Cim['OFFOz'] = -hmupos[i]['vpos'][0]

        # resample!
        if i == 4:
            # does the bed just partly (no point doing all the long bed)
            offresZ = (-.5 * Cnt['SO_IMZ'] * Cnt['SO_VXZ'] - hmupos[0]['HBedPos'])
            # excess of the hrdwr mu-map axially
            excemuZ = offresZ - (-hmupos[4]['vpos'][0])
            # with extra margin of 5
            excevox = int(excemuZ / hmupos[4]['ivs'][0]) - 5
            newoffZ = -hmupos[4]['vpos'][0] + excevox * hmupos[4]['ivs'][0]
            # number of voxels included axially
            # with extra margin...
            inclvox = Cnt['SO_IMZ'] * Cnt['SO_VXZ'] / hmupos[4]['ivs'][0] + 10
            # truncate the image
            im = hmupos[i]['img'][excevox:excevox + inclvox, :, :]
            # update dictionary Cim
            Cim['OFFOz'] = newoffZ
            Cim['VXNOz'] = im.shape[0]
            imr += nimpa.prc.improc.resample(im, A, Cim)
        else:
            imr += nimpa.prc.improc.resample(hmupos[i]['img'], A, Cim)

    return imr


def time_diff_norm_acq(datain):
    if 'lm_dcm' in datain and os.path.isfile(datain['lm_dcm']):
        dcm_lm = dcm.read_file(datain['lm_dcm'])
    elif 'lm_ima' in datain and os.path.isfile(datain['lm_ima']):
        dcm_lm = dcm.read_file(datain['lm_ima'])
    else:
        log.error('dicom header of list-mode data does not exist.')
        return None

    # acq date
    s = dcm_lm[0x08, 0x21].value
    y = int(s[:4])
    m = int(s[4:6])
    d = int(s[6:8])
    # acq time
    s = dcm_lm[0x08, 0x32].value
    hrs = int(s[:2])
    mns = int(s[2:4])
    sec = int(s[4:6])

    # calib date
    s = dcm_lm[0x18, 0x1200].value
    cy = int(s[:4])
    cm = int(s[4:6])
    cd = int(s[6:8])
    # calib time
    s = dcm_lm[0x18, 0x1201].value
    chrs = int(s[:2])
    cmns = int(s[2:4])
    csec = int(s[4:6])

    tdiff = (hrs*3600 + mns*60 + sec) - (chrs*3600 + cmns*60 + csec)
    dhrs = tdiff / 3600
    dmns = (tdiff - 3600*dhrs) / 60
    if dhrs > 12:
        log.warning(
            'time difference between calibration and acquisition is: {} hrs and {} mins'.format(
                dhrs, dmns))

    if np.sum([cy - y, cm - m, cd - d]) != 0:
        log.warning(
            dedent('''\
            daily QC/calibration was performed on different day(!):
            {}-{}-{} vs. {}-{}-{}
            ''').format(cy, cm, cd, y, m, d))


def timings_from_list(flist, offset=0):
    """
    Get start and end frame timings from a list of dynamic PET frame definitions.
    Args:
      flist: can be 1D list of time duration for each dynamic frame, e.g.:
            flist = [15, 15, 15, 15, 30, 30, 30, ...]
        or a 2D list of lists having 2 entries:
        first for the number of repetitions and the other for the frame duration, e.g.:
            flist = [[4,15], [3,15], ...].
      offset: adjusts for the start time (usually when prompts are strong enough over randoms)
    Returns (dict):
      'timings': [[0, 15], [15, 30], [30, 45], [45, 60], [60, 90], [90, 120], [120, 150], ...]
      'total': total time
      'frames': array([ 15,  15,  15,  15,  30,  30,  30,  30, ...])
    """
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
        for i in range(0, farray.shape[0]):
            for _ in range(0, farray[i, 0]):
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
        raise TypeError('Unrecognised data input.')
    return {'total': tsum, 'frames': frms, 'timings': t_frames}


def axial_lut(Cnt):
    ''' Creates lookup tables (LUT) for linear indexes along the diagonals of Michelogram
    for span-11 calculations done on GPU.
    '''
    NRNG = Cnt['NRNG']

    if Cnt['SPN'] == 1:
        # number of rings calculated for the given ring range
        # (optionally we can use only part of the axial FOV)
        NRNG_c = Cnt['RNG_END'] - Cnt['RNG_STRT']
        # number of sinos in span-1
        NSN1_c = NRNG_c**2
        # correct for the max. ring difference in the full axial extent
        # (don't use ring range (1,63) as for this case no correction)
        if NRNG_c == 64:
            NSN1_c -= 12
    else:
        NRNG_c = NRNG
        NSN1_c = Cnt['NSN1']
        if Cnt['RNG_END'] != NRNG or Cnt['RNG_STRT'] != 0:
            log.error('the reduced axial FOV only works in span-1!')
            return None

    # ring dimensions
    rng = np.zeros((NRNG, 2), dtype=np.float32)
    z = -.5 * NRNG * Cnt['AXR']
    for i in range(NRNG):
        rng[i, 0] = z
        z += Cnt['AXR']
        rng[i, 1] = z

    # --create mapping from ring difference to segment number
    # ring difference range
    rd = list(range(-Cnt['MRD'], Cnt['MRD'] + 1))
    # ring difference to segment
    rd2sg = -1 * np.ones((len(rd), 2), dtype=np.int32)
    for i in range(len(rd)):
        for iseg in range(len(Cnt['MNRD'])):
            if (rd[i] >= Cnt['MNRD'][iseg]) and (rd[i] <= Cnt['MXRD'][iseg]):
                rd2sg[i, :] = np.array([rd[i], iseg])

    # create two Michelograms for segments (Mseg)
    # and absolute axial position for individual sinos (Mssrb) which is single slice rebinning
    Mssrb = -1 * np.ones((NRNG, NRNG), dtype=np.int32)
    Mseg = -1 * np.ones((NRNG, NRNG), dtype=np.int32)
    for r1 in range(Cnt['RNG_STRT'], Cnt['RNG_END']):
        for r0 in range(Cnt['RNG_STRT'], Cnt['RNG_END']):
            if abs(r1 - r0) > Cnt['MRD']:
                continue
            ssp = r0 + r1       # segment sino position (axially: 0-126)
            rd = r1 - r0
            jseg = rd2sg[rd2sg[:, 0] == rd, 1]
            Mssrb[r1, r0] = ssp
            Mseg[r1, r0] = jseg # negative segments are on top diagonals

    # np.savetxt("Mssrb.csv", Mssrb, delimiter=",", fmt='%d')
    # np.savetxt("Mseg.csv", Mseg, delimiter=",", fmt='%d')

    # create a Michelogram map from rings to sino number in span-11 (1..837)
    Msn = -1 * np.ones((NRNG, NRNG), dtype=np.int32)
    # number of span-1 sinos per sino in span-11
    Mnos = -1 * np.ones((NRNG, NRNG), dtype=np.int32)
    i = 0
    for iseg in range(0, len(Cnt['SEG'])):
        msk = (Mseg == iseg)
        Mtmp = np.copy(Mssrb)
        Mtmp[~msk] = -1
        uq = np.unique(Mtmp[msk])
        for u in range(0, len(uq)):
            # print(i)
            Msn[Mtmp == uq[u]] = i
            Mnos[Mtmp == uq[u]] = np.sum(Mtmp == uq[u])
            i += 1
    # np.savetxt("Mnos.csv", Mnos, delimiter=",", fmt='%d')
    # np.savetxt("Msn.csv", Msn, delimiter=",", fmt='%d')

    # ===full LUT
    sn1_rno = np.zeros((NSN1_c, 2), dtype=np.int16)
    sn1_ssrb = np.zeros((NSN1_c), dtype=np.int16)
    sn1_sn11 = np.zeros((NSN1_c), dtype=np.int16)
    sn1_sn11no = np.zeros((NSN1_c), dtype=np.int8)
    sni = 0                                           # full linear index, upto 4084
    Msn1 = -1 * np.ones((NRNG, NRNG), dtype=np.int16) # michelogram of sino numbers for spn-1
    for ro in range(0, NRNG):
        if ro == 0:
            oblique = 1
        else:
            oblique = 2
        for m in range(oblique):
            strt = NRNG * (ro + Cnt['RNG_STRT']) + Cnt['RNG_STRT']
            stop = (Cnt['RNG_STRT'] + NRNG_c) * NRNG
            step = NRNG + 1

            # goes along a diagonal started in the first row at r1
            for li in range(strt, stop, step):
                # linear indicies of michelogram
                # --> subscript indecies for positive and negative RDs

                if m == 0:
                    r1 = int(li / NRNG)
                    r0 = int(li - r1*NRNG)
                else:               # for positive now (? or vice versa)
                    r0 = int(li / NRNG)
                    r1 = int(li - r0*NRNG)
                if Msn[r1, r0] < 0: # avoid case when RD>MRD
                    continue

                sn1_rno[sni, 0] = r0
                sn1_rno[sni, 1] = r1

                sn1_ssrb[sni] = Mssrb[r1, r0]
                sn1_sn11[sni] = Msn[r0, r1]

                sn1_sn11no[sni] = Mnos[r0, r1]

                Msn1[r0, r1] = sni
                # --
                sni += 1

    # span-11 sino to SSRB
    sn11_ssrb = np.zeros(Cnt['NSN11'], dtype=np.int32)
    sn11_ssrb[:] -= 1
    sn1_ssrno = np.zeros(Cnt['NSEG0'], dtype=np.int8)
    for i in range(NSN1_c):
        sn11_ssrb[sn1_sn11[i]] = sn1_ssrb[i]
        sn1_ssrno[sn1_ssrb[i]] += 1

    sn11_ssrno = np.zeros(Cnt['NSEG0'], dtype=np.int8)
    for i in range(Cnt['NSN11']):
        if sn11_ssrb[i] > 0: sn11_ssrno[sn11_ssrb[i]] += 1

    sn1_ssrno = sn1_ssrno[np.unique(sn1_ssrb)]
    sn11_ssrno = sn11_ssrno[np.unique(sn1_ssrb)]
    sn11_ssrb = sn11_ssrb[sn11_ssrb >= 0]

    # ---------------------------------------------------------------------
    # linear index (along diagonals of Michelogram) to rings
    # the number of Michelogram elements considered in projection calculations
    NLI2R_c = int(NRNG_c**2 / 2. + NRNG_c/2.)
    # if the whole scanner is used then account for the MRD and subtract 6 ring permutations
    if NRNG_c == NRNG:
        NLI2R_c -= 6

    li2r = np.zeros((NLI2R_c, 2), dtype=np.int8)
    # the same as above but to sinos in span-11
    li2sn = np.zeros((NLI2R_c, 2), dtype=np.int16)
    li2sn1 = np.zeros((NLI2R_c, 2), dtype=np.int16)
    li2rng = np.zeros((NLI2R_c, 2), dtype=np.float32)
    # ...to number of sinos (nos)
    li2nos = np.zeros((NLI2R_c), dtype=np.int8)

    dli = 0
    for ro in range(0, NRNG_c):
        # selects the sub-Michelogram of the whole Michelogram
        strt = NRNG * (ro + Cnt['RNG_STRT']) + Cnt['RNG_STRT']
        stop = (Cnt['RNG_STRT'] + NRNG_c) * NRNG
        step = NRNG + 1

        # goes along a diagonal started in the first row at r2o
        for li in range(strt, stop, step):
            # from the linear indexes of Michelogram get the subscript indexes
            r1 = int(li / NRNG)
            r0 = int(li - r1*NRNG)
            if Msn[r1, r0] < 0:
                # avoid case when RD>MRD
                continue

            # li2r[0, dli] = r0
            # li2r[1, dli] = r1
            # # --
            # li2rng[0, dli] = rng[r0,0];
            # li2rng[1, dli] = rng[r1,0];
            # # --
            # li2sn[0, dli] = Msn[r0,r1]
            # li2sn[1, dli] = Msn[r1,r0]

            li2r[dli, 0] = r0
            li2r[dli, 1] = r1
            # --
            li2rng[dli, 0] = rng[r0, 0]
            li2rng[dli, 1] = rng[r1, 0]
            # --
            li2sn[dli, 0] = Msn[r0, r1]
            li2sn[dli, 1] = Msn[r1, r0]

            li2sn1[dli, 0] = Msn1[r0, r1]
            li2sn1[dli, 1] = Msn1[r1, r0]

            # li2sn[0, dli] = Msn[r1,r0]
            # li2sn[1, dli] = Msn[r0,r1]
            # --
            li2nos[dli] = Mnos[r1, r0]
            # --
            dli += 1
    # log.info('number of diagonal indexes (in Michelogram) accounted for: {}'.format(dli))
    # ---------------------------------------------------------------------

    axLUT = {
        'li2rno': li2r, 'li2sn': li2sn, 'li2sn1': li2sn1, 'li2nos': li2nos, 'li2rng': li2rng,
        'sn1_rno': sn1_rno, 'sn1_ssrb': sn1_ssrb, 'sn1_sn11': sn1_sn11, 'sn1_sn11no': sn1_sn11no,
        'sn11_ssrb': sn11_ssrb, 'sn1_ssrno': sn1_ssrno, 'sn11_ssrno': sn11_ssrno, 'Msn11': Msn,
        'Msn1': Msn1, 'Mnos': Mnos, 'rng': rng}

    log.debug('axial LUTs done.')

    return axLUT


def sino2ssr(sino, axLUT, Cnt):
    if Cnt['SPN'] == 1:
        slut = axLUT['sn1_ssrb']
        snno = Cnt['NSN1']
    elif Cnt['SPN'] == 11:
        slut = axLUT['sn11_ssrb']
        snno = Cnt['NSN11']
    else:
        log.error('unrecognised span! span={}'.format(Cnt['SPN']))
        return None

    ssr = np.zeros((Cnt['NSEG0'], Cnt['NSANGLES'], Cnt['NSBINS']), dtype=np.float32)

    for i in range(snno):
        ssr[slut[i], :, :] += sino[i, :, :]

    return ssr


def reduce_rings(pars, rs=0, re=64):
    """
    Reduce the axial rings for faster reconstructions, particularly simulations.
    This function customises axial FOV for reduced rings in range(rs,re).
    Note it only works in span-1 and ring re is not included in the reduced rings.
    Total number of used rings has to be even at all times.
    Arguments:
        pars -- scanner parameters: constants, LUTs
        rs -- start ring
        re -- end ring (not included in the resulting reduced rings)
    """

    if (re - rs) < 0 or ((re-rs) % 2) != 0:
        raise ValueError('The resulting number of rings has to be even and start ring (rs)'
                         ' smaller than end ring (re)')

    # > reduced rings work in span-1 only
    pars['Cnt']['SPN'] = 1

    # select the number of sinograms for the number of rings
    # RNG_STRT is included in detection
    # RNG_END is not included in detection process
    pars['Cnt']['RNG_STRT'] = rs
    pars['Cnt']['RNG_END'] = re
    # now change the voxels dims too
    vz0 = 2 * pars['Cnt']['RNG_STRT']
    vz1 = 2 * (pars['Cnt']['RNG_END'] - 1)
    # number of axial voxels
    pars['Cnt']['rSO_IMZ'] = vz1 - vz0 + 1
    pars['Cnt']['rSZ_IMZ'] = vz1 - vz0 + 1
    # axial voxel size for scatter (mu-map and emission image)
    # pars['Cnt']['SS_IMZ'] = pars['Cnt']['rSG_IMZ']
    # number of rings customised for the given ring range (only optional in span-1)
    rNRNG = pars['Cnt']['RNG_END'] - pars['Cnt']['RNG_STRT']
    pars['Cnt']['rNRNG'] = rNRNG
    # number of reduced sinos in span-1
    rNSN1 = rNRNG**2
    pars['Cnt']['rNSN1'] = rNSN1
    # correct for the limited max. ring difference in the full axial extent.
    # don't use ring range (1,63) as for this case no correction
    if rNRNG == 64: rNSN1 -= 12
    # apply the new ring subset to axial LUTs
    raxLUT = axial_lut(pars['Cnt'])
    # michelogram for reduced rings in span-1
    Msn1_c = raxLUT['Msn1']
    # michelogram for full ring case in span-1
    Msn1 = np.copy(pars['axLUT']['Msn1'])
    # from full span-1 sinogram index to reduced rings sinogram index
    rlut = np.zeros(rNSN1, dtype=np.int16)
    rlut[Msn1_c[Msn1_c >= 0]] = Msn1[Msn1_c >= 0]
    raxLUT['rLUT'] = rlut
    pars['axLUT'] = raxLUT


def transaxial_lut(Cnt, visualisation=False):
    '''
    Create a template 2D sinogram with gaps represented by 0 and
    any valid bin being represented by 1.
    Also create linear index for the whole sino with only valid bins.
    Angle index of the sino is used as the primary index (fast changing).
    '''

    if visualisation:
        # ---visualisation of the crystal ring in transaxial view
        p = 8      # pixel density of the visualisation
        VISXY = Cnt['SO_IMX'] * p
        T = np.zeros((VISXY, VISXY), dtype=np.float32)

    # --- crystal coordinates transaxially
    # > block width
    bw = 3.209

    # > block gap [cm]
    # dg = 0.474
    NTBLK = 56
    alpha = 0.1122 # 2*pi/NTBLK
    crs = np.zeros((Cnt['NCRS'], 4), dtype=np.float32)

    # > phi angle points in the middle and is used for obtaining the normal of detector block
    phi = 0.5*pi - alpha/2 - 0.001
    for bi in range(NTBLK):
        # > tangent point (ring against detector block)
        # ye = RE*np.sin(phi)
        # xe = RE*np.cos(phi)
        y = Cnt['R_RING'] * np.sin(phi)
        x = Cnt['R_RING'] * np.cos(phi)

        # > vector for the face of crystals
        pv = np.array([-y, x])
        pv /= np.sum(pv**2)**.5

        # > update phi for next block
        phi -= alpha

        # > end block points
        xcp = x + (bw/2) * pv[0]
        ycp = y + (bw/2) * pv[1]

        if visualisation:
            u = int(.5*VISXY + np.floor(xcp / (Cnt['SO_VXY'] / p)))
            v = int(.5*VISXY - np.ceil(ycp / (Cnt['SO_VXY'] / p)))
            T[v, u] = 5

        for n in range(1, 9):
            c = bi*9 + n - 1
            crs[c, 0] = xcp
            crs[c, 1] = ycp
            xc = x + (bw/2 - n*bw/8) * pv[0]
            yc = y + (bw/2 - n*bw/8) * pv[1]
            crs[c, 2] = xc
            crs[c, 3] = yc
            xcp = xc
            ycp = yc

            if visualisation:
                u = int(.5*VISXY + np.floor(xcp / (Cnt['SO_VXY'] / p)))
                v = int(.5*VISXY - np.ceil(ycp / (Cnt['SO_VXY'] / p)))
                T[v, u] = 2.5

    out = {'crs': crs}

    if visualisation:
        out['visual'] = T

    # > crystals reduced by the gaps (dead crystals)
    crsr = -1 * np.ones(Cnt['NCRS'], dtype=np.int16)
    ci = 0
    for i in range(Cnt['NCRS']):
        if (((i + Cnt['OFFGAP']) % Cnt['TGAP']) > 0):
            crsr[i] = ci
            ci += 1
        if visualisation:
            print('crsr[{}] = {}\n'.format(i, crsr[i]))

    out['crsri'] = crsr

    # ----------------------------------
    # sinogram definitions
    # > sinogram mask for dead crystals (gaps)
    msino = np.zeros((Cnt['NSBINS'], Cnt['NSANGLES']), dtype=np.int8)

    # LUT: sino -> crystal and crystal -> sino
    s2cF = np.zeros((Cnt['NSBINS'] * Cnt['NSANGLES'], 2), dtype=np.int16)
    c2sF = -1 * np.ones((Cnt['NCRS'], Cnt['NCRS']), dtype=np.int32)

    # > with projection bin <w> fast changing (c2s has angle changing fast).
    # > this is used in scatter estimation
    c2sFw = -1 * np.ones((Cnt['NCRS'], Cnt['NCRS']), dtype=np.int32)

    # > global sinogram index (linear) of live crystals (excludes gaps)
    awi = 0

    for iw in range(Cnt['NSBINS']):
        for ia in range(Cnt['NSANGLES']):
            c0 = int(
                np.floor((ia + 0.5 * (Cnt['NCRS'] - 2 + Cnt['NSBINS'] / 2 - iw)) % Cnt['NCRS']))
            c1 = int(
                np.floor(
                    (ia + 0.5 * (2 * Cnt['NCRS'] - 2 - Cnt['NSBINS'] / 2 + iw)) % Cnt['NCRS']))

            s2cF[ia + iw * Cnt['NSANGLES'], 0] = c0
            s2cF[ia + iw * Cnt['NSANGLES'], 1] = c1

            c2sF[c1, c0] = ia + iw * Cnt['NSANGLES']
            c2sF[c0, c1] = ia + iw * Cnt['NSANGLES']

            if (((((c0 + Cnt['OFFGAP']) % Cnt['TGAP']) *
                  ((c1 + Cnt['OFFGAP']) % Cnt['TGAP'])) > 0)):
                # > masking gaps in 2D sinogram
                msino[iw, ia] = 1
                awi += 1

            c2sFw[c1, c0] = iw + ia * Cnt['NSBINS']
            c2sFw[c0, c1] = iw + ia * Cnt['NSBINS']

    out['s2cF'] = s2cF
    out['c2sF'] = c2sF
    out['c2sFw'] = c2sFw
    out['msino'] = msino

    # > number of total transaxial live crystals (excludes gaps)
    out['Naw'] = awi

    s2c = np.zeros((out['Naw'], 2), dtype=np.int16)
    s2cr = np.zeros((out['Naw'], 2), dtype=np.int16)
    cr2s = np.zeros((Cnt['NCRSR'], Cnt['NCRSR']), dtype=np.int32)
    aw2sn = np.zeros((out['Naw'], 2), dtype=np.int16)
    aw2ali = np.zeros(out['Naw'], dtype=np.int32)

    # > live crystals which are in coincidence
    cij = np.zeros((Cnt['NCRSR'], Cnt['NCRSR']), dtype=np.int8)

    awi = 0

    for iw in range(Cnt['NSBINS']):
        for ia in range(Cnt['NSANGLES']):

            if (msino[iw, ia] > 0):
                c0 = s2cF[Cnt['NSANGLES'] * iw + ia, 0]
                c1 = s2cF[Cnt['NSANGLES'] * iw + ia, 1]

                s2c[awi, 0] = c0
                s2c[awi, 1] = c1

                s2cr[awi, 0] = crsr[c0]
                s2cr[awi, 1] = crsr[c1]

                # > reduced crystal index (after getting rid of crystal gaps)
                cr2s[crsr[c1], crsr[c0]] = awi
                cr2s[crsr[c0], crsr[c1]] = awi

                aw2sn[awi, 0] = ia
                aw2sn[awi, 1] = iw

                aw2ali[awi] = iw + Cnt['NSBINS'] * ia

                # > square matrix of crystals in coincidence
                cij[crsr[c0], crsr[c1]] = 1
                cij[crsr[c1], crsr[c0]] = 1

                awi += 1

    out['s2c'] = s2c
    out['s2cr'] = s2cr
    out['cr2s'] = cr2s
    out['aw2sn'] = aw2sn
    out['aw2ali'] = aw2ali
    out['cij'] = cij
    # ----------------------------------

    # # cij    - a square matrix of crystals in coincidence (transaxially)
    # # crsri  - indexes of crystals with the gap crystals taken out (therefore reduced)
    # # aw2sn  - LUT array [AW x 2] translating linear index into
    # #          a 2D sinogram with dead LOR (gaps)
    # # aw2ali - LUT from linear index of 2D full sinogram with gaps and bin-driven to
    # #          linear index without gaps and angle driven
    # # msino  - 2D sinogram with gaps marked (0). like a mask.
    # Naw, s2cAll, crsri, cij, aw2sn, aw2ali, msino = mmr_auxe.txlut( Cnt )
    # s2cF = s2cAll[0]
    # s2c  = s2cAll[1]
    # s2cr = s2cAll[2]
    # c2sF = s2cAll[3]
    # cr2s = s2cAll[4]

    # txLUT = {'cij':cij, 'crs':crs, 'crsri':crsri, 'msino':msino, 'aw2sn':aw2sn,
    #          'aw2ali':aw2ali, 's2c':s2c, 's2cr':s2cr, 's2cF':s2cF, 'Naw':Naw,
    #          'c2sF':c2sF, 'cr2s':cr2s}

    return out


# ================================================================================================
# Explore files in folder with raw PET/MR data
# ------------------------------------------------------------------------------------------------


def get_npfiles(dfile, datain):
    log.debug(
        dedent('''\
        ------------------------------------------------------------------
        file: {}
        ------------------------------------------------------------------
        ''').format(dfile))

    # pCT mu-map
    if os.path.basename(dfile) == 'mumap_pCT.npz':
        datain['mumapCT'] = dfile
        log.debug('mu-map for the object.')

    # DICOM UTE/Dixon mu-map
    if os.path.basename(dfile) == 'mumap-from-DICOM.npz':
        datain['mumapNPY'] = dfile
        log.debug('mu-map for the object.')

    if os.path.basename(dfile) == 'hmumap.npz':
        datain['hmumap'] = dfile
        log.debug('mu-map for hardware.')

    if os.path.basename(dfile)[:8] == 'sinos_s1':
        datain['sinos'] = dfile
        log.debug('prompt sinogram data.')

    # if os.path.basename(dfile)[:9]=='sinos_s11':
    #     datain['sinos11'] = dfile
    #     log.debug('prompt sinogram data in span-11.')


def get_niifiles(dfile, datain):
    log.debug(
        dedent('''\
        ------------------------------------------------------------------
        file: {}
        ------------------------------------------------------------------
        ''').format(dfile))

    # > NIfTI file of converted MR-based mu-map from DICOMs
    if os.path.basename(dfile).split('.nii')[0] == 'mumap-from-DICOM':
        datain['mumapNII'] = dfile
        log.debug('mu-map for the object.')

    # > NIfTI file of pseudo CT
    fpct = glob.glob(os.path.join(os.path.dirname(dfile), '*_synth.nii*'))
    if len(fpct) > 0:
        datain['pCT'] = fpct[0]
        log.debug('pseudoCT of the object.')

    fpct = glob.glob(os.path.join(os.path.dirname(dfile), '*_p[cC][tT].nii*'))
    if len(fpct) > 0:
        datain['pCT'] = fpct[0]
        log.debug('pseudoCT of the object.')

    # MR T1
    fmri = glob.glob(os.path.join(os.path.dirname(dfile), '[tT]1*.nii*'))
    if len(fmri) == 1:
        bnm = os.path.basename(fmri[0]).lower()
        if not {'giflabels', 'parcellation', 'pct', 'n4bias'}.intersection(bnm):
            datain['T1nii'] = fmri[0]
            log.debug('NIfTI for T1w of the object.')
    elif len(fmri) > 1:
        for fg in fmri:
            bnm = os.path.basename(fg).lower()
            if not {'giflabels', 'parcellation', 'pct', 'n4bias'}.intersection(bnm):
                if 'preferred' in bnm:
                    datain['T1nii'] = fg
                elif 'usable' in bnm:
                    datain['T1nii_2'] = fg

    # MR T1 N4bias-corrected
    fmri = glob.glob(os.path.join(os.path.dirname(dfile), '[tT]1*[nN]4bias*.nii*'))
    if len(fmri) == 1:
        bnm = os.path.basename(fmri[0]).lower()
        if not {'giflabels', 'parcellation', 'pct'}.intersection(bnm):
            datain['T1N4'] = fmri[0]
            log.debug('NIfTI for T1w of the object.')
    elif len(fmri) > 1:
        for fg in fmri:
            bnm = os.path.basename(fg).lower()
            if not {'giflabels', 'parcellation', 'pct'}.intersection(bnm):
                if 'preferred' in bnm:
                    datain['T1N4'] = fg
                elif 'usable' in bnm:
                    datain['T1N4_2'] = fg

    # T1w corrected
    fbc = glob.glob(os.path.join(os.path.dirname(dfile), '*gifbc.nii*'))
    if len(fbc) == 1:
        datain['T1bc'] = fbc[0]
        log.debug('NIfTI for bias corrected T1w of the object:\n{}'.format(fbc[0]))
    fbc = glob.glob(os.path.join(os.path.dirname(dfile), '*[tT]1*BiasCorrected.nii*'))
    if len(fbc) == 1:
        datain['T1bc'] = fbc[0]
        log.debug('NIfTI for bias corrected T1w of the object:\n{}'.format(fbc[0]))

    # T1-based labels after parcellation
    flbl = glob.glob(os.path.join(os.path.dirname(dfile), '*gif*labels.nii*'))
    if len(flbl) == 1:
        datain['T1lbl'] = flbl[0]
        log.debug('NIfTI for regional parcellations of the object:\n{}'.format(flbl[0]))
    flbl = glob.glob(os.path.join(os.path.dirname(dfile), '*[tT]1*[Pp]arcellation.nii*'))
    if len(flbl) == 1:
        datain['T1lbl'] = flbl[0]
        log.debug('NIfTI for regional parcellations of the object:\n{}'.format(flbl[0]))

    # reconstructed emission data without corrections, minimum 2 osem iter
    fpct = glob.glob(os.path.join(os.path.dirname(dfile), '*__ACbed.nii*'))
    if len(fpct) > 0:
        datain['em_nocrr'] = fpct[0]
        log.debug('pseudoCT of the object.')

    # reconstructed emission data with corrections, minimum 3 osem iter
    fpct = glob.glob(os.path.join(os.path.dirname(dfile), '*QNT*.nii*'))
    if len(fpct) > 0:
        datain['em_crr'] = fpct[0]
        log.debug('pseudoCT of the object.')


def get_dicoms(dfile, datain, Cnt):
    log.debug(
        dedent('''\
        ------------------------------------------------------------------
        file: {}
        ------------------------------------------------------------------
        ''').format(dfile))

    d = dcm.dcmread(dfile)
    dcmtype = nimpa.dcminfo(d)

    # > check if it is norm file
    if 'mmr' in dcmtype and 'norm' in dcmtype:
        if os.path.splitext(dfile)[-1].lower() == '.dcm':
            datain['nrm_dcm'] = dfile

            # > check if the binary file exists
            if os.path.isfile(dfile[:-4] + '.bf'):
                datain['nrm_bf'] = dfile[:-4] + '.bf'
            else:
                log.error('file does not exists:\n{}'.format(dfile[:-4] + '.bf'))
        elif os.path.splitext(dfile)[-1].lower() == '.ima':
            datain['nrm_ima'] = dfile
            # extract the binary norm data from the IMA DICOM
            if [0x7fe1, 0x1010] in d:
                nrm = d[0x7fe1, 0x1010].value
            else:
                log.error('could not find binary normalisation data in the IMA DICOM file.')
            # binary file name
            bf = os.path.splitext(dfile)[0] + '.bf'
            with open(bf, 'wb') as f:
                f.write(nrm)
            datain['nrm_bf'] = bf
            log.debug('saved component norm data to binary file: \n{}'.format(bf))

    # --- check if it is list-mode file
    elif 'mmr' in dcmtype and 'list' in dcmtype:
        if os.path.splitext(dfile)[-1] == '.dcm':
            datain['lm_dcm'] = dfile
            # check if the binary file exists
            if os.path.isfile(dfile[:-4] + '.bf'):
                datain['lm_bf'] = dfile[:-4] + '.bf'
            else:
                log.error('file does not exists: \n{}'.format(dfile[:-4] + '.bf'))
        elif os.path.splitext(dfile)[-1].lower() == '.ima':
            datain['lm_ima'] = dfile
            # extract the binary list-mode data from the IMA DICOM if it does not exist already
            # binary file name
            bf = os.path.splitext(dfile)[0] + '.bf'
            if [0x7fe1, 0x1010] in d and not os.path.isfile(bf):
                lm = d[0x7fe1, 0x1010].value
                with open(bf, 'wb') as f:
                    f.write(lm)
                datain['lm_bf'] = bf
                log.debug('saved list-mode data to binary file: \n{}'.format(bf))
            elif os.path.isfile(bf):
                log.debug(
                    'the binary list-mode data was already extracted from the IMA DICOM file.')
                datain['lm_bf'] = bf
            else:
                log.error('could not find binary list-mode data in the IMA DICOM file.')
                return None

        # > get info about the PET tracer being used
        lmhdr, csahdr = hdr_lm(datain, Cnt)

        # > if there is interfile header get the info from there
        if lmhdr is not None:
            f0 = lmhdr.find('isotope name')
        else:
            f0 = -1

        if f0 >= 0:
            f1 = f0 + lmhdr[f0:].find('\n')
            # regular expression for the isotope symbol
            # the name of isotope:
            istp = re.findall(r'(?<=:=)\s*\S*', lmhdr[f0:f1])[0]
            istp = istp.replace('-', '')
            Cnt['ISOTOPE'] = istp.strip()

        # > if no info in interfile header than look in the CSA header
        else:
            f0 = csahdr.find('RadionuclideCodeSequence')
            if f0 < 0:
                print('w> could not find isotope name.  enter manually into Cnt[' 'ISOTOPE' ']')
                return None
            istp_coded = re.search(r'(?<=CodeValue:)\S*', csahdr[f0:f0 + 100]).group()
            if istp_coded == 'C-111A1': Cnt['ISOTOPE'] = 'F18'
            elif istp_coded == 'C-105A1': Cnt['ISOTOPE'] = 'C11'
            elif istp_coded == 'C-B1038': Cnt['ISOTOPE'] = 'O15'
            elif istp_coded == 'C-128A2': Cnt['ISOTOPE'] = 'Ge68'
            elif istp_coded == 'C-131A3': Cnt['ISOTOPE'] = 'Ga68'
            else:
                print('w> could not find isotope name.  enter manually into Cnt[' 'ISOTOPE' ']')
                return None
        # ---

    # check if MR-based mu-map
    elif 'mumap' in dcmtype:
        datain['mumapDCM'] = os.path.dirname(dfile)
        if '#mumapDCM' not in datain:
            datain['#mumapDCM'] = 1
        else:
            datain['#mumapDCM'] += 1

    # check for MR T1w and T2w images
    elif 'mr' in dcmtype and 't1' in dcmtype:
        datain['T1DCM'] = os.path.dirname(dfile)
        if '#T1DCM' not in datain:
            datain['#T1DCM'] = 1
        else:
            datain['#T1DCM'] += 1

    elif 'mr' in dcmtype and 't2' in dcmtype:
        datain['T2DCM'] = os.path.dirname(dfile)
        if '#T2DCM' not in datain:
            datain['#T2DCM'] = 1
        else:
            datain['#T2DCM'] += 1

    # UTE's two sequences:
    elif 'mr' in dcmtype and 'ute2' in dcmtype:
        datain['UTE2'] = os.path.dirname(dfile)
        if '#UTE2' not in datain:
            datain['#UTE2'] = 1
        else:
            datain['#UTE2'] += 1

    elif 'mr' in dcmtype and 'ute1' in dcmtype:
        datain['UTE1'] = os.path.dirname(dfile)
        if '#UTE1' not in datain:
            datain['#UTE1'] = 1
        else:
            datain['#UTE1'] += 1


def explore_input(fldr, params, print_paths=False, recurse=1):
    """
    Args:
        recurse: int, [default: 1] subfolder deep. Use -1 for infinite recursion.
    """
    fldr, fpth = fspath(fldr), Path(fldr)
    Cnt = params.get('Cnt', params) # two ways of passing Cnt are here decoded

    if not os.path.isdir(fldr):
        log.error('provide a valid folder path for the data.')
        return

    # check for the availble data: list mode data, component-based norm and mu-maps
    # [dcm + bf] is one format of DICOM raw data; [ima] is another one used.
    # mu-map can be given from the scanner as an e.g., UTE-based, or pseudoCT through synthesis.
    datain = {'corepath': fldr}

    for f in fpth.iterdir():
        if f.is_file():
            if hasext(f, ("dcm", "ima")):
                get_dicoms(fspath(f), datain, Cnt)
            # elif hasext(f, "bf"):
            #     get_bf(f, datain, Cnt)
            elif hasext(f, ("npy", "npz", "dic")):
                get_npfiles(fspath(f), datain)
            elif hasext(f, ("nii", "nii.gz")):
                get_niifiles(fspath(f), datain)
        elif f.is_dir() and recurse:
            # go one level into subfolder
            extra = explore_input(f, params, recurse=recurse - 1)
            extra.pop('corepath')
            datain.update(extra)

    if print_paths:
        print('--------------------------------------------------')
        for x in datain:
            print(x, ':', datain[x])
        print('--------------------------------------------------')

    return datain


def putgaps(s, txLUT, Cnt, sino_no=0):

    # number of sino planes (2D sinos) depends on the span used
    if Cnt['SPN'] == 1:
        # number of rings calculated for the given ring range
        # (optionally we can use only part of the axial FOV)
        NRNG_c = Cnt['RNG_END'] - Cnt['RNG_STRT']
        # number of sinos in span-1
        nsinos = NRNG_c**2
        # correct for the max. ring difference in the full axial extent
        # (don't use ring range (1,63) as for this case no correction)
        if NRNG_c == 64:
            nsinos -= 12

    elif Cnt['SPN'] == 11:
        nsinos = Cnt['NSN11']

    # preallocate sino with gaps
    sino = np.zeros((Cnt['NSANGLES'], Cnt['NSBINS'], nsinos), dtype=np.float32)
    # fill the sino with gaps
    mmr_auxe.pgaps(sino, s.astype(np.float32), txLUT, Cnt, sino_no)
    sino = np.transpose(sino, (2, 0, 1))

    return sino.astype(s.dtype)


def remgaps(sino, txLUT, Cnt, output=None):
    # number of sino planes (2D sinos) depends on the span used
    nsinos = sino.shape[0]

    # preallocate output sino without gaps
    if output is None:
        output = cu.zeros((txLUT['Naw'], nsinos), dtype=np.float32)
    else:
        assert output.shape == txLUT['Naw'], nsinos
        assert output.dtype == np.dtype('float32')
    # fill the sino with gaps
    mmr_auxe.rgaps(output, cu.asarray(sino, dtype=np.float32), txLUT, Cnt)

    # return in the same data type as the input sino
    return output.astype(sino.dtype)


def mmrinit():
    # get the constants for the mMR
    Cnt = resources.get_mmr_constants()

    # transaxial look up tables
    txLUT = transaxial_lut(Cnt)
    Cnt['Naw'] = txLUT['Naw']

    # axial look up tables
    axLUT = axial_lut(Cnt)

    return Cnt, txLUT, axLUT


def get_mmrparams(hmu_dir=None):
    """ get all scanner parameters in one dictionary.
        hmudir: folder with the mMR hardware mu-maps if known;
                they will be stored in resources.py for the future use.
    """

    log.info(
        dedent("""\
            --------------------------------------------------------------
            Finding hardware mu-maps
            --------------------------------------------------------------"""))

    Cnt, txLUT, axLUT = mmrinit()

    # > hardware mu-maps
    if Cnt.get("HMUDIR", None):
        hmu_dir = Path(Cnt["HMUDIR"])
        # check each piece of the hardware components
        for i in Cnt["HMULIST"]:
            if not (hmu_dir / i).is_file():
                hmu_dir = None
                break
    # prompt for installation path
    if hmu_dir is None:
        Cnt["HMUDIR"] = tls.askdirectory(title="Folder for hardware mu-maps: ", name="HMUDIR")
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # update the path in resources.py
    tls.update_resources(Cnt)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    log.info("hardware mu-maps have been located")

    return {'Cnt': Cnt, 'txLUT': txLUT, 'axLUT': axLUT}
