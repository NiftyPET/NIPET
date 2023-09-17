"""Inveon auxiliary functions for raw PET data processing."""

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

import numpy as np
import pydicom as dcm
from miutil.fdio import hasext

from niftypet import nimpa

from . import mmr_auxe, resources

log = logging.getLogger(__name__)

#====================================================================
def create_dir(pth):
    if not os.path.exists(pth):
        os.makedirs(pth)

#====================================================================
def fwhm2sig(fwhm):
    Cnt = resources.get_mmr_constants()
    return (fwhm / Cnt['SZ_VOXY']) / (2 * (2 * np.log(2))**.5)



#====================================================================
def get_invpars():
    """
        get all scanner parameters in one dictionary
    """

    # > get the constants for the mMR
    Cnt = resources.get_inv_constants()
    
    # > transaxial look-up tables
    txLUT = transaxial_lut(Cnt)
    Cnt['NAW'] = txLUT['NAW']

    # > axial look-up tables
    axLUT = axial_lut(Cnt)

    return {'Cnt': Cnt, 'txLUT': txLUT, 'axLUT': axLUT}


#====================================================================
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


#====================================================================
def axial_lut(Cnt):
    ''' Creates lookup tables (LUT) for linear indexes along the diagonals of Michelogram
    for calculations done on GPU.
    '''
    NRNG = Cnt['NRNG']

    if Cnt['SPN'] == 1:
        # number of rings calculated for the given ring range
        # (optionally we can use only part of the axial FOV)
        NRNG_c = Cnt['RNG_END'] - Cnt['RNG_STRT']
        # number of sinos in span-1
        NSN1_c = NRNG_c**2
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



    #---------------------------------------------------------------------
    # > Michelogram for single slice rebinning
    # > (absolute axial position for individual sinos) 
    Mssrb = -1*np.ones((NRNG,NRNG), dtype=np.int16)
    for r1 in range(NRNG):
        for r0 in range(NRNG):
            ssp = r0+r1  #segment sino position
            Mssrb[r1,r0] = ssp 

    #---------------------------------------------------------------------
    # Michelogram for span-1 sino
    Msn = -1*np.ones((NRNG,NRNG), dtype=np.int16)
    # sino index -> ring index
    sn_rno = np.zeros((Cnt['NSN1'],2), dtype=np.int16)
    sn_ssrb= np.zeros((Cnt['NSN1']), dtype=np.int16)
    # full sinogram linear index, upto NRNG**2
    sni = 0 
    # go through all ring permutations
    for ro in range(0,NRNG):
        if ro==0:
            oblique = 1
        else:
            oblique = 2
        for m in range(oblique):
            strt = NRNG*ro
            stop = NRNG*NRNG
            step = NRNG+1
            #goes along a diagonal started in the first row at r1
            for li in range(strt, stop, step): 
                #linear indexes of Michelogram --> subscript indexes for positive and negative RDs
                if m==0:
                    r1 = int(li/NRNG)
                    r0 = int(li - r1*NRNG)
                else: 
                    #for positive now (? or vice versa)
                    r0 = int(li/NRNG)
                    r1 = int(li - r0*NRNG)
                sn_rno[sni,0] = r0
                sn_rno[sni,1] = r1
                sn_ssrb[sni] = Mssrb[r1,r0]
                Msn[r0,r1] = sni
                #--
                sni += 1

    # ring numbers for span-1 sino index to SSRB
    sn_ssrno = np.zeros(Cnt['NSEG0'], dtype=np.int8)
    for i in range(Cnt['NSN1']):
        sn_ssrno[sn_ssrb[i]] += 1
    sn_ssrno  =  sn_ssrno[np.unique(sn_ssrb)]
 

    #---------------------------------------------------------------------
    #linear index (along diagonals of Michelogram) to rings
    NLI2R = int(NRNG**2/2 + NRNG/2)
    li2r   = np.zeros((NLI2R,2), dtype=np.int8)
    li2sn  = np.zeros((NLI2R,2), dtype=np.int16)
    li2rng = np.zeros((NLI2R,2), dtype=np.float32)

    dli = 0
    for ro in range(0, NRNG):
        # selects the sub-Michelogram of the whole Michelogram
        strt = NRNG*ro
        stop = NRNG*NRNG
        step = NRNG+1

        # go along a diagonal starting in the first row
        for li in range(strt, stop, step): 
            #from the linear indexes of Michelogram get the subscript indexes
            r1 = int(li/NRNG)
            r0 = int(li - r1*NRNG)

            li2r[dli,0] = r0
            li2r[dli,1] = r1
            #--            
            li2rng[dli,0] = rng[r0,0]
            li2rng[dli,1] = rng[r1,0]
            #-- 
            li2sn[dli, 0] = Msn[r0,r1]
            li2sn[dli, 1] = Msn[r1,r0]

            dli += 1


    li2nos = np.ones((NLI2R), dtype=np.int8)

    return {'rng':rng, 'Msn':Msn, 'Mssrb':Mssrb,
            'li2nos':li2nos, 'li2rno':li2r, 'li2sn':li2sn, 'li2sn1':li2sn, 'li2rng':li2rng, 
            'sn1_rno':sn_rno, 'sn1_ssrb':sn_ssrb, 'sn1_ssrno':sn_ssrno
            }

    log.debug('axial LUTs done.')

    return axLUT


#====================================================================
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
    crs = np.zeros((Cnt['NCRS'], 4), dtype=np.float32)

    # > phi angle points in the middle and is used for obtaining the normal of detector block
    phi = 0.5*pi - 0.001 #- Cnt['ALPHA']/2
    for bi in range(Cnt['NTXBLK']):
        # > tangent point (ring against detector block)
        y = Cnt['R_RING'] * np.sin(phi)
        x = Cnt['R_RING'] * np.cos(phi)

        # > vector for the face of crystals
        pv = np.array([-y, x])
        pv /= np.sum(pv**2)**.5

        # > update phi for next block
        phi -= Cnt['ALPHA']

        # > end block points
        xcp = x + (Cnt['BLKWDTH']/2) * pv[0]
        ycp = y + (Cnt['BLKWDTH']/2) * pv[1]

        if visualisation:
            u = int(.5*VISXY + np.floor(xcp / (Cnt['SO_VXY'] / p)))
            v = int(.5*VISXY - np.ceil(ycp / (Cnt['SO_VXY'] / p)))
            T[v, u] = 5

        for n in range(1,Cnt['NCRSBLK']+1):
            c = bi*Cnt['NCRSBLK'] + n -1
            crs[c,0] = xcp
            crs[c,1] = ycp
            xc = x + (0.5*Cnt['BLKWDTH']-n*Cnt['BLKWDTH']/Cnt['NCRSBLK'])*pv[0]
            yc = y + (0.5*Cnt['BLKWDTH']-n*Cnt['BLKWDTH']/Cnt['NCRSBLK'])*pv[1]
            crs[c,2] = xc
            crs[c,3] = yc
            xcp = xc
            ycp = yc

            if visualisation:
                u = int(.5*VISXY + np.floor(xcp / (Cnt['SO_VXY'] / p)))
                v = int(.5*VISXY - np.ceil(ycp / (Cnt['SO_VXY'] / p)))
                T[v, u] = 2.5

    out = {'crs': crs}

    if visualisation:
        out['visual'] = T


    # ----------------------------------
    # sinogram definitions

    # LUT: sino -> crystal and crystal -> sino
    s2c = np.zeros((Cnt['NSBINS'] * Cnt['NSANGLES'], 2), dtype=np.int16)
    c2s = -1 * np.ones((Cnt['NCRS'], Cnt['NCRS']), dtype=np.int32)

    # > with projection bin <w> fast changing (c2s has angle changing fast).
    # > this is used in scatter estimation
    c2s_w = -1 * np.ones((Cnt['NCRS'], Cnt['NCRS']), dtype=np.int32)

    # > live crystals which are in coincidence
    cij = np.zeros((Cnt['NCRS'], Cnt['NCRS']), dtype=np.int8)

    aw2ali = np.zeros(Cnt['NSBINS'] * Cnt['NSANGLES'], dtype=np.int32)

    aw2sn = np.zeros((Cnt['NSBINS'] * Cnt['NSANGLES'], 2), dtype=np.int16)


    # > global sinogram index (linear) of live crystals (excludes gaps)
    awi = 0

    for iw in range(Cnt['NSBINS']):
        for ia in range(Cnt['NSANGLES']):
            c0 = int(
                np.floor((ia + 0.5 * (Cnt['NCRS'] - 0 + Cnt['NSBINS'] / 2 - iw)) % Cnt['NCRS']))
            c1 = int(
                np.floor(
                    (ia + 0.5 * (2 * Cnt['NCRS'] - 0 - Cnt['NSBINS'] / 2 + iw)) % Cnt['NCRS']))

            s2c[ia + iw * Cnt['NSANGLES'], 0] = c0
            s2c[ia + iw * Cnt['NSANGLES'], 1] = c1

            c2s[c1, c0] = ia + iw * Cnt['NSANGLES']
            c2s[c0, c1] = ia + iw * Cnt['NSANGLES']

            c2s_w[c1, c0] = iw + ia * Cnt['NSBINS']
            c2s_w[c0, c1] = iw + ia * Cnt['NSBINS']

            # > square matrix of crystals in coincidence
            cij[c0, c1] = 1
            cij[c1, c0] = 1

            # > LUT from linear index of 2D full sinogram bin-driven to
            # > linear index and angle driven
            aw2ali[awi] = iw + Cnt['NSBINS'] * ia

            aw2sn[awi, 0] = ia
            aw2sn[awi, 1] = iw

            awi += 1


    out['s2c'] = s2c
    out['c2s'] = c2s
    out['c2s_w'] = c2s_w
    out['cij'] = cij
    out['aw2sn'] = aw2sn
    out['aw2ali'] = aw2ali

    # > number of total transaxial live crystals (excludes gaps)
    out['NAW'] = awi

    # ----------------------------------

    # cij    - a square matrix of crystals in coincidence (transaxially)
    # aw2sn  - LUT array [AW x 2] translating linear index into
    #          a 2D sinogram with dead LOR (gaps)
    # aw2ali - LUT from linear index of 2D full sinogram with gaps and bin-driven to
    #          linear index without gaps and angle driven

    return out
