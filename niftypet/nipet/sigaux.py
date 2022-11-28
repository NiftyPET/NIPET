import logging
import os
from math import pi

import h5py
import numpy as np
import math

from . import resources


log = logging.getLogger(__name__)


def constants_h5(pthfn):
    with h5py.File(pthfn, 'r') as f:
        # coincidence event mode
        cncdmd = f['HeaderData']['AcqParameters']['EDCATParameters']['coinOutputMode'][0]
        if cncdmd == 802:
            # bytes per event in this mode:
            bpe = 6
            log.info("list-mode data in NOMINAL mode (6 bytes per event)")
        elif cncdmd == 803:
            bpe = 16
            log.error(
                "list-mode data in CALIBRATION mode (16 bytes per event) not currently supported")
        elif cncdmd == 805:
            bpe = 8
            log.error(
                "the list-mode data in ENERGY mode (8 bytes per event) not currently supported")
        else:
            bpe = 0
            log.error("list-mode data in UNKNOWN mode")

        # toff: scan start time marker (used as offset)
        CntH5 = {
            'toff': f['HeaderData']['AcqStats']['frameStartCoincTStamp'][0],
            'Deff': f['HeaderData']['SystemGeometry']['effectiveRingDiameter'][0]/10,
            'TFOV': f['HeaderData']['AcqParameters']['EDCATParameters']['transAxialFOV'][0],
            'cpitch': f['HeaderData']['SystemGeometry']['interCrystalPitch'][0],
            'bpitch': f['HeaderData']['SystemGeometry']['interBlockPitch'][0],
            'AXBGAP' : f['HeaderData']['SystemGeometry']['axialBlockGap'][0]    /10,
            'AXCGAP' : f['HeaderData']['SystemGeometry']['axialCassetteGap'][0] /10,
            'TXBGAP' : f['HeaderData']['SystemGeometry']['radialBlockGap'][0]   /10,
            'TXCGAP' : f['HeaderData']['SystemGeometry']['radialCassetteGap'][0]/10,
            'exLOR': f['HeaderData']['AcqParameters']['RxScanParameters']['extraRsForTFOV'][0],
            'AXCB': f['HeaderData']['SystemGeometry']['axialCrystalsPerBlock'][0],
            'AXBU': f['HeaderData']['SystemGeometry']['axialBlocksPerUnit'][0],
            'AXUM': f['HeaderData']['SystemGeometry']['axialUnitsPerModule'][0],
            'AXMN': f['HeaderData']['SystemGeometry']['axialModulesPerSystem'][0],
            'TXCB': f['HeaderData']['SystemGeometry']['radialCrystalsPerBlock'][0],
            'TXBU': f['HeaderData']['SystemGeometry']['radialBlocksPerUnit'][0],
            'TXUM': f['HeaderData']['SystemGeometry']['radialUnitsPerModule'][0],
            'TXMN': f['HeaderData']['SystemGeometry']['radialModulesPerSystem'][0],
            'MRDLM': f['HeaderData']['AcqParameters']['BackEndAcqFilters']['maxRingDiff'][0],
            'tau0': f['HeaderData']['AcqParameters']['EDCATParameters']['negCoincidenceWindow'][0],
            'tau1': f['HeaderData']['AcqParameters']['EDCATParameters']['posCoincidenceWindow'][0],
            'tauP': f['HeaderData']['AcqParameters']['EDCATParameters']['coincTimingPrecision'][0],
            'TOFC': f['HeaderData']['AcqParameters']['RxScanParameters']['tofCompressionFactor'][0],
            'LLD': f['HeaderData']['AcqParameters']['EDCATParameters']['lower_energy_limit'][0],
            'ULD': f['HeaderData']['AcqParameters']['EDCATParameters']['upper_energy_limit'][0],
            'BPELM': bpe}
    return CntH5


def get_nbins(Cnt):
    txUno = Cnt['TXUM'] * Cnt['TXMN']
    txCU =  Cnt['TXCB'] * Cnt['TXBU']
    cpitch = Cnt['cpitch']
    minValue = np.ceil(2 * (np.arcsin(
        Cnt['TFOV'] / Cnt['Deff']) - np.floor(txUno * np.arcsin(
            Cnt['TFOV'] / Cnt['Deff']) / pi) * pi / txUno) / cpitch)
    if txCU < minValue:
        minValue = txCU
    halfFanLORs = np.floor(txUno * np.arcsin(
        Cnt['TFOV'] / Cnt['Deff']) / pi) * txCU + minValue
    W = 2 * int(halfFanLORs) + 2 * Cnt['exLOR'] + 1
    C = Cnt['TXCB'] * Cnt['TXBU'] * Cnt['TXUM'] * Cnt['TXMN'] * Cnt['AXCB']
    if W > C:
        W = C
    return W


# ===================================================================================
# SCANNER CONSTANTS
def get_sig_constants(pthfn):


    Cnt = resources.get_sig_constants()

    if not os.path.isfile(pthfn):
        print('e> could not open the file HDF5 to get SIGNA constants')
        return

    Cnth5 = constants_h5(pthfn)

    Cnt.update(Cnth5)

    # > check angles, bins and sinograms
    # number of sinogram angles
    NSBINS = get_nbins(Cnt)
    # number of transaxial crystals
    NCRS = Cnt['TXCB'] * Cnt['TXBU'] * Cnt['TXUM'] * Cnt['TXMN']
    # number of rings
    NRNG = Cnt['AXCB'] * Cnt['AXBU'] * Cnt['AXUM'] * Cnt['AXMN']

    if (Cnt['BPELM']!=Cnt['BPE']) or (NRNG!=Cnt['NRNG']) or (NCRS!=Cnt['NCRS']) or (NSBINS!=Cnt['NSBINS']):
        raise ValueError('Inconsistent constants with the list-mode file constants')


    # LM processing
    # integration time of 1 sec
    Cnt['ITIME'] = 1000
    # number of CUDA streams
    Cnt['NSTREAMS'] = 32
    # number of elements per data chunk
    # 2^{28} = 268435456 elements (6Bytes) to make up 1.6GB
    Cnt['ELECHNK'] = (268435456 // Cnt['NSTREAMS'])

    # projection view integration time (length of the short time frames t = 2^VTIME)
    Cnt['VTIME'] = 2
    # the short time frame projection views are limited to 90 mins only
    Cnt['MXNITAG'] = 5400

    # transaxial and axial crystal sizes in cm
    Cnt['TXCRS'] = 0.395
    Cnt['AXCRS'] = 0.530

    Cnt['TXUNT'] = 6.45
    # gap between axial detector blocks
    #Cnt['AXBGAP'] = 0.280
    # axial FOV
    Cnt['AXFOV'] = (Cnt['AXUM'] - 1) * Cnt['AXBGAP'] + Cnt['AXUM'] * Cnt['AXCB'] * Cnt['AXCRS']

    Cnt['NAW'] = Cnt['NSBINS']*Cnt['NSANGLES']

    return Cnt


# ===================================================================================
# AXIAL LUTS

def get_segments(Cnt):
    '''
    form the structure of the sinogram segments (axially);
    outputs an array of rows corresponding to all segments (45)
    and columns to the global sinogram index and number of 
    sinograms for each segment (positive and negative).
    '''

    sid = 0
    seg = []
    for k in range(Cnt['NRNG'],0,-2): 
        if k==Cnt['NRNG']:
            sid+=2*k-1
            seg.append([0, sid])
        else:
            seg.append([sid, 2*k-1])
            sid+=2*k-1

            seg.append([sid, 2*k-1])
            sid+=2*k-1

    return np.array(seg)



def get_axLUT(Cnt):

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
        NRNG_c = Cnt['NRNG']
        NSN1_c = Cnt['NSN1']
        if Cnt['RNG_END'] != Cnt['NRNG'] or Cnt['RNG_STRT'] != 0:
            log.error('the reduced axial FOV only works in span-1!')
            return None


    #--------------------------------------------------------
    # > ring positioning/dimensions
    rng = np.zeros((Cnt['NRNG'], 2), dtype=np.float32)

    # > start counting the rings from the front of the scanner
    z = -.5 * Cnt['AXFOV']
    ir = -1#NRNG_c

    for axu in range(Cnt['AXUM']):
        for axc in range(Cnt['AXCB']):
            #ir -= 1
            ir += 1
            rng[ir, 0] = z
            rng[ir, 1] = z + Cnt['AXCRS']
            if Cnt['LOG']<=10:
                print(f'ring {ir}: z0 = {z}, z1 = {rng[ir, 1]}')
            z = rng[ir, 1]
        if Cnt['LOG']<=10:
            print('===========G A P=============')
        z += Cnt['AXBGAP']
    #--------------------------------------------------------




    # get the sino LUTs
    M = np.zeros((NRNG_c, NRNG_c), dtype=np.int16)
    # sino index
    Msn = np.zeros((NRNG_c, NRNG_c), dtype=np.int16)
    # sino index SIGNA native
    Msig = np.zeros((NRNG_c, NRNG_c), dtype=np.int16)

    sn_rno = np.zeros((NRNG_c**2, 2), dtype=np.int16)

    # diagonal linear index (positive only)
    dli = 0
    # full diagonal linear index (positive and negative)
    sni = 0

    for ro in range(0, NRNG_c):
        if ro == 0:
            oblique = 1
        else:
            oblique = 2
        for m in range(oblique):
            # selects the sub-michelogram of the whole michelogram
            strt = Cnt['NRNG'] * (ro + Cnt['RNG_STRT']) + Cnt['RNG_STRT']
            stop = (Cnt['RNG_STRT'] + NRNG_c) * Cnt['NRNG']
            step = Cnt['NRNG'] + 1

            for li in range(strt, stop, step):
                # goes along a diagonal started in the first row at r2o
                # from the linear indices of michelogram get the subscript indices
                if m == 0:
                    r0 = li // Cnt['NRNG']
                    r1 = li - r0 * Cnt['NRNG']
                    M[r1, r0] = dli
                    dli += 1
                else:
                    r1 = li // Cnt['NRNG']
                    r0 = li - r1 * Cnt['NRNG']

                Msn[r1, r0] = sni

                sn_rno[sni, 0] = r0
                sn_rno[sni, 1] = r1

                sni += 1

                # --- SIGNA native ---
                rdiff = r0 - r1
                rsum = r0 + r1
                if (rdiff > 1):
                    angle = rdiff // 2
                    if (angle <= Cnt['MRD'] / 2):
                        snis = rsum + (4*angle - 2) * Cnt['NRNG'] - (4*angle*angle - 1)
                elif (rdiff < -1):
                    angle = -rdiff // 2
                    if (angle <= Cnt['MRD'] // 2):
                        snis = rsum + (4*angle) * Cnt['NRNG'] - ((angle+1) * 4 * angle)
                else:
                    snis = rsum
                Msig[r1, r0] = snis
                # ------

    #---------------------------------------------------------------------
    # > Michelogram of number of span-1 sinograms mashed per sinogram (half) span-2 index
    Mnos = -1 * np.ones((NRNG_c, NRNG_c), dtype=np.int8)
    uq = np.unique(Msig)
    for r0 in range(0, NRNG_c):
        for r1 in range(0, NRNG_c):
            sni = Msig[r0,r1]
            Mnos[r0,r1] = np.sum(Msig==sni)
    #---------------------------------------------------------------------


    # ---------------------------------------------------------------------
    # > linear index (along diagonals of Michelogram) to rings
    # > the number of Michelogram elements considered in projection calculations
    NLI2R_c = (NRNG_c**2-NRNG_c)//2 + NRNG_c #int(NRNG_c**2/2 + NRNG_c/2)

    li2r = np.zeros((NLI2R_c, 2), dtype=np.int8)

    # > the same as above but to sinograms in span-1 and span-2
    # > (it is half span-2 as only the cross sinograms are mashed)
    li2sn = np.zeros((NLI2R_c, 2), dtype=np.int16)
    li2sn1 = np.zeros((NLI2R_c, 2), dtype=np.int16)
    li2rng = np.zeros((NLI2R_c, 2), dtype=np.float32)
    
    # > and now LUTs to number of sinograms (nos)
    # > only for the half span-2
    li2nos = np.zeros((NLI2R_c), dtype=np.int8)

    dli = 0
    for ro in range(0, NRNG_c):
        # selects the sub-Michelogram of the whole Michelogram
        strt = Cnt['NRNG'] * (ro + Cnt['RNG_STRT']) + Cnt['RNG_STRT']
        stop = (Cnt['RNG_STRT'] + NRNG_c) * Cnt['NRNG']
        step = Cnt['NRNG'] + 1

        # goes along a diagonal started in the first row at r2o
        for li in range(strt, stop, step):
            # from the linear indexes of Michelogram get the subscript indexes
            r0 = int(li / Cnt['NRNG'])
            r1 = int(li - r0*Cnt['NRNG'])
            if Msn[r1, r0] < 0:
                # avoid case when RD>MRD
                continue

            li2r[dli, 0] = r0
            li2r[dli, 1] = r1

            # --
            li2rng[dli, 0] = rng[r0, 0]
            li2rng[dli, 1] = rng[r1, 0]
            # --

            li2sn[dli, 0] = Msig[r0, r1]
            li2sn[dli, 1] = Msig[r1, r0]

            li2sn1[dli, 0] = Msn[r0, r1]
            li2sn1[dli, 1] = Msn[r1, r0]

            # --
            li2nos[dli] = Mnos[r1, r0]
            # --

            dli += 1
    # log.info('number of diagonal indexes (in Michelogram) accounted for: {}'.format(dli))
    # ---------------------------------------------------------------------


    seg = get_segments(Cnt)

    return {'r2s': Msn, 'r2sig': Msig, 's2r': sn_rno, 'rng': rng, 'seg': seg,
             'li2rno': li2r, 'li2sn': li2sn, 'li2sn1': li2sn1, 'li2nos': li2nos, 'li2rng': li2rng}
    



# ===================================================================================
# TRANSIAXIAL LUTS
def get_txLUT(Cnt, visualisation=False):

    '''
    Create 2D (transaxial) LUTs.
    Create linear index for the whole sino.
    Angle index of the sino is used as the primary index (fast changing).
    '''

    R = Cnt['R_RING']

    if visualisation:
        # ---visualisation of the crystal ring in transaxial view
        VISSZ = 0.05
        VISXY = int((R+1)*2/VISSZ)
        T = np.zeros((VISXY, VISXY), dtype=np.float32)

    # --- crystal coordinates transaxially
    # > block/module width
    bw = Cnt['TXUNT']

    alpha = 0.2244 # 2*pi/bw
    crs = np.zeros((Cnt['NCRS'], 4), dtype=np.float32)

    # > phi angle points in the middle and is used for obtaining the normal of detector block
    phi = 0.5*pi - 0.0001
    for bi in range(Cnt['TXMN']):
        # > tangent point (ring against detector block)
        y = R * np.sin(phi)
        x = R * np.cos(phi)

        # > vector for the face of crystals
        pv = np.array([-y, x])
        pv /= np.sum(pv**2)**.5

        # > update phi for next block
        phi -= alpha

        # > end block points
        xcp = x + (bw/2) * pv[0]
        ycp = y + (bw/2) * pv[1]

        if visualisation:
            u = int(.5*VISXY + np.floor(xcp/VISSZ))
            v = int(.5*VISXY -  np.ceil(ycp/VISSZ))
            T[v, u] = 3

        NCRSBLK = Cnt['TXCB']*Cnt['TXBU']

        for n in range(NCRSBLK):
            c = bi*NCRSBLK + n
            crs[c, 0] = xcp
            crs[c, 1] = ycp
            xc = x + (bw/2 - (n+1)*bw/NCRSBLK) * pv[0]
            yc = y + (bw/2 - (n+1)*bw/NCRSBLK) * pv[1]
            crs[c, 2] = xc
            crs[c, 3] = yc
            xcp = xc
            ycp = yc

            if visualisation:
                u = int(.5*VISXY + np.floor(xcp/VISSZ))
                v = int(.5*VISXY -  np.ceil(ycp/VISSZ))
                T[v, u] += 2


    #crs = np.roll(crs, -2, axis=0)

    out = {'crs': crs}

    if visualisation:
        out['visual'] = T


    # > TRANSAXIAL SINOGRAM LUTS
    # number of bins per sinogram angle
    NSBINS = Cnt['NSBINS']
    # number of sinogram angles
    NSANGLES = Cnt['NSANGLES']
    # number of rings
    NCRS = Cnt['NCRS']
    # crystal to sinogram index lookup table (LUT)
    c2s = np.zeros((NCRS, NCRS), dtype=np.int32)
    # sinogram to crystal index LUT
    s2c = np.zeros((NSANGLES * NSBINS, 2), dtype=np.int16)

    for c0 in range(NCRS):
        for c1 in range(NCRS):
            
            if ((NCRS // 2) <= (c0+c1)) and ((c0+c1) < (3 * NCRS // 2)):
                iw = (NSBINS-1) // 2 + (c0 - c1 - NCRS//2)
            else:
                iw = (NSBINS-1) // 2 - (c0 - c1 - NCRS//2)
            
            if (iw >= 0) and (iw <= (NSBINS - 1)):
                ia = ((c0 + c1 + NCRS//2) % NCRS) // 2
                aw = ia + NSANGLES*iw
                c2s[c1, c0] = aw
                c2s[c0, c1] = aw
                s2c[aw, 0] = c0
                s2c[aw, 1] = c1
            else:
                c2s[c1, c0] = -1
                c2s[c0, c1] = -1

    out['c2s'] = c2s
    out['s2c'] = s2c
    return out
# ===================================================================================












def init_sig(pthfn):

    # get the constants for the Signa
    Cnt = get_sig_constants(pthfn)

    # transaxial look up tables
    txLUT = get_txLUT(Cnt)

    # axial look up tables
    axLUT = get_axLUT(Cnt)

    return Cnt, txLUT, axLUT




def get_cylinder(Cnt, rad=25, xo=0, yo=0, unival=1, gpu_dim=False, mask=True, two_d=False):
    '''Outputs image with a uniform cylinder of intensity = unival,
        radius = rad, and transaxial centre (xo, yo).
    '''

    if mask:  unival = 1

    imdsk = np.zeros((1, Cnt['SZ_IMX'], Cnt['SZ_IMY']), dtype=np.float32)
    
    for t in np.arange(0, math.pi, math.pi/(2*360)):
        x = xo+rad*math.cos(t)
        y = yo+rad*math.sin(t)
        yf = np.arange(-y+2*yo, y, Cnt['SZ_VOXY']/2)
        v = np.int32(.5*Cnt['SZ_IMX'] - np.ceil(yf/Cnt['SZ_VOXY']))
        u = np.int32(.5*Cnt['SZ_IMY'] + np.floor(x/Cnt['SZ_VOXY']))
        imdsk[0,v,u] = unival
    
    if two_d:
        imdsk = np.squeeze(imdsk)
    else:
        imdsk = np.repeat(imdsk, Cnt['SZ_IMZ'], axis=0)

    if mask: imdsk = imdsk.astype(dtype=bool)
    
    if gpu_dim and not two_d:
        return np.transpose(imdsk, (1, 2, 0))
    else:
        return imdsk

















'''
def transaxial_lut(Cnt, visualisation=False):

    if visualisation:
        # ---visualisation of the crystal ring in transaxial view
        p = 8      # pixel density of the visualisation
        VISXY = 320 * p
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
'''