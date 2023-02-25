"""Resources file for NiftyPET to be used with SynchroPET"""
#---------------------------------------------------------------
__author__      = "Pawel Markiewicz"
__copyright__   = "Copyright 2023"
#---------------------------------------------------------------
import numpy as np
from math import pi

from . import resources



# > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > 
# S C A N N E R   L O O K - U P   T A B L E S
# > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > 

# >---------------------------
# > TRANSAXIAL
# >---------------------------

def transaxial_lut(Cnt, visualisation=False):
    '''Generate look-up tables (LUT) for the transaxial part of the scanner. 
    '''
    #----------------------------------
    # crystal definitions transaxially
    #----------------------------------
    
    if visualisation:
        #---visualisation of the crystal ring in transaxial view
        p = 20. #pixel density of the visualisation
        VISXY = int(int(np.ceil(Cnt['R_RING']*2)+1)*p);
        T = np.zeros((VISXY,VISXY), dtype=np.float32)
        #---

    # > initialise crystal dimensions array
    crs = np.zeros((Cnt['NCRS'], 4), dtype=np.float32)
    
    # > initialise the angle
    phi = 0.5*pi - 0.001 #+ 0.5*Cnt['ALPHA']a

     
    for bi in range(Cnt['NTXBLK']):
        # tangent point (ring against detector block)
        y  =  Cnt['R_RING']*np.sin(phi)
        x  =  Cnt['R_RING']*np.cos(phi)
        # generating vector for the crystal faces in a block
        pv  = np.array([-y, x])
        pv /= np.sum(pv**2)**.5
        # update phi for next block
        phi -= Cnt['ALPHA'] 
        # end block points
        xcp = x + .5*Cnt['BLKWDTH']*pv[0]
        ycp = y + .5*Cnt['BLKWDTH']*pv[1]
        # ---
        if visualisation:
            u = int(.5*VISXY + np.floor(xcp*p))
            v = int(.5*VISXY - np.ceil (ycp*p))
            T[v,u] = 5
        # ---
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
            # ---
            if visualisation:
                u = int(.5*VISXY + np.floor(xcp*p))
                v = int(.5*VISXY - np.ceil (ycp*p))
                T[v,u] = 2.5
            # ---
    

    dctout = {'crs':crs}
    if visualisation:
        dctout['visual'] = T
    


    #----------------------------------
    # sinogram definitions
    # LUT: sino -> crystal and crystal -> sino
    s2c = np.zeros((Cnt['NSBINS']*Cnt['NSANGLES'], 2), dtype=np.int16)
    c2s = -1*np.ones((Cnt['NCRS'], Cnt['NCRS']), dtype=np.int32)
    
    #> with projection bin <w> fast changing (opposite to angle changing fast).
    #> this one used in scatter estimation
    c2sw = -1*np.ones((Cnt['NCRS'], Cnt['NCRS']), dtype=np.int32)

    awi = 0
    for iw in range(Cnt['NSBINS']):
        for ia in range(Cnt['NSANGLES']):
            c0 = int( np.floor( (ia + 0.5*(Cnt['NCRS'] - 2 + Cnt['NSBINS']/2 - iw))   % Cnt['NCRS'] ) )
            c1 = int( np.floor( (ia + 0.5*(2*Cnt['NCRS'] - 2 - Cnt['NSBINS']/2 + iw)) % Cnt['NCRS'] ) )

            s2c[awi,0] = c0
            s2c[awi,1] = c1

            c2s[c1, c0] = ia + iw*Cnt['NSANGLES']
            c2s[c0, c1] = ia + iw*Cnt['NSANGLES']

            c2sw[c1, c0] = iw + ia*Cnt['NSBINS']
            c2sw[c0, c1] = iw + ia*Cnt['NSBINS']

            awi += 1
    #----------------------------------


    dctout['s2cF'] = s2c
    dctout['c2s'] = c2s
    dctout['c2sw'] = c2sw
    dctout['Naw'] = Cnt['NSBINS']*Cnt['NSANGLES']

    return dctout



# >---------------------------
# > AXIAL
# >---------------------------

def axial_lut(Cnt, printout=False):
    '''Creates lookup tables (LUT) for linear indexes along the diagonals of Michelogram
    '''
    # > number of axial detector rings
    NRNG = Cnt['NRNG']

    # > ring dimensions
    rng = np.zeros((NRNG,2), dtype = np.float32)

    z = -0.5*( Cnt['NAXBLK']*Cnt['BLKHGHT'])
    for i in range(Cnt['NRNG']):
        rng[i,0] = z
        z += Cnt['AXR']
        rng[i,1] = z
        if printout:
            print('i={}, z=({},{})'.format(i, rng[i,0], rng[i,1]))

    # number of sinograms in span-1
    NSN = NRNG**2

    #---------------------------------------------------------------------
    # Michelogram for single slice rebinning (absolute axial position for individual sinos) 
    Mssrb = -1*np.ones((NRNG,NRNG), dtype=np.int32)
    for r1 in range(NRNG):
        for r0 in range(NRNG):
            ssp = r0+r1  #segment sino position
            Mssrb[r1,r0] = ssp 

    #---------------------------------------------------------------------
    # Michelogram for span-1 sino
    Msn = -1*np.ones((NRNG,NRNG), dtype=np.int16)
    # sino index -> ring index
    sn_rno = np.zeros((NSN,2), dtype=np.int16)
    sn_ssrb= np.zeros((NSN), dtype=np.int16)
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
                    r0 = int(li/NRNG)
                    r1 = int(li - r0*NRNG)
                else: 
                    #for positive now (? or vice versa)
                    r1 = int(li/NRNG)
                    r0 = int(li - r1*NRNG)
                sn_rno[sni,0] = r0
                sn_rno[sni,1] = r1
                sn_ssrb[sni] = Mssrb[r1,r0]
                Msn[r1,r0] = sni
                #--
                sni += 1

    # ring numbers for span-1 sino index to SSRB
    sn_ssrno = np.zeros(Cnt['NSEG0'], dtype=np.int8)
    for i in range(NSN):
        sn_ssrno[sn_ssrb[i]] += 1
    sn_ssrno  =  sn_ssrno[np.unique(sn_ssrb)]
 

    #---------------------------------------------------------------------
    #linear index (along diagonals of Michelogram) to rings
    NLI2R = int(NRNG**2/2. + NRNG/2.)
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
            r0 = int(li/NRNG)
            r1 = int(li - r0*NRNG)

            li2r[dli,0] = r0
            li2r[dli,1] = r1
            #--            
            li2rng[dli,0] = rng[r1,0]
            li2rng[dli,1] = rng[r0,0]
            #-- 
            li2sn[dli, 0] = Msn[r1,r0]
            li2sn[dli, 1] = Msn[r0,r1]

            dli += 1


    li2nos = np.ones((NLI2R), dtype=np.int8)

    return {'rng':rng, 'Msn':Msn, 
            'li2nos':li2nos, 'li2rno':li2r, 'li2sn1':li2sn, 'li2rng':li2rng, 
            'sn1_rno':sn_rno, 'sn1_ssrb':sn_ssrb, 'sn1_ssrno':sn_ssrno}


def init_synchropet():

    # get the constants for the Signa
    Cnt = resources.get_synchropet_constants()

    # transaxial look up tables
    txLUT = transaxial_lut(Cnt)

    # axial look up tables
    axLUT = axial_lut(Cnt)

    return Cnt, txLUT, axLUT

def get_synpars():

    Cnt, txLUT, axLUT = init_synchropet()

    return dict(Cnt=Cnt, txLUT=txLUT, axLUT=axLUT)