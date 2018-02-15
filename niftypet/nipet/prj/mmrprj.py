"""Forward and back projector for PET data reconstruction"""
__author__      = "Pawel Markiewicz"
__copyright__   = "Copyright 2018"
#------------------------------------------------------------------------------
import numpy as np
import sys
import os

import petprj

from niftypet.nipet.img import mmrimg
from niftypet.nipet import mmraux

#=========================================================================
# forward projector
#-------------------------------------------------------------------------
def frwd_prj(im, txLUT, axLUT, Cnt, isub=np.array([-1], dtype=np.int32)):

    if Cnt['SPN']==1:
        # number of rings calculated for the given ring range (optionally we can use only part of the axial FOV)
        NRNG_c = Cnt['RNG_END'] - Cnt['RNG_STRT']
        # number of sinos in span-1
        nsinos = NRNG_c**2
        # correct for the max. ring difference in the full axial extent (don't use ring range (1,63) as for this case no correction)
        if NRNG_c==64:
            nsinos -= 12

    elif  Cnt['SPN']==11: nsinos=Cnt['NSN11']
    elif  Cnt['SPN']==0:  nsinos=Cnt['NSEG0']

    if im.shape[0]==Cnt['SO_IMZ'] and im.shape[1]==Cnt['SO_IMY'] and im.shape[2]==Cnt['SO_IMX']:
        ims = mmrimg.convert2dev(im, Cnt)
    elif im.shape[0]==Cnt['SZ_IMX'] and im.shape[1]==Cnt['SZ_IMY'] and im.shape[2]==Cnt['SZ_IMZ']:
        ims = im
    else:
        print 'e> wrong image size;  it has to be one of these: (z,y,x) = (127,344,344) or (y,x,z) = (320,320,128)'

    print 'i> number of sinos:', nsinos
    
    #predefine the sinogram.  if subsets are used then only preallocate those bins which will be used.
    if isub[0]<0:
        sinog = np.zeros((txLUT['Naw'], nsinos), dtype=np.float32)
    else:
        sinog = np.zeros((len(isub), nsinos), dtype=np.float32)
    
    # --------------------
    petprj.fprj(sinog, ims, txLUT, axLUT, isub, Cnt, 0)
    # --------------------
    # get the sinogram bins in proper sinogram
    sino = np.zeros((txLUT['Naw'], nsinos), dtype=np.float32)
    if isub[0]>=0:    sino[isub,:] = sinog
    else:  sino = sinog 
    # put the gaps back to form displayable sinogram
    sino = mmraux.putgaps(sino, txLUT, Cnt)
    return sino

#=========================================================================
# attenuation factors by forward projecting the mu-map
#-------------------------------------------------------------------------
def att_prj(mu, txLUT, axLUT, Cnt):

    if    Cnt['SPN']==1:  nsinos=Cnt['NSN1']
    elif  Cnt['SPN']==11: nsinos=Cnt['NSN11']
    elif  Cnt['SPN']==0:  nsinos=Cnt['NSEG0']

    if mu.shape[0]==Cnt['SO_IMZ'] and mu.shape[1]==Cnt['SO_IMY'] and mu.shape[2]==Cnt['SO_IMX']:
        ims = mmrimg.convert2dev(mu, Cnt)
    elif mu.shape[0]==Cnt['SZ_IMX'] and mu.shape[1]==Cnt['SZ_IMY'] and mu.shape[2]==Cnt['SZ_IMZ']:
        ims = mu
    else:
        print 'e> wrong image size;  it has to be one of these: (z,y,x) = (127,344,344) or (y,x,z) = (320,320,128)'

    #predefine the sinogram
    sinog = np.zeros((txLUT['Naw'], nsinos), dtype=np.float32)
    petprj.fprj(sinog, ims, txLUT, axLUT, np.array([-1], dtype=np.int32), Cnt, 1)
    sino = mmraux.putgaps(sinog, txLUT, Cnt)
    return sino

#=========================================================================
# back projector
#-------------------------------------------------------------------------
def back_prj(sino, txLUT, axLUT, Cnt, isub=np.array([-1], dtype=np.int32)):

    if    Cnt['SPN']==1:  nsinos=Cnt['NSN1']
    elif  Cnt['SPN']==11: nsinos=Cnt['NSN11']
    elif  Cnt['SPN']==0:  nsinos=Cnt['NSEG0']

    if len(sino.shape)==3:
        if sino.shape[0]!=nsinos or sino.shape[1]!=Cnt['NSANGLES'] or sino.shape[2]!=Cnt['NSBINS']:
            print 'e> problem with input sino dimensions.'
            return None
        sinog = mmraux.remgaps(sino, txLUT, Cnt)

    elif len(sino.shape)==2:
        if sino.shape[0]!=txLUT["Naw"] or sino.shape[1]!=nsinos:
            print 'e> problem with input sino dimensions.'
            return None
        sinog = sino

    else:
        print 'e> wrong input sino dimensions.'
        return None
    
    if isub[0]>=0:
        sinog = sinog[isub,:]

    #predefine the output image
    bimg = np.zeros((Cnt['SZ_IMX'], Cnt['SZ_IMY'], Cnt['SZ_IMZ']), dtype=np.float32)
    #run the backprojection algorithm
    petprj.bprj(bimg, sinog, txLUT, axLUT, isub, Cnt)
    #change from GPU optimised image dimensions to standard dimensions
    bimg = mmrimg.convert2e7(bimg, Cnt)
    return bimg
#-------------------------------------------------------------------------
