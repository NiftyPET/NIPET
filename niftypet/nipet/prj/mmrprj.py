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
def frwd_prj(im, scanner_params, isub=np.array([-1], dtype=np.int32), dev_out=False, attenuation=False):
    ''' Calculate forward projection (a set of sinograms) for the provided input image.
        Arguments:
        im -- input image (can be emission or mu-map image).
        scanner_params -- dictionary of all scanner parameters, containing scanner constants,
            transaxial and axial look up tables (LUT).
        isub -- array of transaxial indices of all sinograms (angles x bins) used for subsets.
            when the first element is negative, all transaxial bins are used (as in pure EM-ML).
        dev_out -- if True, output sinogram is in the device form, i.e., with two dimensions
            (# bins/angles, # sinograms) instead of default three (# sinograms, # bins, # angles).
        attenuation -- controls whether emission or LOR attenuation probability sinogram
            is calculated; the default is False, meaning emission sinogram; for attenuation
            calculations (attenuation=True), the exponential of the negative of the integrated
            mu-values along LOR path is taken at the end.
    '''

    # Get particular scanner parameters: Constants, transaxial and axial LUTs
    Cnt   = scanner_params['Cnt']
    txLUT = scanner_params['txLUT']
    axLUT = scanner_params['axLUT']

    #>choose between attenuation forward projection (mu-map is the input)
    #>or the default for emission image forward projection
    if attenuation:
        att = 1
    else:
        att = 0

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
    elif im.shape[0]==Cnt['rSO_IMZ'] and im.shape[1]==Cnt['SO_IMY'] and im.shape[2]==Cnt['SO_IMX']:
        ims = mmrimg.convert2dev(im, Cnt)
    elif im.shape[0]==Cnt['SZ_IMX'] and im.shape[1]==Cnt['SZ_IMY'] and im.shape[2]==Cnt['rSZ_IMZ']:
        ims = im
    else:
        print 'e> wrong image size;  it has to be one of these: (z,y,x) = (127,344,344) or (y,x,z) = (320,320,128)'

    if Cnt['VERBOSE']: print 'i> number of sinos:', nsinos
    
    #predefine the sinogram.  if subsets are used then only preallocate those bins which will be used.
    if isub[0]<0:
        sinog = np.zeros((txLUT['Naw'], nsinos), dtype=np.float32)
    else:
        sinog = np.zeros((len(isub), nsinos), dtype=np.float32)
    
    # --------------------
    petprj.fprj(sinog, ims, txLUT, axLUT, isub, Cnt, att)
    # --------------------
    # get the sinogram bins in a proper sinogram
    sino = np.zeros((txLUT['Naw'], nsinos), dtype=np.float32)
    if isub[0]>=0:    sino[isub,:] = sinog
    else:  sino = sinog

    # put the gaps back to form displayable sinogram
    if not dev_out:
        sino = mmraux.putgaps(sino, txLUT, Cnt)
    
    return sino

#=========================================================================
# back projector
#-------------------------------------------------------------------------
def back_prj(sino, scanner_params, isub=np.array([-1], dtype=np.int32)):
    ''' Calculate forward projection for the provided input image.
        Arguments:
        sino -- input emission sinogram to be back projected to the image space.
        scanner_params -- dictionary of all scanner parameters, containing scanner constants,
            transaxial and axial look up tables (LUT).
        isub -- array of transaxial indices of all sinograms (angles x bins) used for subsets;
            when the first element is negative, all transaxial bins are used (as in pure EM-ML).
    '''

    # Get particular scanner parameters: Constants, transaxial and axial LUTs
    Cnt   = scanner_params['Cnt']
    txLUT = scanner_params['txLUT']
    axLUT = scanner_params['axLUT']

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


    #> check first the Siemens default sinogram;
    #> for this default shape only full sinograms are expected--no subsets.
    if len(sino.shape)==3:
        if sino.shape[0]!=nsinos or sino.shape[1]!=Cnt['NSANGLES'] or sino.shape[2]!=Cnt['NSBINS']:
            raise ValueError('Unexpected sinogram array dimensions/shape for Siemens defaults.')
        sinog = mmraux.remgaps(sino, txLUT, Cnt)

    elif len(sino.shape)==2:
        if isub[0]<0 and sino.shape[0]!=txLUT["Naw"]:
            raise ValueError('Unexpected number of transaxial elements in the full sinogram.')
        elif isub[0]>=0 and sino.shape[0]!=len(isub):
            raise ValueError('Unexpected number of transaxial elements in the subset sinogram.')
        #> check if the number of sinograms is correct
        if sino.shape[1]!=nsinos:
            raise ValueError('Inconsistent number of sinograms in the array.')
        #> when found the dimensions/shape are fine:
        sinog = sino
    else:
        raise ValueError('Unexpected shape of the input sinogram.')

    #predefine the output image depending on the number of rings used
    if Cnt['SPN']==1 and 'rSZ_IMZ' in Cnt:
        nvz = Cnt['rSZ_IMZ']
    else:
        nvz = Cnt['SZ_IMZ']
    bimg = np.zeros((Cnt['SZ_IMX'], Cnt['SZ_IMY'], nvz), dtype=np.float32)

    #> run back-projection
    petprj.bprj(bimg, sinog, txLUT, axLUT, isub, Cnt)

    #> change from GPU optimised image dimensions to the standard Siemens shape
    bimg = mmrimg.convert2e7(bimg, Cnt)

    return bimg
#-------------------------------------------------------------------------