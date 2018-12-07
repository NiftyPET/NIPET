"""Simulations for image reconstruction with recommended reduced axial field of view"""
__author__      = "Pawel Markiewicz"
__copyright__   = "Copyright 2018, University College London"

import numpy as np

from niftypet import nimpa

import mmrprj
from niftypet.nipet import mmraux


def simulate_sino(petim, ctim, scanner_params, slice_idx=-1):
    ''' Simulate the measured sinogram with photon attenuation.
        Arguments:
        petim -- the input PET image based on which the emission sinogram is found
        ctim -- CT image, in register with PET and the same dimensions, is used 
            for estimating the attenuation factors, which are then applied to 
            simulate emission sinogram with realistic photon attenuation. 
        slice_idx -- chosen 2D slice out of the 3D image for the fast simulation.
        scanner_params -- scanner parameters containing scanner constants and
            axial and transaxial look up tables (LUTs)
    '''

    #> decompose the scanner constants and LUTs for easier access
    Cnt = scanner_params['Cnt']
    txLUT = scanner_params['txLUT']
    axLUT = scanner_params['axLUT']

    if petim.shape != ctim.shape:
        raise ValueError('The shapes of the PET and CT images are inconsistent.')


    if len(petim.shape)==3:
        
        # make sure that the shape of the input image matches the image size of the scanner
        if petim.shape[1:]!=(Cnt['SO_IMY'], Cnt['SO_IMX']):
            raise ValueError('The input image shape for x and y does not match the scanner image size.')

        # pick the right slice index (slice_idx) if not given or mistaken
        if slice_idx<0:
            print 'w> the axial index <slice_idx> is chosen to be in the middle of axial FOV.'
            slice_idx = petim.shape[0]/2
        if slice_idx>=petim.shape[0]:
            raise ValueError('The axial index for 2D slice selection is outside the image.')

    elif len(petim.shape)==2:
        
        # make sure that the shape of the input image matches the image size of the scanner
        if petim.shape != (Cnt['SO_IMY'], Cnt['SO_IMX']):
            raise ValueError('The input image shape for x and y does not match the scanner image size.')

        petim.shape = (1,) + petim.shape
        ctim.shape  = (1,) + ctim.shape
        slice_idx = 0



    if not 'rSZ_IMZ' in Cnt:
        raise ValueError('Missing reduced axial FOV parameters.')

    #--------------------
    #> get the mu-map from CT
    mui = nimpa.ct2mu(ctim)
    mui[mui<0] = 0
    #--------------------

    #--------------------
    #> create a number of slides of the same chosen image slice for reduced (fast) 3D simulation
    rmu = mui[slice_idx,:,:]
    rmu.shape = (1,) + rmu.shape
    rmu = np.repeat(rmu, Cnt['rSZ_IMZ'], axis=0)
    #--------------------

    #--------------------
    #> form a short 3D image of the same emission image slice
    rpet = petim[slice_idx,:,:].copy()
    rpet.shape = (1,) + rpet.shape
    rpet = np.repeat(rpet, Cnt['rSZ_IMZ'], axis=0)
    #--------------------

    #> forward project the mu-map to obtain attenuation factors
    attsino = mmrprj.frwd_prj(rmu,  scanner_params, attenuation=True)

    #> forward project the PET image to obtain non-attenuated emission sino
    emisino = mmrprj.frwd_prj(rpet, scanner_params, attenuation=False)

    #> return the simulated emission sino with photon attenuation 
    return attsino*emisino



def simulate_recon(
    measured_sino,
    ctim,
    scanner_params,
    nitr = 60,
    slice_idx = -1,
    randoms=None):

    ''' Reconstruct PET image from simulated input data using the EM-ML algorithm.
        Arguments:
        measured_sino -- simulated emission data with photon attenuation
        ctim -- either a 2D CT image or a 3D CT image from which a 2D slice is chosen (slice_idx) for estimation
            of the attenuation factors
        slice_idx -- index to extract one 2D slice for this simulation if input image is 3D
        nitr -- number of iterations used for the EM-ML reconstruction algorithm
        scanner_params -- scanner parameters containing scanner constants and
            axial and transaxial look up tables (LUTs)
        randoms[=None] -- possibility of using randoms and scatter events in the simulation  
    '''

    #> decompose the scanner constants and LUTs for easier access
    Cnt = scanner_params['Cnt']
    txLUT = scanner_params['txLUT']
    axLUT = scanner_params['axLUT']

    if len(ctim.shape)==3:
        
        # make sure that the shape of the input image matches the image size of the scanner
        if ctim.shape[1:]!=(Cnt['SO_IMY'], Cnt['SO_IMX']):
            raise ValueError('The input image shape for x and y does not match the scanner image size.')

        # pick the right slice index (slice_idx) if not given or mistaken
        if slice_idx<0:
            print 'w> the axial index <slice_idx> is chosen to be in the middle of axial FOV.'
            slice_idx = ctim.shape[0]/2
        if slice_idx>=ctim.shape[0]:
            raise ValueError('The axial index for 2D slice selection is outside the image.')

    elif len(ctim.shape)==2:
        
        # make sure that the shape of the input image matches the image size of the scanner
        if ctim.shape != (Cnt['SO_IMY'], Cnt['SO_IMX']):
            raise ValueError('The input image shape for x and y does not match the scanner image size.')

        ctim.shape  = (1,) + ctim.shape
        slice_idx = 0

        

    if not 'rSZ_IMZ' in Cnt:
        raise ValueError('Missing reduced axial FOV parameters.')

    #--------------------
    #> get the mu-map from CT
    mui = nimpa.ct2mu(ctim)
    mui[mui<0] = 0
    #--------------------

    #--------------------
    #> create a number of slides of the same chosen image slice for reduced (fast) 3D simulation
    rmu = mui[slice_idx,:,:]
    rmu.shape = (1,) + rmu.shape
    rmu = np.repeat(rmu, Cnt['rSZ_IMZ'], axis=0)
    #--------------------


    #--------------------
    #> get the input mu-map from CT
    mui = nimpa.ct2mu(ctim)
    mui[mui<0] = 0
    #> create a number of slides of the same chosen image slice for reduced (fast) 3D simulation
    rmu = mui[slice_idx,:,:]
    rmu.shape = (1,) + rmu.shape
    rmu = np.repeat(rmu, Cnt['rSZ_IMZ'], axis=0)
    #--------------------


    #> attenuation factor sinogram
    attsino = mmrprj.frwd_prj(rmu,  scanner_params, attenuation=True, dev_out=True)

    #> randoms and scatter put together
    if randoms==None:
        rndsct = np.zeros((Cnt['Naw'], Cnt['rNSN1']), dtype=np.float32)
    else:
        rndsct = randoms

    #> sensitivity image for the EM-ML reconstruction
    sim = mmrprj.back_prj(attsino, scanner_params)

    #> estimated image, initialised to ones
    eim = np.ones(rmu.shape, dtype=np.float32)

    for i in range(nitr):
        print '>---- EM iteration:', i
        #> remove gaps from the measured sinogram
        #> then forward project the estimated image
        #> after which divide the measured sinogram by the estimated sinogram (forward projected)
        crrsino = mmraux.remgaps(measured_sino, txLUT, Cnt) / \
                    (mmrprj.frwd_prj(eim, scanner_params, dev_out=True) + rndsct)

        #> back project the correction factors sinogram
        bim = mmrprj.back_prj(crrsino, scanner_params) 

        #> divide the back-projected image by the sensitivity image
        msk = sim>0
        bim[msk] /= sim[msk]
        bim[~msk] = 0

        #> update the estimated image and remove NaNs 
        eim *= bim
        eim[np.isnan(eim)] = 0

    return eim