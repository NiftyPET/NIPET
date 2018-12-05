"""Image functions for PET data reconstruction and processing."""
__author__      = "Pawel J. Markiewicz"
__copyright__   = "Copyright 2018"
#-------------------------------------------------------------------------------

import sys
import os

import numpy as np
import math
from math import pi
import random
import scipy.ndimage as ndi

import nibabel as nib
import pydicom as dcm
import re
import glob

from subprocess import call
import time
import multiprocessing

from niftypet.nipet import mmraux
from niftypet import nimpa


#===================================================================================
# IMAGE ROUTINES
#===================================================================================

def convert2e7(img, Cnt):
    '''Convert GPU optimised image to Siemens/E7 image shape (127,344,344).'''

    margin = (Cnt['SO_IMX']-Cnt['SZ_IMX'])/2

    #permute the dims first
    imo = np.transpose(img, (2,0,1))

    nvz = img.shape[2]

    #> get the x-axis filler and apply it
    filler = np.zeros((nvz, Cnt['SZ_IMY'], margin), dtype=np.float32)
    imo = np.concatenate((filler, imo, filler), axis=2)

    #> get the y-axis filler and apply it
    filler = np.zeros((nvz, margin, Cnt['SO_IMX']), dtype=np.float32)
    imo = np.concatenate((filler, imo, filler), axis=1)
    return imo

def convert2dev(im, Cnt):
    '''Reshape Siemens/E7 (default) image for optimal GPU execution.'''

    if im.shape[1]!=Cnt['SO_IMY'] or im.shape[2]!=Cnt['SO_IMX']:
        raise ValueError('e> input image array is not of the correct Siemens shape.')
    
    if 'rSZ_IMZ' in Cnt and im.shape[0]!=Cnt['rSZ_IMZ']:
        print 'w> the axial number of voxels does not match the reduced rings.'
    elif not 'rSZ_IMZ' in Cnt and im.shape[0]!=Cnt['SZ_IMZ']:
        print 'w> the axial number of voxels does not match the rings.'

    im_sqzd = np.zeros((im.shape[0], Cnt['SZ_IMY'], Cnt['SZ_IMX']), dtype=np.float32)
    margin = (Cnt['SO_IMX']-Cnt['SZ_IMX'])/2
    margin_=-margin
    if margin==0: 
        margin = None
        margin_= None

    im_sqzd = im[:, margin:margin_, margin:margin_]
    im_sqzd = np.transpose(im_sqzd, (1, 2, 0))
    return im_sqzd

def cropxy(im, imsize, datain, Cnt, store_pth=''):
    '''Crop image transaxially to the size in tuple <imsize>.  
    Return the image and the affine matrix.
    '''
    if not imsize[0]%2==0 and not imsize[1]%2==0:
        print 'e> image size has to be an even number!'
        return None

    # cropping indexes
    i0 = (Cnt['SO_IMX']-imsize[0])/2
    i1 = (Cnt['SO_IMY']+imsize[1])/2

    B = image_affine(datain, Cnt, gantry_offset=False)
    B[0,3] -= 10*Cnt['SO_VXX']*i0
    B[1,3] += 10*Cnt['SO_VXY']*(Cnt['SO_IMY']-i1)

    cim = im[:, i0:i1, i0:i1]

    if store_pth!='':
        nimpa.array2nii( cim[::-1,::-1,:], B, store_pth, descrip='cropped')
        if Cnt['VERBOSE']:  print 'i> saved cropped image to:', store_pth

    return cim, B
#-------------------------------------------------------------------------------------------

def image_affine(datain, Cnt, gantry_offset=False):
    '''Creates a blank reference image, to which another image will be resampled'''

    #------get necessary data for -----
    # gantry offset
    if gantry_offset:
        goff, tpo = mmraux.lm_pos(datain, Cnt)
    else:
        goff = np.zeros((3))
    vbed, hbed = mmraux.vh_bedpos(datain, Cnt)
    # create a reference empty mu-map image
    B = np.diag(np.array([-10*Cnt['SO_VXX'], 10*Cnt['SO_VXY'], 10*Cnt['SO_VXZ'], 1]))
    B[0,3] = 10*(.5*Cnt['SO_IMX']*Cnt['SO_VXX']      + goff[0])
    B[1,3] = 10*((-.5*Cnt['SO_IMY']+1)*Cnt['SO_VXY'] - goff[1])
    B[2,3] = 10*((-.5*Cnt['SO_IMZ']+1)*Cnt['SO_VXZ'] - goff[2] + hbed)
    # -------------------------------------------------------------------------------------
    return B


#================================================================================

def getmu_off(mu, Cnt, Offst=np.array([0., 0., 0.])):
    #number of voxels
    nvx = mu.shape[0]
    #change the shape to 3D
    mu.shape = (Cnt['SO_IMZ'], Cnt['SO_IMY'], Cnt['SO_IMX'])

    #-------------------------------------------------------------------------
    # CORRECT THE MU-MAP for GANTRY OFFSET
    #-------------------------------------------------------------------------
    Cim = {
        'VXSOx':0.208626,
        'VXSOy':0.208626,
        'VXSOz':0.203125,
        'VXNOx':344,
        'VXNOy':344,
        'VXNOz':127,

        'VXSRx':0.208626,
        'VXSRy':0.208626,
        'VXSRz':0.203125,
        'VXNRx':344,
        'VXNRy':344,
        'VXNRz':127
    }
    #original image offset
    Cim['OFFOx'] = -0.5*Cim['VXNOx']*Cim['VXSOx']
    Cim['OFFOy'] = -0.5*Cim['VXNOy']*Cim['VXSOy']
    Cim['OFFOz'] = -0.5*Cim['VXNOz']*Cim['VXSOz']
    #resampled image offset
    Cim['OFFRx'] = -0.5*Cim['VXNRx']*Cim['VXSRx']
    Cim['OFFRy'] = -0.5*Cim['VXNRy']*Cim['VXSRy']
    Cim['OFFRz'] = -0.5*Cim['VXNRz']*Cim['VXSRz']
    #transformation matrix
    A = np.array(
        [[ 1., 0., 0.,  Offst[0] ],
        [  0., 1., 0.,  Offst[1] ],
        [  0., 0., 1.,  Offst[2] ],
        [  0., 0., 0.,  1. ]], dtype=np.float32
        )
    #apply the gantry offset to the mu-map
    mur = nimpa.prc.improc.resample(mu, A, Cim)
    return mur

def getinterfile_off(fmu, Cnt, Offst=np.array([0., 0., 0.])):
    '''
    Return the floating point mu-map in an array from Interfile, accounting for image offset (does slow interpolation).
    '''
    #read the image file
    f = open(fmu, 'rb')
    mu = np.fromfile(f, np.float32)
    f.close()
    
    # save_im(mur, Cnt, os.path.dirname(fmu) + '/mur.nii')
    #-------------------------------------------------------------------------
    mur = getmu_off(mu, Cnt)
    #create GPU version of the mu-map
    murs = convert2dev(mur, Cnt)
    #get the basic stats
    mumax = np.max(mur)
    mumin = np.min(mur)
    #number of voxels greater than 10% of max image value
    n10mx = np.sum(mur>0.1*mumax)
    #return image dictionary with the image itself and some other stats
    mu_dct = {'im':mur,
              'ims':murs,
              'max':mumax,
              'min':mumin,
              'nvx':nvx,
              'n10mx':n10mx}
    return mu_dct

#================================================================================
def getinterfile(fim, Cnt):
    '''Return the floating point image file in an array from an Interfile file.'''
    
    #read the image file
    f = open(fim, 'rb')
    im = np.fromfile(f, np.float32)
    f.close()

    #number of voxels
    nvx = im.shape[0]
    #change the shape to 3D
    im.shape = (Cnt['SO_IMZ'], Cnt['SO_IMY'], Cnt['SO_IMX'])

    #get the basic stats
    immax = np.max(im)
    immin = np.min(im)

    #number of voxels greater than 10% of max image value
    n10mx = np.sum(im>0.1*immax)

    #reorganise the image for optimal gpu execution
    im_sqzd = convert2dev(im, Cnt)

    #return image dictionary with the image itself and some other stats
    im_dct = {'im':im,
              'ims':im_sqzd,
              'max':immax,
              'min':immin,
              'nvx':nvx,
              'n10mx':n10mx}

    return im_dct
#================================================================================


#=================================================================================
#-define uniform cylinder
def get_cylinder(Cnt, rad=25, xo=0, yo=0, unival=1, gpu_dim=False):
    '''Outputs image with a uniform cylinder of intensity = unival, radius = rad, and transaxial centre (xo, yo)'''
    imdsk = np.zeros((1, Cnt['SO_IMX'], Cnt['SO_IMY']), dtype=np.float32)
    for t in np.arange(0, math.pi, math.pi/(2*360)):
        x = xo+rad*math.cos(t)
        y = yo+rad*math.sin(t)
        yf = np.arange(-y+2*yo, y, Cnt['SO_VXY']/2)
        v = np.int32(.5*Cnt['SO_IMX'] - np.ceil(yf/Cnt['SO_VXY']))
        u = np.int32(.5*Cnt['SO_IMY'] + np.floor(x/Cnt['SO_VXY']))
        imdsk[0,v,u] = unival
    if 'rSO_IMZ' in Cnt:
        nvz = Cnt['rSO_IMZ']
    else:
        nvz = Cnt['SO_IMZ']
    imdsk = np.repeat(imdsk, nvz, axis=0)
    if gpu_dim: imdsk = convert2dev(imdsk, Cnt)
    return imdsk


#================================================================================
def hu2mu(im):
    '''HU units to 511keV PET mu-values'''

    # convert nans to -1024 for the HU values only
    im[np.isnan(im)] = -1024
    # constants
    muwater  = 0.096
    mubone   = 0.172
    rhowater = 0.158
    rhobone  = 0.326
    uim = np.zeros(im.shape, dtype=np.float32)
    uim[im<=0] = muwater * ( 1+im[im<=0]*1e-3 )
    uim[im> 0] = muwater * \
        ( 1+im[im>0]*1e-3 * rhowater/muwater*(mubone-muwater)/(rhobone-rhowater) )
    # remove negative values
    uim[uim<0] = 0
    return uim


# =====================================================================================
# object/patient mu-map resampling to NIfTI
# better use dcm2niix
def mudcm2nii(datain, Cnt):
    '''DICOM mu-map to NIfTI'''

    mu, pos, ornt = nimpa.dcm2im(datain['mumapDCM'])
    mu *= 0.0001
    A = pos['AFFINE']
    A[0,0] *= -1
    A[0,3] *= -1
    A[1,3] += A[1,1]
    nimpa.array2nii(mu[:,::-1,:], A, os.path.join(os.path.dirname(datain['mumapDCM']),'mu.nii.gz'))

    #------get necessary data for creating a blank reference image (to which resample)-----
    # gantry offset
    goff, tpo = mmraux.lm_pos(datain, Cnt)
    ihdr, csainfo = mmraux.hdr_lm(datain)
    #start horizontal bed position
    p = re.compile(r'start horizontal bed position.*\d{1,3}\.*\d*')
    m = p.search(ihdr)
    fi = ihdr[m.start():m.end()].find('=')
    hbedpos = 0.1*float(ihdr[m.start()+fi+1:m.end()])

    B = np.diag(np.array([-10*Cnt['SO_VXX'], 10*Cnt['SO_VXY'], 10*Cnt['SO_VXZ'], 1]))
    B[0,3] = 10*(.5*Cnt['SO_IMX']*Cnt['SO_VXX']      + goff[0])
    B[1,3] = 10*((-.5*Cnt['SO_IMY']+1)*Cnt['SO_VXY'] - goff[1])
    B[2,3] = 10*((-.5*Cnt['SO_IMZ']+1)*Cnt['SO_VXZ'] - goff[2] + hbedpos)
    im = np.zeros((Cnt['SO_IMZ'], Cnt['SO_IMY'], Cnt['SO_IMX']), dtype=np.float32)
    nimpa.array2nii(im, B, os.path.join(os.path.dirname(datain['mumapDCM']),'muref.nii.gz'))
    # -------------------------------------------------------------------------------------
    fmu = os.path.join(os.path.dirname(datain['mumapDCM']),'mu_r.nii.gz')
    if os.path.isfile( Cnt['RESPATH'] ):
        call( [ Cnt['RESPATH'],
                    '-ref', os.path.join(os.path.dirname(datain['mumapDCM']),'muref.nii.gz'),
                    '-flo', os.path.join(os.path.dirname(datain['mumapDCM']),'mu.nii.gz'),
                    '-res', fmu,
                    '-pad', '0'] )
    else:
        print 'e> path to resampling executable is incorrect!'
        raise IOError('Error launching NiftyReg for image resampling.')

    return fmu

# =====================================================================================
def obj_mumap(datain, params, outpath='', store=False, comment='', gantry_offset=True):
    '''Get the object mu-map from DICOM images'''

    # two ways of passing Cnt are here decoded
    if 'Cnt' in params:
        Cnt = params['Cnt']
    else:
        Cnt = params

    # output folder
    if outpath=='':
        fmudir = os.path.join( datain['corepath'], 'mumap-obj' )
    else:
        fmudir = os.path.join( outpath, 'mumap-obj' )
    mmraux.create_dir(fmudir)
    # ref file name
    fmuref = os.path.join(fmudir, 'muref.nii.gz')
    # ref affine
    B = image_affine(datain, Cnt, gantry_offset=gantry_offset)
    # ref image (blank)
    im = np.zeros((Cnt['SO_IMZ'], Cnt['SO_IMY'], Cnt['SO_IMX']), dtype=np.float32)
    # store ref image
    nimpa.array2nii(im, B, fmuref)

    # check if the object dicom files for MR-based mu-map exists
    if not 'mumapDCM' in datain or not os.path.isdir(datain['mumapDCM']):
        print 'e> DICOM forlder for the mu-map does not exist.'
        return None

    fnii = 'converted-from-object-DICOM_'
    tstmp = nimpa.time_stamp(simple_ascii=True)

    # find residual(s) from previous runs and delete them
    resdcm = glob.glob( os.path.join(fmudir, '*'+fnii+'*.nii*') )
    for d in resdcm:
        os.remove(d)

    # convert the DICOM mu-map images to nii
    call( [ Cnt['DCM2NIIX'], '-f', fnii+tstmp, '-o', fmudir, datain['mumapDCM'] ] )
    #files for the T1w, pick one:
    fmunii = glob.glob( os.path.join(fmudir, '*'+fnii+tstmp+'*.nii*') )[0]
    # fmunii = glob.glob( os.path.join(datain['mumapDCM'], '*converted*.nii*') )
    # fmunii = fmunii[0]

    # the converted nii image resample to the reference size
    fmu = os.path.join(fmudir, comment+'mumap_tmp.nii.gz')
    if os.path.isfile( Cnt['RESPATH'] ):
        cmd = [ Cnt['RESPATH'],
                    '-ref', fmuref,
                    '-flo', fmunii,
                    '-res', fmu,
                    '-pad', '0']
        if not Cnt['VERBOSE']: cmd.append('-voff')
        call(cmd)
    else:
        print 'e> path to resampling executable is incorrect!'
        sys.exit()

    nim = nib.load(fmu)
    # get the affine transform
    A = nim.get_sform()
    mu = nim.get_data()
    mu = np.transpose(mu[:,::-1,::-1], (2, 1, 0))
    # convert to mu-values
    mu = np.float32(mu)/1e4
    mu[mu<0] = 0

    # # del the temporary file for mumap
    # os.remove(fmu)
    # os.remove(fmunii)
    
    #return image dictionary with the image itself and some other stats
    mu_dct = {  'im':mu,
                'fmuref':fmuref,
                'affine':A}

    # store the mu-map if requested (by default no)
    if store:
        # to numpy array
        fnp = os.path.join(fmudir, 'mumap-from-DICOM.npy' )
        np.save(fnp, (mu, A))
        # with this file name
        fmudir = os.path.join(fmudir, comment+'mumap-from-DICOM.nii.gz')
        nimpa.array2nii(mu[::-1,::-1,:], A, fmudir)
        mu_dct['fim'] = fmudir

    return mu_dct

#=================================================================================
# pCT/UTE MU-MAP ALIGNED
#---------------------------------------------------------------------------------
def align_mumap(
        datain,
        scanner_params,
        outpath='',
        hst=[],
        t0=0, t1=0,
        itr=2,
        faff='',
        fpet='',
        fcomment='',
        store=False,
        petopt='ac',
        musrc='ute', # another option is pct for mu-map source
        ute_name='UTE1',
    ):

    if not os.path.isfile(faff):
        from niftypet.nipet.prj import mmrrec
        #-histogram the list data if needed
        if not hst:
            from niftypet.nipet import mmrhist
            hst = mmrhist(datain, scanner_params, t0=t0, t1=t1)

    #-constants, transaxial and axial LUTs are extracted
    Cnt   = scanner_params['Cnt']
    txLUT = scanner_params['txLUT']
    axLUT = scanner_params['axLUT']

    #=========================================================
    #-get hardware mu-map
    if 'hmumap' in datain and os.path.isfile(datain['hmumap']):
        muh, _, _ = np.load(datain['hmumap'])
        if Cnt['VERBOSE']: print 'i> loaded hardware mu-map from file:', datain['hmumap']
    elif outpath!='':
        hmupath = os.path.join( os.path.join(outpath,'mumap-hdw'), 'hmumap.npy')
        if os.path.isfile( hmupath ):
            muh, _, _ = np.load(hmupath)
            datain['hmumap'] = hmupath
        else:
            raise IOError('Invalid path to the hardware mu-map')
    else:
        print 'e> obtain the hardware mu-map first.'
        raise IOError('Could not find the hardware mu-map.  Have you run the routine for hardware mu-map?')
    #=========================================================
    #-check if T1w image is available
    if not 'MRT1W#' in datain and not 'T1nii' in datain and not 'T1bc' in datain \
    and not 'T1N4' in datain:
        print 'e> no MR T1w images required for co-registration!'
        raise IOError('T1w image could not be obtained')
    #=========================================================

    #-output dictionary
    mu_dct = {}
    #-if the affine is not given, 
    #-it will be generated by reconstructing PET image, with some or no corrections
    if not os.path.isfile(faff):
        # first recon pet to get the T1 aligned to it
        if petopt=='qnt':
            # ---------------------------------------------
            # OPTION 1 (quantitative recon with all corrections using MR-based mu-map)
            # get UTE object mu-map (may not be in register with the PET data)
            mudic = obj_mumap(datain, Cnt)
            muo = mudic['im']
            # reconstruct PET image with UTE mu-map to which co-register T1w
            recout = mmrrec.osemone(   
                datain, [muh, muo], 
                hst, scanner_params,
                recmod=3, itr=itr, fwhm=0.,
                fcomment=fcomment+'_qntUTE',
                outpath=os.path.join(outpath,'PET'),
                store_img=True)
        elif petopt=='nac':
            # ---------------------------------------------
            # OPTION 2 (recon without any corrections for scatter and attenuation)
            # reconstruct PET image with UTE mu-map to which co-register T1w
            muo = np.zeros(muh.shape, dtype=muh.dtype)
            recout = mmrrec.osemone(
                datain, [muh, muo],
                hst, scanner_params, 
                recmod=1, itr=itr, fwhm=0., 
                fcomment=fcomment+'_NAC',
                outpath=os.path.join(outpath,'PET'),
                store_img=True)
        elif petopt=='ac':
            # ---------------------------------------------
            # OPTION 3 (recon with attenuation correction only but no scatter)
            # reconstruct PET image with UTE mu-map to which co-register T1w
            mudic = obj_mumap(datain, Cnt, outpath=outpath)
            muo = mudic['im']
            recout = mmrrec.osemone(
                datain, [muh, muo],
                hst, scanner_params, 
                recmod=1, itr=itr, fwhm=0., 
                fcomment=fcomment+'_AC',
                outpath=os.path.join(outpath,'PET'),
                store_img=True)       

        fpet = recout.fpet
        mu_dct['fpet'] = fpet

        #------------------------------
        if musrc=='ute' and ute_name in datain and os.path.exists(datain[ute_name]):
            # change to NIfTI if the UTE sequence is in DICOM files (folder)
            if os.path.isdir(datain[ute_name]):
                fnew =  os.path.basename(datain[ute_name])
                call( [ Cnt['DCM2NIIX'], '-f', fnew, datain[ute_name] ] )
                fute = glob.glob(os.path.join(datain[ute_name], fnew+'*nii*'))[0]
            elif os.path.isfile(datain[ute_name]):
                fute = datain[ute_name]
            # get the affine transformation
            faff, _ = nimpa.affine_niftyreg(
                fpet, fute,
                outpath=outpath,
                #fcomment=fcomment,
                exepath = Cnt['REGPATH'],
                omp = multiprocessing.cpu_count()/2,
                rigOnly = True,
                affDirect = False,
                maxit=5,
                speed=True,
                pi=50, pv=50,
                smof=0, smor=0,
                rmsk=True,
                fmsk=True,
                rfwhm=15., #millilitres
                rthrsh=0.05,
                ffwhm = 15., #millilitres
                fthrsh=0.05,
                verbose=Cnt['VERBOSE']
            )
            os.remove(fute)
        elif musrc=='pct':
            faff, _ = nimpa.reg_mr2pet(  
            fpet, datain, Cnt,
            rigOnly = True,
            outpath=outpath,
            #fcomment=fcomment
        )
        else: 
            raise IOError('Floating MR image not provided or is invalid.')
        

    #-resampling output
    if outpath=='':
        if musrc=='pct':
            mudir = os.path.dirname(datain['pCT'])
        elif musrc=='ute':
            mudir = os.path.dirname(datain['UTE'])
        else:
            raise NameError('Non-existent mu-map source selection.')
    else:
        mudir = os.path.join(outpath, 'mumap-obj')
    #-create the folder, if not existent
    nimpa.create_dir(mudir)
    
    #-output file name for the aligned pseudo-CT
    if musrc=='pct':
        freg = os.path.join(mudir, 'pCT_r_tmp'+fcomment+'.nii.gz')
        fflo = datain['pCT']
    elif musrc=='ute':
        freg = os.path.join(mudir, 'UTE-r-tmp'+fcomment+'.nii.gz')
        if 'UTE' not in datain:
            fnii = 'converted'
            # convert the DICOM mu-map images to nii
            if 'mumapDCM' not in datain: 
                raise IOError('DICOM with the UTE mu-map are not given.')
            call( [ Cnt['DCM2NIIX'], '-f', fnii, datain['mumapDCM'] ] )
            #files for the T1w, pick one:
            fflo = glob.glob( os.path.join(datain['mumapDCM'], '*converted*.nii*') )
            fflo = fflo[0]
        else:
            if os.path.isfile(datain['UTE']):
                fflo = datain['UTE']
            else:
                raise IOError('The provided NIfTI UTE path is not valid.')

    #-call the resampling routine to get the pCT/UTE in place
    if os.path.isfile( Cnt['RESPATH'] ):
        cmd = [Cnt['RESPATH'],
            '-ref', fpet,
            '-flo', fflo,
            '-trans', faff,
            '-res', freg,
            '-pad', '0']
        if not Cnt['VERBOSE']: cmd.append('-voff')
        call(cmd)
    else:
        raise IOError('Wrong executive for resampling.')


    #-get the NIfTI of registered image
    nim = nib.load(freg)
    A   = nim.affine
    imreg = np.float32( nim.get_data() )
    imreg = imreg[:,::-1,::-1]
    imreg = np.transpose(imreg, (2, 1, 0))

    #-convert to mu-values; sort out the file name too.
    if musrc=='pct':
        mu = hu2mu(imreg)
        fname = 'mumapCT'
    elif musrc=='ute':
        mu = np.float32(imreg)/1e4
        #-remove the converted file from DICOMs
        os.remove(fflo)
        fname = 'mumapUTE'
    else:
        raise NameError('Confused o_O.')

    # get rid of negatives
    mu[mu<0] = 0
    # return image dictionary with the image itself and other parameters
    mu_dct['im'] = mu
    mu_dct['affine'] = A
    mu_dct['faff'] = faff
    
    if store:
        # now save to numpy array and NIfTI in this folder
        if outpath=='':
            mumapdir = os.path.join( datain['corepath'], 'mumap-obj' )
        else:
            mumapdir = os.path.join(outpath, 'mumap-obj')
        nimpa.create_dir(mumapdir)
        # numpy
        fnp = os.path.join(mumapdir, fname+'.npy')
        np.save(fnp, (mu, A, fnp))
        # NIfTI
        fmu = os.path.join(mumapdir, fname +fcomment+ '.nii.gz')
        nimpa.array2nii(mu[::-1,::-1,:], A, fmu)
        mu_dct['fim'] = fmu
        datain[fname] = fnp

    return mu_dct


#=================================================================================
# PSEUDO CT MU-MAP
#---------------------------------------------------------------------------------
def pct_mumap(
        datain, scanner_params,
        hst=[], t0=0, t1=0, itr=2,
        faff='', fpet='', fcomment='', outpath='',
        store=False, petopt='ac',
        #smor=0.0, smof=0.0, rthrsh=0.05, fthrsh=0.05
        ):
        
    '''
    GET THE MU-MAP from pCT IMAGE (which is in T1w space)
    * the mu-map will be registered to PET which will be reconstructed for time frame t0-t1
    * it f0 and t1 are not given the whole LM dataset will be reconstructed 
    * the reconstructed PET can be attenuation and scatter corrected or NOT using petopt
    '''

    if not os.path.isfile(faff):
        from niftypet.nipet.prj import mmrrec
        # histogram the list data if needed
        if not hst:
            from niftypet.nipet.lm import mmrhist
            hst = mmrhist.mmrhist(datain, scanner_params, t0=t0, t1=t1)
        
    # constants, transaxial and axial LUTs are extracted
    Cnt   = scanner_params['Cnt']
    txLUT = scanner_params['txLUT']
    axLUT = scanner_params['axLUT']

    # get hardware mu-map
    if 'hmumap' in datain and os.path.isfile(datain['hmumap']):
        muh, _, _ = np.load(datain['hmumap'])
        if Cnt['VERBOSE']: print 'i> loaded hardware mu-map from file:', datain['hmumap']
    elif outpath!='':
        hmupath = os.path.join( os.path.join(outpath,'mumap-hdw'), 'hmumap.npy')
        if os.path.isfile( hmupath ):
            muh, _, _ = np.load(hmupath)
            datain['hmumap'] = hmupath
        else:
            raise IOError('Invalid path to the hardware mu-map')
    else:
        print 'e> obtain the hardware mu-map first.'
        raise IOError('Could not find the hardware mu-map.  Have you run the routine for hardware mu-map?')

    if not 'MRT1W#' in datain and not 'T1nii' in datain and not 'T1bc' in datain:
        print 'e> no MR T1w images required for co-registration!'
        raise IOError('Missing MR data')
    # ----------------------------------

    # output dictionary
    mu_dct = {}
    if not os.path.isfile(faff):
        # first recon pet to get the T1 aligned to it
        if petopt=='qnt':
            # ---------------------------------------------
            # OPTION 1 (quantitative recon with all corrections using MR-based mu-map)
            # get UTE object mu-map (may not be in register with the PET data)
            mudic = obj_mumap(datain, Cnt)
            muo = mudic['im']
            # reconstruct PET image with UTE mu-map to which co-register T1w
            recout = mmrrec.osemone(   
                datain, [muh, muo], 
                hst, scanner_params,
                recmod=3, itr=itr, fwhm=0.,
                fcomment=fcomment+'_qntUTE',
                outpath=os.path.join(outpath,'PET'),
                store_img=True)
        elif petopt=='nac':
            # ---------------------------------------------
            # OPTION 2 (recon without any corrections for scatter and attenuation)
            # reconstruct PET image with UTE mu-map to which co-register T1w
            muo = np.zeros(muh.shape, dtype=muh.dtype)
            recout = mmrrec.osemone(
                datain, [muh, muo],
                hst, scanner_params, 
                recmod=1, itr=itr, fwhm=0., 
                fcomment=fcomment+'_NAC',
                outpath=os.path.join(outpath,'PET'),
                store_img=True)
        elif petopt=='ac':
            # ---------------------------------------------
            # OPTION 3 (recon with attenuation correction only but no scatter)
            # reconstruct PET image with UTE mu-map to which co-register T1w
            mudic = obj_mumap(datain, Cnt, outpath=outpath)
            muo = mudic['im']
            recout = mmrrec.osemone(
                datain, [muh, muo],
                hst, scanner_params, 
                recmod=1, itr=itr, fwhm=0., 
                fcomment=fcomment+'_AC',
                outpath=os.path.join(outpath,'PET'),
                store_img=True)       

        fpet = recout.fpet
        mu_dct['fpet'] = fpet
        #------------------------------
        # get the affine transformation
        faff, _ = nimpa.reg_mr2pet(  
                fpet, datain, Cnt,
                rigOnly = True,
                outpath=outpath,
                #fcomment=fcomment
        )
        #------------------------------

    # pCT file name
    if outpath=='':
        pctdir = os.path.dirname(datain['pCT'])
    else:
        pctdir = os.path.join(outpath, 'mumap-obj')
    mmraux.create_dir(pctdir)
    fpct = os.path.join(pctdir, 'pCT_r_tmp'+fcomment+'.nii.gz')

    #call the resampling routine to get the pCT in place
    if os.path.isfile( Cnt['RESPATH'] ):
        cmd = [Cnt['RESPATH'],
            '-ref', fpet,
            '-flo', datain['pCT'],
            '-trans', faff,
            '-res', fpct,
            '-pad', '0']
        if not Cnt['VERBOSE']: cmd.append('-voff')
        call(cmd)
    else:
        print 'e> path to resampling executable is incorrect!'
        sys.exit()


    # get the NIfTI of the pCT
    nim = nib.load(fpct)
    A   = nim.get_sform()
    pct = np.float32( nim.get_data() )
    pct = pct[:,::-1,::-1]
    pct = np.transpose(pct, (2, 1, 0))
    # convert the HU units to mu-values
    mu = hu2mu(pct)
    # get rid of negatives
    mu[mu<0] = 0
    
    # return image dictionary with the image itself and other parameters
    mu_dct['im'] = mu
    mu_dct['affine'] = A
    mu_dct['faff'] = faff

    if store:
        # now save to numpy array and NIfTI in this folder
        if outpath=='':
            pctumapdir = os.path.join( datain['corepath'], 'mumap-obj' )
        else:
            pctumapdir = os.path.join(outpath, 'mumap-obj')
        mmraux.create_dir(pctumapdir)
        # numpy
        fnp = os.path.join(pctumapdir, 'mumap-pCT.npy')
        np.save(fnp, (mu, A, fnp))
        # NIfTI
        fmu = os.path.join(pctumapdir, 'mumap-pCT' +fcomment+ '.nii.gz')
        nimpa.array2nii(mu[::-1,::-1,:], A, fmu)
        mu_dct['fim'] = fmu
        datain['mumapCT'] = fnp

    return mu_dct


#*********************************************************************************
#GET HARDWARE MU-MAPS with positions and offsets
#---------------------------------------------------------------------------------
def hdr_mu(datain, Cnt):
    '''Get the headers from DICOM data file'''
    #get one of the DICOM files of the mu-map
    if 'mumapDCM' in datain:
        files = glob.glob(os.path.join(datain['mumapDCM'],'*.dcm'))
        files.extend(glob.glob(os.path.join(datain['mumapDCM'],'*.DCM')))
        files.extend(glob.glob(os.path.join(datain['mumapDCM'],'*.ima')))
        files.extend(glob.glob(os.path.join(datain['mumapDCM'],'*.IMA')))
        dcmf = files[0]
    else:
        print 'e> path to the DICOM mu-map is not given but it is required.'
        raise NameError('No DICOM mu-map')
    if os.path.isfile( dcmf ):
        dhdr = dcm.read_file( dcmf )
    else:
        print 'e> DICOM mMR mu-maps are not valid files!'
        return None
    # CSA Series Header Info
    if [0x29,0x1020] in dhdr:
        csahdr = dhdr[0x29,0x1020].value
        if Cnt['VERBOSE']: print 'i> got CSA mu-map info.'
    return csahdr, dhdr

def hmu_shape(hdr):
    #regular expression to find the shape
    p = re.compile(r'(?<=:=)\s*\d{1,4}')
    # x: dim [1]
    i0 = hdr.find('matrix size[1]')
    i1 = i0+hdr[i0:].find('\n')
    u = int(p.findall(hdr[i0:i1])[0])
    # x: dim [2]
    i0 = hdr.find('matrix size[2]')
    i1 = i0+hdr[i0:].find('\n')
    v = int(p.findall(hdr[i0:i1])[0])
    # x: dim [3]
    i0 = hdr.find('matrix size[3]')
    i1 = i0+hdr[i0:].find('\n')
    w = int(p.findall(hdr[i0:i1])[0])
    return w,v,u

def hmu_voxsize(hdr):
    #regular expression to find the shape
    p = re.compile(r'(?<=:=)\s*\d{1,2}[.]\d{1,10}')
    # x: dim [1]
    i0 = hdr.find('scale factor (mm/pixel) [1]')
    i1 = i0+hdr[i0:].find('\n')
    vx = float(p.findall(hdr[i0:i1])[0])
    # x: dim [2]
    i0 = hdr.find('scale factor (mm/pixel) [2]')
    i1 = i0+hdr[i0:].find('\n')
    vy = float(p.findall(hdr[i0:i1])[0])
    # x: dim [3]
    i0 = hdr.find('scale factor (mm/pixel) [3]')
    i1 = i0+hdr[i0:].find('\n')
    vz = float(p.findall(hdr[i0:i1])[0])
    return np.array([0.1*vz, 0.1*vy, 0.1*vx])

def hmu_origin(hdr):
    #regular expression to find the origin
    p = re.compile(r'(?<=:=)\s*\d{1,5}[.]\d{1,10}')
    # x: dim [1]
    i0 = hdr.find('$umap origin (pixels) [1]')
    i1 = i0+hdr[i0:].find('\n')
    x = float(p.findall(hdr[i0:i1])[0])
    # x: dim [2]
    i0 = hdr.find('$umap origin (pixels) [2]')
    i1 = i0+hdr[i0:].find('\n')
    y = float(p.findall(hdr[i0:i1])[0])
    # x: dim [3]
    i0 = hdr.find('$umap origin (pixels) [3]')
    i1 = i0+hdr[i0:].find('\n')
    z = -float(p.findall(hdr[i0:i1])[0])
    return np.array([z, y, x])

def hmu_offset(hdr):
    #regular expression to find the origin
    p = re.compile(r'(?<=:=)\s*\d{1,5}[.]\d{1,10}')
    if hdr.find('$origin offset')>0:
        # x: dim [1]
        i0 = hdr.find('$origin offset (mm) [1]')
        i1 = i0+hdr[i0:].find('\n')
        x = float(p.findall(hdr[i0:i1])[0])
        # x: dim [2]
        i0 = hdr.find('$origin offset (mm) [2]')
        i1 = i0+hdr[i0:].find('\n')
        y = float(p.findall(hdr[i0:i1])[0])
        # x: dim [3]
        i0 = hdr.find('$origin offset (mm) [3]')
        i1 = i0+hdr[i0:].find('\n')
        z = -float(p.findall(hdr[i0:i1])[0])
        return np.array([0.1*z, 0.1*y, 0.1*x])
    else:
        return np.array([0.0, 0.0, 0.0])

def rd_hmu(fh):
    #--read hdr file--
    f = open(fh, 'r')
    hdr = f.read()
    f.close()
    #-----------------
    #regular expression to find the file name
    p = re.compile(r'(?<=:=)\s*\w*[.]\w*')
    i0 = hdr.find('!name of data file')
    i1 = i0+hdr[i0:].find('\n')
    fbin = p.findall(hdr[i0:i1])[0]
    #--read img file--
    f = open(os.path.join(os.path.dirname(fh), fbin.strip()), 'rb')
    im = np.fromfile(f, np.float32)
    f.close()
    #-----------------
    return  hdr, im 


def get_hmupos(datain, parts, Cnt, outpath=''):

    # check if registration executable exists
    if not os.path.isfile(Cnt['RESPATH']):
        print 'e> no registration executable found!'
        sys.exit()

    #----- get positions from the DICOM list-mode file -----
    ihdr, csainfo = mmraux.hdr_lm(datain, Cnt)
    #table position origin
    fi = csainfo.find('TablePositionOrigin')
    tpostr = csainfo[fi:fi+200]
    tpo = re.sub(r'[^a-zA-Z0-9\-\.]', '', tpostr).split('M')
    tpozyx = np.array([float(tpo[-1]), float(tpo[-2]), float(tpo[-3])]) / 10
    if Cnt['VERBOSE']: print 'i> table position (z,y,x) (cm):', tpozyx
    #--------------------------------------------------------

    #------- get positions from the DICOM mu-map file -------
    csamu, dhdr = hdr_mu(datain, Cnt)
    tmp = re.search('GantryTableHomeOffset(?!_)', csamu)
    gtostr1  = csamu[ tmp.start():tmp.start()+300 ]
    gtostr2 = re.sub(r'[^a-zA-Z0-9\-\.]', '', gtostr1)
    # gantry table offset, through conversion of string to float
    gtoxyz = re.findall(r'(?<=M)-*[\d]{1,4}\.[\d]{6,9}', gtostr2)
    gtozyx = np.float32(gtoxyz)[::-1]/10
    #--------------------------------------------------------

    if Cnt['VERBOSE']: print 'i> gantry table offset (z,y,x) (cm):', gtozyx

    ## ----
    ## old II
    # csamu, dhdr = nipet.img.mmrimg.hdr_mu(datain, Cnt)
    # tmp = re.search('GantryTableHomeOffset(?!_)', csamu)
    # gtostr = csamu[ tmp.start():tmp.start()+300 ]
    # gto = re.sub(r'[^a-zA-Z0-9\-\.]', '', gtostr).split('M')
    # # get the first three numbers
    # zyx = np.zeros(3, dtype=np.float32)
    # c = 0
    # for i in range(len(gto)):
    #     if re.search(r'[\d]{1,3}\.[\d]{6}', gto[i])!=None and c<3:
    #         zyx[c] = np.float32(re.sub(r'[^0-9\-\.]', '', gto[i]))
    #         c+=1
    # #gantry table offset
    # gtozyx = zyx[::-1]/10
    ## ----

    ## ----
    ## old I: only worked for syngo MR B20P
    # fi = csamu.find('GantryTableHomeOffset') 
    # gtostr =csamu[fi:fi+300]
    # if dhdr[0x0018, 0x1020].value == 'syngo MR B20P':
    #     gto = re.sub(r'[^a-zA-Z0-9\-\.]', '', gtostr).split('M')
    #     # get the first three numbers
    #     zyx = np.zeros(3, dtype=np.float32)
    #     c = 0
    #     for i in range(len(gto)):
    #         if re.search(r'[\d]', gto[i])!=None and c<3:
    #             zyx[c] = np.float32(re.sub(r'[^0-9\-\.]', '', gto[i]))
    #             c+=1
    #     #gantry table offset
    #     gtozyx = zyx[::-1]/10
    #     if Cnt['VERBOSE']: print 'i> gantry table offset (z,y,x) (cm):', gtozyx
    # # older scanner version
    # elif dhdr[0x0018, 0x1020].value == 'syngo MR B18P':
    #     zyx = np.zeros(3, dtype=np.float32)
    #     for k in range(3):
    #         tmp = re.search(r'\{\s*[\-0-9.]*\s*\}', gtostr)
    #         i0 = tmp.start()
    #         i1 = tmp.end()
    #         if gtostr[i0+1:i1-1]!=' ':  zyx[k] = np.float32(gtostr[i0+1:i1-1])
    #         gtostr = gtostr[i1:]
    #     #gantry table offset
    #     gtozyx = zyx[::-1]/10
    #     if Cnt['VERBOSE']: print 'i> gantry table offset (z,y,x) (cm):', gtozyx
    ## -----

    # create the folder for hardware mu-maps
    if outpath=='':
        dirhmu = os.path.join( datain['corepath'], 'mumap-hdw')
    else:
        dirhmu = os.path.join( outpath, 'mumap-hdw')
    mmraux.create_dir(dirhmu)
    # get the reference nii image
    fref = os.path.join(dirhmu, 'hmuref.nii.gz')

    #start horizontal bed position
    p = re.compile(r'start horizontal bed position.*\d{1,3}\.*\d*')
    m = p.search(ihdr)
    fi = ihdr[m.start():m.end()].find('=')
    hbedpos = 0.1*float(ihdr[m.start()+fi+1:m.end()])

    #start vertical bed position
    p = re.compile(r'start vertical bed position.*\d{1,3}\.*\d*')
    m = p.search(ihdr)
    fi = ihdr[m.start():m.end()].find('=')
    vbedpos = 0.1*float(ihdr[m.start()+fi+1:m.end()])

    if Cnt['VERBOSE']: print 'i> creating reference nii image for resampling'
    B = np.diag(np.array([-10*Cnt['SO_VXX'], 10*Cnt['SO_VXY'], 10*Cnt['SO_VXZ'], 1]))
    B[0,3] = 10*(.5*Cnt['SO_IMX'])*Cnt['SO_VXX']
    B[1,3] = 10*( -.5*Cnt['SO_IMY']+1)*Cnt['SO_VXY']
    B[2,3] = 10*((-.5*Cnt['SO_IMZ']+1)*Cnt['SO_VXZ'] + hbedpos )
    nimpa.array2nii(  np.zeros((Cnt['SO_IMZ'], Cnt['SO_IMY'], Cnt['SO_IMX']), dtype=np.float32), B, fref)

    #define a dictionary of all positions/offsets of hardware mu-maps
    hmupos = [None]*5
    hmupos[0] = {
        'TabPosOrg' :   tpozyx, #from DICOM of LM file
        'GanTabOff' :   gtozyx, #from DICOM of mMR mu-map file
        'HBedPos'   :   hbedpos, #from Interfile of LM file [cm]
        'VBedPos'   :   vbedpos, #from Interfile of LM file [cm]
        'niipath'   :   fref
        }

    #--------------------------------------------------------------------------
    # iteratively go through the mu-maps and add them as needed
    for i in parts:
        fh = os.path.join(Cnt['HMUDIR'], Cnt['HMULIST'][i-1])
        # get the interfile header and binary data 
        hdr, im = rd_hmu(fh)
        #get shape, origin, offset and voxel size
        s = hmu_shape(hdr)
        im.shape = s
        # get the origin, offset and voxel size for the mu-map interfile data
        org = hmu_origin(hdr)
        off = hmu_offset(hdr)
        vs  = hmu_voxsize(hdr)
        # corner voxel position for the interfile image data
        vpos = (-org*vs + off + gtozyx - tpozyx)
        #add to the dictionary
        hmupos[i] = {
            'vpos'    :   vpos,
            'shape'   :   s,   #from interfile
            'iorg'    :   org, #from interfile
            'ioff'    :   off, #from interfile
            'ivs'     :   vs,  #from interfile
            'img'     :   im, #from interfile
            'niipath' :   os.path.join(dirhmu, '_'+Cnt['HMULIST'][i-1].split('.')[0]+'.nii.gz')
        }
        #save to NIfTI
        if Cnt['VERBOSE']: print 'i> creating mu-map for:', Cnt['HMULIST'][i-1]
        A = np.diag(np.append(10*vs[::-1], 1))
        A[0,0] *= -1
        A[0,3] =  10*(-vpos[2])
        A[1,3] = -10*((s[1]-1)*vs[1] + vpos[1])
        A[2,3] = -10*((s[0]-1)*vs[0] - vpos[0])
        nimpa.array2nii(im[::-1,::-1,:], A, hmupos[i]['niipath'])

        # resample using nify.reg
        fout = os.path.join(    os.path.dirname (hmupos[0]['niipath']),
                                'r'+os.path.basename(hmupos[i]['niipath']).split('.')[0]+'.nii.gz' )
        cmd = [ Cnt['RESPATH'],
                '-ref', hmupos[0]['niipath'],
                '-flo', hmupos[i]['niipath'],
                '-res', fout,
                '-pad', '0']
        if not Cnt['VERBOSE']: cmd.append('-voff')
        call(cmd)
                
    return hmupos


#-------------------------------------------------------------------------------------
def hdw_mumap(datain, hparts, params, outpath='', use_stored=False):
    ''' Get hardware mu-map components, including bed, coils etc.
    '''

    # two ways of passing Cnt are here decoded
    if 'Cnt' in params:
        Cnt = params['Cnt']
    else:
        Cnt = params

    if outpath!='':
        fmudir = os.path.join(outpath, 'mumap-hdw')
    else:
        fmudir = os.path.join(datain['corepath'], 'mumap-hdw')

    # if requested to use the stored hardware mu_map get it from the path in datain
    if 'hmumap' in datain and os.path.isfile(datain['hmumap']) and use_stored:
        hmu, A, fmu = np.load(datain['hmumap'])
        if Cnt['VERBOSE']: print 'i> loaded hardware mu-map from file:', datain['hmumap']
        fnp = datain['hmumap']
    elif outpath!='' and os.path.isfile(os.path.join(fmudir, 'hmumap.npy')):
        fnp = os.path.join(fmudir, 'hmumap.npy')
        hmu, A, fmu = np.load(fnp)
        datain['hmumap'] = fnp
    # otherwise generate it from the parts through resampling the high resolution CT images
    else:
        hmupos = get_hmupos(datain, hparts, Cnt, outpath=outpath)
        # just to get the dims, get the ref image
        nimo = nib.load(hmupos[0]['niipath'])
        A = nimo.get_sform()
        imo = np.float32( nimo.get_data() )
        imo[:] = 0

        for i in hparts:
            fin  = os.path.join(    os.path.dirname (hmupos[0]['niipath']),
                                    'r'+os.path.basename(hmupos[i]['niipath']).split('.')[0]+'.nii.gz' )
            nim = nib.load(fin)
            mu = nim.get_data()
            mu[mu<0] = 0
            
            imo += mu

        hdr = nimo.header
        hdr['cal_max'] = np.max(imo)
        hdr['cal_min'] = np.min(imo)
        fmu  = os.path.join(os.path.dirname (hmupos[0]['niipath']), 'hardware_umap.nii.gz' )
        nib.save(nimo, fmu)

        hmu = np.transpose(imo[:,::-1,::-1], (2, 1, 0))

        # save the objects to numpy arrays
        fnp = os.path.join(fmudir, 'hmumap.npy')
        np.save(fnp, (hmu, A, fmu))
        #update the datain dictionary (assuming it is mutable)
        datain['hmumap'] = fnp

    #return image dictionary with the image itself and some other stats
    hmu_dct = { 'im':hmu,
                'fim':fmu,
                'fnp':fnp,
                'affine':A}

    return hmu_dct


def rmumaps(datain, Cnt, t0=0, t1=0, use_stored=False):
    '''
    get the mu-maps for hardware and object and trim it axially for reduced rings case
    '''

    from niftypet.nipet.lm  import mmrhist
    from niftypet.nipet.prj import mmrrec

    fcomment = '(R)'

    # get hardware mu-map
    if os.path.isfile(datain['hmumap']) and use_stored:
        muh, _ = np.load(datain['hmumap'])
        if Cnt['VERBOSE']: print 'i> loaded hardware mu-map from file:', datain['hmumap']
    else:
        hmudic = hdw_mumap(datain, [1,2,4], Cnt)
        muh = hmudic['im']

    # get pCT mu-map if stored in numpy file and then exit, otherwise do all the processing
    if os.path.isfile(datain['mumapCT']) and use_stored:
        mup, _ = np.load(datain['mumapCT'])
        muh = muh[2*Cnt['RNG_STRT'] : 2*Cnt['RNG_END'], :, :]
        mup = mup[2*Cnt['RNG_STRT'] : 2*Cnt['RNG_END'], :, :]
        return [muh, mup]

    # get UTE object mu-map (may be not in register with the PET data)
    if os.path.isfile(datain['mumapUTE']) and use_stored:
        muo, _ = np.load(datain['mumapUTE'])
    else:
        mudic = obj_mumap(datain, Cnt, store=True)
        muo = mudic['im']

    if os.path.isfile(datain['pCT']):
        # reconstruct PET image with default settings to be used to alight pCT mu-map
        params = mmraux.mMR_params()
        Cnt_ = params['Cnt']
        txLUT_ = params['txLUT']
        axLUT_ = params['axLUT']
        
        # histogram for reconstruction with UTE mu-map
        hst = mmrhist.hist(datain, txLUT_, axLUT_, Cnt_, t0=t0, t1=t1)
        # reconstruct PET image with UTE mu-map to which co-register T1w
        recute = mmrrec.osemone(
            datain, [muh, muo], hst, params, 
            recmod=3, itr=4, fwhm=0., store_img=True, fcomment=fcomment+'_QNT-UTE'
        )
        # --- MR T1w
        if os.path.isfile(datain['T1nii']):
            ft1w = datain['T1nii']
        elif os.path.isfile(datain['T1bc']):
            ft1w = datain['T1bc']
        elif os.path.isdir(datain['MRT1W']):
            # create file name for the converted NIfTI image
            fnii = 'converted'
            call( [ Cnt['DCM2NIIX'], '-f', fnii, datain['T1nii'] ] )
            ft1nii = glob.glob( os.path.join(datain['T1nii'], '*converted*.nii*') )
            ft1w = ft1nii[0]
        else:
            print 'e> disaster: no T1w image!'
            sys.exit()

        #output for the T1w in register with PET
        ft1out = os.path.join(os.path.dirname(ft1w), 'T1w_r'+'.nii.gz')
        #text file fo rthe affine transform T1w->PET
        faff   = os.path.join(os.path.dirname(ft1w), fcomment+'mr2pet_affine'+'.txt')  #time.strftime('%d%b%y_%H.%M',time.gmtime())
        #call the registration routine
        if os.path.isfile( Cnt['REGPATH'] ):
            cmd = [Cnt['REGPATH'],
                 '-ref', recute.fpet,
                 '-flo', ft1w,
                 '-rigOnly', '-speeeeed',
                 '-aff', faff,
                 '-res', ft1out]
            if not Cnt['VERBOSE']: cmd.append('-voff')
            call(cmd)
        else:
            print 'e> path to registration executable is incorrect!'
            sys.exit()

        #get the pCT mu-map with the above faff
        pmudic = pct_mumap(datain, txLUT, axLUT, Cnt, faff=faff, fpet=recute.fpet, fcomment=fcomment)
        mup = pmudic['im']

        muh = muh[2*Cnt['RNG_STRT'] : 2*Cnt['RNG_END'], :, :]
        mup = mup[2*Cnt['RNG_STRT'] : 2*Cnt['RNG_END'], :, :]
        return [muh, mup]
    else:
        muh = muh[2*Cnt['RNG_STRT'] : 2*Cnt['RNG_END'], :, :]
        muo = muo[2*Cnt['RNG_STRT'] : 2*Cnt['RNG_END'], :, :]
        return [muh, muo]

#------------------------------------------------------------------------
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