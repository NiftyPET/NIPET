"""Image functions for PET data reconstruction and processing."""

import glob
import logging
import math
import multiprocessing
import os
import re
import shutil
from subprocess import run

import nibabel as nib
import numpy as np
import pydicom as dcm

from niftypet import nimpa

from .. import mmraux
from .. import resources as rs

log = logging.getLogger(__name__)
OFFSET_DEFAULT = np.array([0., 0., 0.])
ct_nans = -1024

# ==================================================================================
# IMAGE ROUTINES
# ==================================================================================


def convert2e7(img, Cnt):
    '''Convert GPU optimised image to Siemens/E7 image shape (127,344,344).'''

    margin = (Cnt['SO_IMX'] - Cnt['SZ_IMX']) // 2

    # permute the dims first
    imo = np.transpose(img, (2, 0, 1))

    nvz = img.shape[2]

    # > get the x-axis filler and apply it
    filler = np.zeros((nvz, Cnt['SZ_IMY'], margin), dtype=np.float32)
    imo = np.concatenate((filler, imo, filler), axis=2)

    # > get the y-axis filler and apply it
    filler = np.zeros((nvz, margin, Cnt['SO_IMX']), dtype=np.float32)
    imo = np.concatenate((filler, imo, filler), axis=1)
    return imo


def convert2dev(im, Cnt):
    '''Reshape Siemens/E7 (default) image for optimal GPU execution.'''
    if im.shape[1] != Cnt['SO_IMY'] or im.shape[2] != Cnt['SO_IMX']:
        raise ValueError('e> input image array is not of the correct Siemens shape.')

    if 'rSZ_IMZ' in Cnt and im.shape[0] != Cnt['rSZ_IMZ']:
        log.warning('the axial number of voxels does not match the reduced rings.')
    elif 'rSZ_IMZ' not in Cnt and im.shape[0] != Cnt['SZ_IMZ']:
        log.warning('the axial number of voxels does not match the rings.')

    im_sqzd = np.zeros((im.shape[0], Cnt['SZ_IMY'], Cnt['SZ_IMX']), dtype=np.float32)
    margin = int((Cnt['SO_IMX'] - Cnt['SZ_IMX']) / 2)
    margin_ = -margin
    if margin == 0:
        margin = None
        margin_ = None

    im_sqzd = im[:, margin:margin_, margin:margin_]
    im_sqzd = np.transpose(im_sqzd, (1, 2, 0))
    return im_sqzd


def cropxy(im, imsize, datain, Cnt, store_pth=''):
    '''
    Crop image transaxially to the size in tuple <imsize>.
    Return the image and the affine matrix.
    '''
    if not imsize[0] % 2 == 0 and not imsize[1] % 2 == 0:
        log.error('image size has to be an even number!')
        return None

    # cropping indexes
    i0 = int((Cnt['SO_IMX'] - imsize[0]) / 2)
    i1 = int((Cnt['SO_IMY'] + imsize[1]) / 2)

    B = image_affine(datain, Cnt, gantry_offset=False)
    B[0, 3] -= 10 * Cnt['SO_VXX'] * i0
    B[1, 3] += 10 * Cnt['SO_VXY'] * (Cnt['SO_IMY'] - i1)

    cim = im[:, i0:i1, i0:i1]

    if store_pth != '':
        nimpa.array2nii(cim[::-1, ::-1, :], B, store_pth, descrip='cropped')
        log.info('saved cropped image to:\n{}'.format(store_pth))

    return cim, B


def image_affine(datain, Cnt, gantry_offset=False):
    '''Creates a blank reference image, to which another image will be resampled'''

    # ------get necessary data for -----
    # gantry offset
    if gantry_offset:
        goff, tpo = mmraux.lm_pos(datain, Cnt)
    else:
        goff = np.zeros((3))
    vbed, hbed = mmraux.vh_bedpos(datain, Cnt)

    if 'rNRNG' in Cnt and 'rSO_IMZ' in Cnt:
        imz = Cnt['rSO_IMZ']
    else:
        imz = Cnt['SO_IMZ']

    # create a reference empty mu-map image
    B = np.diag(np.array([-10 * Cnt['SO_VXX'], 10 * Cnt['SO_VXY'], 10 * Cnt['SO_VXZ'], 1]))
    B[0, 3] = 10 * (.5 * Cnt['SO_IMX'] * Cnt['SO_VXX'] + goff[0])
    B[1, 3] = 10 * ((-.5 * Cnt['SO_IMY'] + 1) * Cnt['SO_VXY'] - goff[1])
    B[2, 3] = 10 * ((-.5 * imz + 1) * Cnt['SO_VXZ'] - goff[2] + hbed)
    # -------------------------------------------------------------------------------------
    return B


def getmu_off(mu, Cnt, Offst=OFFSET_DEFAULT):

    # phange the shape to 3D
    mu.shape = (Cnt['SO_IMZ'], Cnt['SO_IMY'], Cnt['SO_IMX'])

    # -------------------------------------------------------------------------
    # CORRECT THE MU-MAP for GANTRY OFFSET
    # -------------------------------------------------------------------------
    Cim = {
        'VXSOx': 0.208626, 'VXSOy': 0.208626, 'VXSOz': 0.203125, 'VXNOx': 344, 'VXNOy': 344,
        'VXNOz': 127, 'VXSRx': 0.208626, 'VXSRy': 0.208626, 'VXSRz': 0.203125, 'VXNRx': 344,
        'VXNRy': 344, 'VXNRz': 127}
    # priginal image offset
    Cim['OFFOx'] = -0.5 * Cim['VXNOx'] * Cim['VXSOx']
    Cim['OFFOy'] = -0.5 * Cim['VXNOy'] * Cim['VXSOy']
    Cim['OFFOz'] = -0.5 * Cim['VXNOz'] * Cim['VXSOz']
    # pesampled image offset
    Cim['OFFRx'] = -0.5 * Cim['VXNRx'] * Cim['VXSRx']
    Cim['OFFRy'] = -0.5 * Cim['VXNRy'] * Cim['VXSRy']
    Cim['OFFRz'] = -0.5 * Cim['VXNRz'] * Cim['VXSRz']
    # pransformation matrix
    A = np.array(
        [[1., 0., 0., Offst[0]], [0., 1., 0., Offst[1]], [0., 0., 1., Offst[2]], [0., 0., 0., 1.]],
        dtype=np.float32)
    # ppply the gantry offset to the mu-map
    mur = nimpa.prc.improc.resample(mu, A, Cim)
    return mur


def getinterfile_off(fmu, Cnt, Offst=OFFSET_DEFAULT):
    '''
    Return the floating point mu-map in an array from Interfile,
    accounting for image offset (does slow interpolation).
    '''
    # pead the image file
    f = open(fmu, 'rb')
    mu = np.fromfile(f, np.float32)
    f.close()

    # save_im(mur, Cnt, os.path.dirname(fmu) + '/mur.nii')
    # -------------------------------------------------------------------------
    mur = getmu_off(mu, Cnt)
    # > create GPU version of the mu-map
    murs = convert2dev(mur, Cnt)
    # > number of voxels
    nvx = mu.shape[0]
    # > get the basic stats
    mumax = np.max(mur)
    mumin = np.min(mur)
    # > number of voxels greater than 10% of max image value
    n10mx = np.sum(mur > 0.1 * mumax)
    # > return image dictionary with the image itself and some other stats
    mu_dct = {'im': mur, 'ims': murs, 'max': mumax, 'min': mumin, 'nvx': nvx, 'n10mx': n10mx}
    return mu_dct


def getinterfile(fim, Cnt):
    '''Return the floating point image file in an array from an Interfile file.'''
    # pead the image file
    f = open(fim, 'rb')
    im = np.fromfile(f, np.float32)
    f.close()

    # pumber of voxels
    nvx = im.shape[0]
    # phange the shape to 3D
    im.shape = (Cnt['SO_IMZ'], Cnt['SO_IMY'], Cnt['SO_IMX'])

    # pet the basic stats
    immax = np.max(im)
    immin = np.min(im)

    # pumber of voxels greater than 10% of max image value
    n10mx = np.sum(im > 0.1 * immax)

    # peorganise the image for optimal gpu execution
    im_sqzd = convert2dev(im, Cnt)

    # peturn image dictionary with the image itself and some other stats
    im_dct = {'im': im, 'ims': im_sqzd, 'max': immax, 'min': immin, 'nvx': nvx, 'n10mx': n10mx}

    return im_dct


# define uniform cylinder


def get_cylinder(Cnt, rad=25, xo=0, yo=0, unival=1, gpu_dim=False):
    """
    Outputs image with a uniform cylinder of
    intensity = unival, radius = rad, and transaxial centre (xo, yo)
    """
    imdsk = np.zeros((1, Cnt['SO_IMX'], Cnt['SO_IMY']), dtype=np.float32)
    for t in np.arange(0, math.pi, math.pi / (2*360)):
        x = xo + rad * math.cos(t)
        y = yo + rad * math.sin(t)
        yf = np.arange(-y + 2*yo, y, Cnt['SO_VXY'] / 2)
        v = np.int32(.5 * Cnt['SO_IMX'] - np.ceil(yf / Cnt['SO_VXY']))
        u = np.int32(.5 * Cnt['SO_IMY'] + np.floor(x / Cnt['SO_VXY']))
        imdsk[0, v, u] = unival
    if 'rSO_IMZ' in Cnt:
        nvz = Cnt['rSO_IMZ']
    else:
        nvz = Cnt['SO_IMZ']
    imdsk = np.repeat(imdsk, nvz, axis=0)
    if gpu_dim: imdsk = convert2dev(imdsk, Cnt)
    return imdsk


def hu2mu(im):
    '''HU units to 511keV PET mu-values'''
    # convert nans to -1024 for the HU values only
    im[np.isnan(im)] = ct_nans
    # constants
    muwater = 0.096
    mubone = 0.172
    rhowater = 0.158
    rhobone = 0.326
    uim = np.zeros(im.shape, dtype=np.float32)
    uim[im <= 0] = muwater * (1 + im[im <= 0] * 1e-3)
    uim[im > 0] = muwater * (1 + im[im > 0] * 1e-3 * rhowater / muwater * (mubone-muwater) /
                             (rhobone-rhowater))
    # remove negative values
    uim[uim < 0] = 0
    return uim


# =====================================================================================
# object/patient mu-map resampling to NIfTI
# better use dcm2niix
def mudcm2nii(datain, Cnt):
    '''DICOM mu-map to NIfTI'''
    mu, pos, ornt = nimpa.dcm2im(datain['mumapDCM'])
    mu *= 0.0001
    A = pos['AFFINE']
    A[0, 0] *= -1
    A[0, 3] *= -1
    A[1, 3] += A[1, 1]
    nimpa.array2nii(mu[:, ::-1, :], A,
                    os.path.join(os.path.dirname(datain['mumapDCM']), 'mu.nii.gz'))

    # ------get necessary data for creating a blank reference image (to which resample)-----
    # gantry offset
    goff, tpo = mmraux.lm_pos(datain, Cnt)
    ihdr, csainfo = mmraux.hdr_lm(datain)
    # ptart horizontal bed position
    p = re.compile(r'start horizontal bed position.*\d{1,3}\.*\d*')
    m = p.search(ihdr)
    fi = ihdr[m.start():m.end()].find('=')
    hbedpos = 0.1 * float(ihdr[m.start() + fi + 1:m.end()])

    B = np.diag(np.array([-10 * Cnt['SO_VXX'], 10 * Cnt['SO_VXY'], 10 * Cnt['SO_VXZ'], 1]))
    B[0, 3] = 10 * (.5 * Cnt['SO_IMX'] * Cnt['SO_VXX'] + goff[0])
    B[1, 3] = 10 * ((-.5 * Cnt['SO_IMY'] + 1) * Cnt['SO_VXY'] - goff[1])
    B[2, 3] = 10 * ((-.5 * Cnt['SO_IMZ'] + 1) * Cnt['SO_VXZ'] - goff[2] + hbedpos)
    im = np.zeros((Cnt['SO_IMZ'], Cnt['SO_IMY'], Cnt['SO_IMX']), dtype=np.float32)
    nimpa.array2nii(im, B, os.path.join(os.path.dirname(datain['mumapDCM']), 'muref.nii.gz'))
    # -------------------------------------------------------------------------------------
    opth = os.path.dirname(datain['mumapDCM'])
    fmu = os.path.join(opth, 'mu_r.nii.gz')
    nimpa.resample_dipy(os.path.join(opth, 'muref.nii.gz'), os.path.join(opth, 'mu.nii.gz'),
                        fimout=fmu, intrp=1, dtype_nifti=np.float32)
    return fmu


def obj_mumap(
    datain,
    params=None,
    outpath='',
    comment='',
    store=False,
    store_npy=False,
    gantry_offset=True,
    del_auxilary=True,
):
    '''Get the object mu-map from DICOM images'''
    if params is None:
        params = {}

    # three ways of passing scanner constants <Cnt> are here decoded
    if 'Cnt' in params:
        Cnt = params['Cnt']
    elif 'SO_IMZ' in params:
        Cnt = params
    else:
        Cnt = rs.get_mmr_constants()

    # output folder
    if outpath == '':
        fmudir = os.path.join(datain['corepath'], 'mumap-obj')
    else:
        fmudir = os.path.join(outpath, 'mumap-obj')
    nimpa.create_dir(fmudir)

    # > ref file name
    fmuref = os.path.join(fmudir, 'muref.nii.gz')

    # > ref affine
    B = image_affine(datain, Cnt, gantry_offset=gantry_offset)

    # > ref image (blank)
    im = np.zeros((Cnt['SO_IMZ'], Cnt['SO_IMY'], Cnt['SO_IMX']), dtype=np.float32)

    # > store ref image
    nimpa.array2nii(im, B, fmuref)

    # check if the object dicom files for MR-based mu-map exists
    if 'mumapDCM' not in datain or not os.path.isdir(datain['mumapDCM']):
        log.error('DICOM folder for the mu-map does not exist.')
        return None

    fnii = 'converted-from-object-DICOM_'
    tstmp = nimpa.time_stamp(simple_ascii=True)

    # find residual(s) from previous runs and delete them
    resdcm = glob.glob(os.path.join(fmudir, '*' + fnii + '*.nii*'))
    for d in resdcm:
        os.remove(d)

    # > convert the DICOM mu-map images to NIfTI
    run([Cnt['DCM2NIIX'], '-f', fnii + tstmp, '-o', fmudir, datain['mumapDCM']])
    # piles for the T1w, pick one:
    fmunii = glob.glob(os.path.join(fmudir, '*' + fnii + tstmp + '*.nii*'))[0]
    # fmunii = glob.glob( os.path.join(datain['mumapDCM'], '*converted*.nii*') )
    # fmunii = fmunii[0]

    # > resampled the NIfTI converted image to the reference shape/size
    fmu = os.path.join(fmudir, comment + 'mumap_tmp.nii.gz')
    nimpa.resample_dipy(fmuref, fmunii, fimout=fmu, intrp=1, dtype_nifti=np.float32)

    nim = nib.load(fmu)
    # get the affine transform
    A = nim.get_sform()
    mu = nim.get_fdata(dtype=np.float32)
    mu = np.transpose(mu[:, ::-1, ::-1], (2, 1, 0))
    # convert to mu-values
    mu = np.float32(mu) / 1e4
    mu[mu < 0] = 0

    # > return image dictionary with the image itself and some other stats
    mu_dct = {'im': mu, 'affine': A}
    if not del_auxilary:
        mu_dct['fmuref'] = fmuref

    # > store the mu-map if requested
    if store_npy:
        # to numpy array
        fnp = os.path.join(fmudir, "mumap-from-DICOM.npz")
        np.savez(fnp, mu=mu, A=A)

    if store:
        # with this file name
        fmumap = os.path.join(fmudir, 'mumap-from-DICOM_no-alignment' + comment + '.nii.gz')
        nimpa.array2nii(mu[::-1, ::-1, :], A, fmumap)
        mu_dct['fim'] = fmumap

    if del_auxilary:
        os.remove(fmuref)
        os.remove(fmunii)
        os.remove(fmu)

        if [f for f in os.listdir(fmudir)
                if not f.startswith('.') and not f.endswith('.json')] == []:
            shutil.rmtree(fmudir)

    return mu_dct


# ================================================================================
# pCT/UTE MU-MAP ALIGNED
# --------------------------------------------------------------------------------


def align_mumap(
    datain,
    scanner_params=None,
    outpath='',
    reg_tool='dipy',
    use_stored=False,
    hst=None,
    t0=0,
    t1=0,
    itr=2,
    faff='',
    fpet='',
    fcomment='',
    store=False,
    store_npy=False,
    petopt='ac',
    musrc='ute',         # another option is pct for mu-map source
    ute_name='UTE2',
    del_auxilary=True,
    verbose=False,
):
    '''
    Align the a pCT or MR-derived mu-map to a PET image reconstructed to chosen
    specifications (e.g., with/without attenuation and scatter corrections)

    use_sotred only works if hst or t0/t1 given but not when faff.
    '''
    if scanner_params is None:
        scanner_params = {}

    # > output folder
    if outpath == '':
        opth = os.path.join(datain['corepath'], 'mumap-obj')
    else:
        opth = os.path.join(outpath, 'mumap-obj')

    # > create the folder, if not existent
    nimpa.create_dir(opth)

    # > get the timing of PET if affine not given
    if faff == '' and hst is not None and isinstance(hst, dict) and 't0' in hst:
        t0 = hst['t0']
        t1 = hst['t1']

    # > file name for the output mu-map
    fnm = 'mumap-' + musrc.upper()

    # > output dictionary
    mu_dct = {}

    # ---------------------------------------------------------------------------
    # > used stored if requested
    if use_stored:
        fmu_stored = fnm + '-aligned-to_t'\
                     + str(t0)+'-'+str(t1)+'_'+petopt.upper()\
                     + fcomment
        fmupath = os.path.join(opth, fmu_stored + '.nii.gz')

        if os.path.isfile(fmupath):
            mudct_stored = nimpa.getnii(fmupath, output='all')
            # > create output dictionary
            mu_dct['im'] = mudct_stored['im']
            mu_dct['affine'] = mudct_stored['affine']
            # pu_dct['faff'] = faff
            return mu_dct
    # ---------------------------------------------------------------------------

    # > tmp folder for not aligned mu-maps
    tmpdir = os.path.join(opth, 'tmp')
    nimpa.create_dir(tmpdir)

    # > three ways of passing scanner constants <Cnt> are here decoded
    if 'Cnt' in scanner_params:
        Cnt = scanner_params['Cnt']
    elif 'SO_IMZ' in scanner_params:
        Cnt = scanner_params
    else:
        Cnt = rs.get_mmr_constants()

    # > if affine not provided histogram the LM data for recon and registration
    if not os.path.isfile(faff):
        from niftypet.nipet.prj import mmrrec

        # -histogram the list data if needed
        if hst is None:
            from niftypet.nipet import mmrhist
            if 'txLUT' in scanner_params:
                hst = mmrhist(datain, scanner_params, t0=t0, t1=t1)
            else:
                raise ValueError('Full scanner are parameters not provided\
                     but are required for histogramming.')

    # ========================================================
    # -get hardware mu-map
    if 'hmumap' in datain and os.path.isfile(datain['hmumap']):
        muh = np.load(datain['hmumap'], allow_pickle=True)["hmu"]
        (log.info if verbose else log.debug)('loaded hardware mu-map from file:\n{}'.format(
            datain['hmumap']))
    elif outpath != '':
        hmupath = os.path.join(outpath, "mumap-hdw", "hmumap.npz")
        if os.path.isfile(hmupath):
            muh = np.load(hmupath, allow_pickle=True)["hmu"]
            datain["hmumap"] = hmupath
        else:
            raise IOError('Invalid path to the hardware mu-map')
    else:
        log.error('the hardware mu-map is required first.')
        raise IOError('Could not find the hardware mu-map!')
    # ========================================================
    # -check if T1w image is available
    if not {'MRT1W#', 'T1nii', 'T1bc', 'T1N4'}.intersection(datain):
        log.error('no MR T1w images required for co-registration!')
        raise IOError('T1w image could not be obtained!')
    # ========================================================

    # -if the affine is not given,
    # -it will be generated by reconstructing PET image, with some or no corrections
    if not os.path.isfile(faff):
        # first recon pet to get the T1 aligned to it
        if petopt == 'qnt':
            # ---------------------------------------------
            # OPTION 1 (quantitative recon with all corrections using MR-based mu-map)
            # get UTE object mu-map (may not be in register with the PET data)
            mudic = obj_mumap(datain, Cnt, outpath=tmpdir, del_auxilary=del_auxilary)
            muo = mudic['im']
            # reconstruct PET image with UTE mu-map to which co-register T1w
            recout = mmrrec.osemone(datain, [muh, muo], hst, scanner_params, recmod=3, itr=itr,
                                    fwhm=0., fcomment=fcomment + '_QNT-UTE',
                                    outpath=os.path.join(outpath, 'PET',
                                                         'positioning'), store_img=True)
        elif petopt == 'nac':
            # ---------------------------------------------
            # OPTION 2 (recon without any corrections for scatter and attenuation)
            # reconstruct PET image with UTE mu-map to which co-register T1w
            muo = np.zeros(muh.shape, dtype=muh.dtype)
            recout = mmrrec.osemone(datain, [muh, muo], hst, scanner_params, recmod=1, itr=itr,
                                    fwhm=0., fcomment=fcomment + '_NAC',
                                    outpath=os.path.join(outpath, 'PET',
                                                         'positioning'), store_img=True)
        elif petopt == 'ac':
            # ---------------------------------------------
            # OPTION 3 (recon with attenuation correction only but no scatter)
            # reconstruct PET image with UTE mu-map to which co-register T1w
            mudic = obj_mumap(datain, Cnt, outpath=tmpdir, del_auxilary=del_auxilary)
            muo = mudic['im']

            recout = mmrrec.osemone(datain, [muh, muo], hst, scanner_params, recmod=1, itr=itr,
                                    fwhm=0., fcomment=fcomment + '_AC-UTE',
                                    outpath=os.path.join(outpath, 'PET',
                                                         'positioning'), store_img=True)

        fpet = recout.fpet
        mu_dct['fpet'] = fpet

        # ------------------------------
        if musrc == 'ute' and ute_name in datain and os.path.exists(datain[ute_name]):
            # change to NIfTI if the UTE sequence is in DICOM files (folder)
            if os.path.isdir(datain[ute_name]):
                fnew = os.path.basename(datain[ute_name])
                run([Cnt['DCM2NIIX'], '-f', fnew, datain[ute_name]])
                fute = glob.glob(os.path.join(datain[ute_name], fnew + '*nii*'))[0]
            elif os.path.isfile(datain[ute_name]):
                fute = datain[ute_name]

            # get the affine transformation
            if reg_tool == 'spm':
                regdct = nimpa.coreg_spm(fpet, fute,
                                         outpath=os.path.join(outpath, 'PET', 'positioning'))
            elif reg_tool == 'niftyreg':
                if not os.path.exists(Cnt['REGPATH']):
                    raise ValueError('e> no valid NiftyReg executable')
                regdct = nimpa.affine_niftyreg(
                    fpet,
                    fute,
                    outpath=os.path.join(outpath, 'PET', 'positioning'),
                    executable=Cnt['REGPATH'],
                    omp=multiprocessing.cpu_count() / 2,                 # pcomment=fcomment,
                    rigOnly=True,
                    affDirect=False,
                    maxit=5,
                    speed=True,
                    pi=50,
                    pv=50,
                    smof=0,
                    smor=0,
                    rmsk=True,
                    fmsk=True,
                    rfwhm=15.,                                           # pillilitres
                    rthrsh=0.05,
                    ffwhm=15.,                                           # pillilitres
                    fthrsh=0.05,
                    verbose=verbose)

            elif reg_tool == 'dipy':
                regdct = nimpa.affine_dipy(
                    fpet,
                    fute,
                    nbins=32,
                    metric='MI',
                    level_iters=[10000, 1000, 200],
                    sigmas=[3.0, 1.0, 0.0],
                    factors=[4, 2, 1],
                    outpath=os.path.join(outpath, 'PET', 'positioning'),
                    faffine=None,
                    pickname='ref',
                    fcomment='',
                    rfwhm=2.,
                    ffwhm=2.,
                    verbose=verbose)
            else:
                raise ValueError('unknown registration tool requested')

            faff_mrpet = regdct['faff']

        elif musrc == 'pct':

            ft1w = nimpa.pick_t1w(datain)

            if reg_tool == 'spm':
                regdct = nimpa.coreg_spm(fpet, ft1w,
                                         outpath=os.path.join(outpath, 'PET', 'positioning'))
            elif reg_tool == 'niftyreg':
                if not os.path.exists(Cnt['REGPATH']):
                    raise ValueError('e> no valid NiftyReg executable')
                regdct = nimpa.affine_niftyreg(
                    fpet,
                    ft1w,
                    outpath=os.path.join(outpath, 'PET', 'positioning'),
                    executable=Cnt['REGPATH'],
                    omp=multiprocessing.cpu_count() / 2,
                    rigOnly=True,
                    affDirect=False,
                    maxit=5,
                    speed=True,
                    pi=50,
                    pv=50,
                    smof=0,
                    smor=0,
                    rmsk=True,
                    fmsk=True,
                    rfwhm=15.,                                           # pillilitres
                    rthrsh=0.05,
                    ffwhm=15.,                                           # pillilitres
                    fthrsh=0.05,
                    verbose=verbose)

            elif reg_tool == 'dipy':
                regdct = nimpa.affine_dipy(
                    fpet,
                    ft1w,
                    nbins=32,
                    metric='MI',
                    level_iters=[10000, 1000, 200],
                    sigmas=[3.0, 1.0, 0.0],
                    factors=[4, 2, 1],
                    outpath=os.path.join(outpath, 'PET', 'positioning'),
                    faffine=None,
                    pickname='ref',
                    fcomment='',
                    rfwhm=2.,
                    ffwhm=2.,
                    verbose=verbose)

            else:
                raise ValueError('unknown registration tool requested')

            faff_mrpet = regdct['faff']

        else:
            raise IOError('Floating MR image not provided or is invalid.')

    else:
        faff_mrpet = faff
        regdct = {}
        if not os.path.isfile(fpet):
            raise IOError('e> the reference PET should be supplied with the affine.')

    # > output file name for the aligned mu-maps
    if musrc == 'pct':

        # > convert to mu-values before resampling to avoid artefacts with negative values
        nii = nib.load(datain['pCT'])
        img = nii.get_fdata(dtype=np.float32)
        img_mu = hu2mu(img)
        nii_mu = nib.Nifti1Image(img_mu, nii.affine)
        fflo = os.path.join(tmpdir, 'pct2mu-not-aligned.nii.gz')
        nib.save(nii_mu, fflo)

        freg = os.path.join(opth, 'pct2mu-aligned-' + fcomment + '.nii.gz')

    elif musrc == 'ute':
        freg = os.path.join(opth, 'UTE-res-tmp' + fcomment + '.nii.gz')
        if 'UTE' not in datain:
            fnii = 'converted-from-DICOM_'
            tstmp = nimpa.time_stamp(simple_ascii=True)
            # convert the DICOM mu-map images to nii
            if 'mumapDCM' not in datain:
                raise IOError('DICOM with the UTE mu-map are not given.')
            run([Cnt['DCM2NIIX'], '-f', fnii + tstmp, '-o', opth, datain['mumapDCM']])
            # piles for the T1w, pick one:
            fflo = glob.glob(os.path.join(opth, '*' + fnii + tstmp + '*.nii*'))[0]
        else:
            if os.path.isfile(datain['UTE']):
                fflo = datain['UTE']
            else:
                raise IOError('The provided NIfTI UTE path is not valid.')

    # > call the resampling routine to get the pCT/UTE in place
    if reg_tool == 'spm':
        nimpa.resample_spm(fpet, fflo, faff_mrpet, fimout=freg, del_ref_uncmpr=True,
                           del_flo_uncmpr=True, del_out_uncmpr=True)
    elif reg_tool == 'dipy':
        nimpa.resample_dipy(fpet, fflo, faff=faff_mrpet, fimout=freg)
    else:
        nimpa.resample_niftyreg(fpet, fflo, faff_mrpet, fimout=freg, executable=Cnt['RESPATH'],
                                verbose=verbose)

    # -get the NIfTI of registered image
    nim = nib.load(freg)
    A = nim.affine
    imreg = nim.get_fdata(dtype=np.float32)
    imreg = imreg[:, ::-1, ::-1]
    imreg = np.transpose(imreg, (2, 1, 0))

    # -convert to mu-values; sort out the file name too.
    if musrc == 'pct':
        mu = imreg
    elif musrc == 'ute':
        mu = np.float32(imreg) / 1e4
        # -remove the converted file from DICOMs
        os.remove(fflo)
    else:
        raise NameError('Confused o_O')

    # > get rid of negatives and nans
    mu[mu < 0] = 0
    mu[np.isnan(mu)] = 0

    # > return image dictionary with the image itself and other parameters
    mu_dct['im'] = mu
    mu_dct['affine'] = A
    mu_dct['faff'] = faff_mrpet

    if store or store_npy:
        nimpa.create_dir(opth)
        if faff == '':
            fname = fnm + '-aligned-to_t'\
                    + str(t0)+'-'+str(t1)+'_'+petopt.upper()\
                    + fcomment
        else:
            fname = fnm + '-aligned-to-given-affine' + fcomment
    if store_npy:
        fnp = os.path.join(opth, fname + ".npz")
        np.savez(fnp, mu=mu, A=A)
    if store:
        # > NIfTI
        fmu = os.path.join(opth, fname + '.nii.gz')
        nimpa.array2nii(mu[::-1, ::-1, :], A, fmu)
        mu_dct['fim'] = fmu

    if del_auxilary:
        os.remove(freg)

        if musrc == 'ute' and not os.path.isfile(faff):
            os.remove(fute)
        shutil.rmtree(tmpdir)

    return mu_dct



# ********************************************************************************
# GET HARDWARE MU-MAPS with positions and offsets
# --------------------------------------------------------------------------------


def hdr_mu(datain, Cnt):
    '''Get the headers from DICOM data file'''
    # pet one of the DICOM files of the mu-map
    if 'mumapDCM' in datain:
        files = glob.glob(os.path.join(datain['mumapDCM'], '*.dcm'))
        files.extend(glob.glob(os.path.join(datain['mumapDCM'], '*.DCM')))
        files.extend(glob.glob(os.path.join(datain['mumapDCM'], '*.ima')))
        files.extend(glob.glob(os.path.join(datain['mumapDCM'], '*.IMA')))
        dcmf = files[0]
    else:
        raise NameError('no DICOM or DICOM filed <CSA Series Header Info> found!')
    if os.path.isfile(dcmf):
        dhdr = dcm.read_file(dcmf)
    else:
        log.error('DICOM mMR mu-maps are not valid files!')
        return None
    # CSA Series Header Info
    if [0x29, 0x1020] in dhdr:
        csahdr = dhdr[0x29, 0x1020].value
        log.info('got CSA mu-map info from the DICOM header.')
    return csahdr, dhdr


def hmu_shape(hdr):
    # pegular expression to find the shape
    p = re.compile(r'(?<=:=)\s*\d{1,4}')
    # x: dim [1]
    i0 = hdr.find('matrix size[1]')
    i1 = i0 + hdr[i0:].find('\n')
    u = int(p.findall(hdr[i0:i1])[0])
    # x: dim [2]
    i0 = hdr.find('matrix size[2]')
    i1 = i0 + hdr[i0:].find('\n')
    v = int(p.findall(hdr[i0:i1])[0])
    # x: dim [3]
    i0 = hdr.find('matrix size[3]')
    i1 = i0 + hdr[i0:].find('\n')
    w = int(p.findall(hdr[i0:i1])[0])
    return w, v, u


def hmu_voxsize(hdr):
    # pegular expression to find the shape
    p = re.compile(r'(?<=:=)\s*\d{1,2}[.]\d{1,10}')
    # x: dim [1]
    i0 = hdr.find('scale factor (mm/pixel) [1]')
    i1 = i0 + hdr[i0:].find('\n')
    vx = float(p.findall(hdr[i0:i1])[0])
    # x: dim [2]
    i0 = hdr.find('scale factor (mm/pixel) [2]')
    i1 = i0 + hdr[i0:].find('\n')
    vy = float(p.findall(hdr[i0:i1])[0])
    # x: dim [3]
    i0 = hdr.find('scale factor (mm/pixel) [3]')
    i1 = i0 + hdr[i0:].find('\n')
    vz = float(p.findall(hdr[i0:i1])[0])
    return np.array([0.1 * vz, 0.1 * vy, 0.1 * vx])


def hmu_origin(hdr):
    # pegular expression to find the origin
    p = re.compile(r'(?<=:=)\s*\d{1,5}[.]\d{1,10}')
    # x: dim [1]
    i0 = hdr.find('$umap origin (pixels) [1]')
    i1 = i0 + hdr[i0:].find('\n')
    x = float(p.findall(hdr[i0:i1])[0])
    # x: dim [2]
    i0 = hdr.find('$umap origin (pixels) [2]')
    i1 = i0 + hdr[i0:].find('\n')
    y = float(p.findall(hdr[i0:i1])[0])
    # x: dim [3]
    i0 = hdr.find('$umap origin (pixels) [3]')
    i1 = i0 + hdr[i0:].find('\n')
    z = -float(p.findall(hdr[i0:i1])[0])
    return np.array([z, y, x])


def hmu_offset(hdr):
    # pegular expression to find the origin
    p = re.compile(r'(?<=:=)\s*\d{1,5}[.]\d{1,10}')
    if hdr.find('$origin offset') > 0:
        # x: dim [1]
        i0 = hdr.find('$origin offset (mm) [1]')
        i1 = i0 + hdr[i0:].find('\n')
        x = float(p.findall(hdr[i0:i1])[0])
        # x: dim [2]
        i0 = hdr.find('$origin offset (mm) [2]')
        i1 = i0 + hdr[i0:].find('\n')
        y = float(p.findall(hdr[i0:i1])[0])
        # x: dim [3]
        i0 = hdr.find('$origin offset (mm) [3]')
        i1 = i0 + hdr[i0:].find('\n')
        z = -float(p.findall(hdr[i0:i1])[0])
        return np.array([0.1 * z, 0.1 * y, 0.1 * x])
    else:
        return np.array([0.0, 0.0, 0.0])


def rd_hmu(fh):
    # --read hdr file--
    f = open(fh, 'r')
    hdr = f.read()
    f.close()
    # -----------------
    # pegular expression to find the file name
    p = re.compile(r'(?<=:=)\s*\w*[.]\w*')
    i0 = hdr.find('!name of data file')
    i1 = i0 + hdr[i0:].find('\n')
    fbin = p.findall(hdr[i0:i1])[0]
    # --read img file--
    f = open(os.path.join(os.path.dirname(fh), fbin.strip()), 'rb')
    im = np.fromfile(f, np.float32)
    f.close()
    # -----------------
    return hdr, im


def get_hmupos(datain, parts, Cnt, outpath=''):

    # ----- get positions from the DICOM list-mode file -----
    ihdr, csainfo = mmraux.hdr_lm(datain, Cnt)
    # pable position origin
    fi = csainfo.find(b'TablePositionOrigin')
    tpostr = csainfo[fi:fi + 200]
    tpo = re.sub(b'[^a-zA-Z0-9.\\-]', b'', tpostr).split(b'M')
    tpozyx = np.array([float(tpo[-1]), float(tpo[-2]), float(tpo[-3])]) / 10
    log.info('table position (z,y,x) (cm): {}'.format(tpozyx))
    # --------------------------------------------------------

    # ------- get positions from the DICOM mu-map file -------
    csamu, dhdr = hdr_mu(datain, Cnt)
    # > get the indices where the table offset may reside:
    idxs = [m.start() for m in re.finditer(b'GantryTableHomeOffset(?!_)', csamu)]
    # > loop over the indices and find those which are correct
    found_off = False
    for i in idxs:
        gtostr1 = csamu[i:i + 300]
        gtostr2 = re.sub(b'[^a-zA-Z0-9.\\-]', b'', gtostr1)
        # gantry table offset, through conversion of string to float
        gtoxyz = re.findall(b'(?<=M)-*[\\d]{1,4}\\.[\\d]{6,9}', gtostr2)
        gtozyx = np.float32(gtoxyz)[::-1] / 10
        if len(gtoxyz) > 3:
            log.warning('the gantry table offset got more than 3 entries detected--check needed.')
            gtozyx = gtozyx[-3:]
        if abs(gtozyx[0]) > 20 and abs(gtozyx[1]) < 20 and abs(gtozyx[2]) < 2:
            found_off = True
            break

    if found_off:
        log.info('gantry table offset (z,y,x) (cm): {}'.format(gtozyx))
    else:
        raise ValueError('Could not find the gantry table offset or the offset is unusual.')
    # --------------------------------------------------------

    # create the folder for hardware mu-maps
    if outpath == '':
        dirhmu = os.path.join(datain['corepath'], 'mumap-hdw')
    else:
        dirhmu = os.path.join(outpath, 'mumap-hdw')
    mmraux.create_dir(dirhmu)
    # get the reference nii image
    fref = os.path.join(dirhmu, 'hmuref.nii.gz')

    # ptart horizontal bed position
    p = re.compile(r'start horizontal bed position.*\d{1,3}\.*\d*')
    m = p.search(ihdr)
    fi = ihdr[m.start():m.end()].find('=')
    hbedpos = 0.1 * float(ihdr[m.start() + fi + 1:m.end()])

    # ptart vertical bed position
    p = re.compile(r'start vertical bed position.*\d{1,3}\.*\d*')
    m = p.search(ihdr)
    fi = ihdr[m.start():m.end()].find('=')
    vbedpos = 0.1 * float(ihdr[m.start() + fi + 1:m.end()])

    log.info('creating reference NIfTI image for resampling')
    B = np.diag(np.array([-10 * Cnt['SO_VXX'], 10 * Cnt['SO_VXY'], 10 * Cnt['SO_VXZ'], 1]))
    B[0, 3] = 10 * (.5 * Cnt['SO_IMX']) * Cnt['SO_VXX']
    B[1, 3] = 10 * (-.5 * Cnt['SO_IMY'] + 1) * Cnt['SO_VXY']
    B[2, 3] = 10 * ((-.5 * Cnt['SO_IMZ'] + 1) * Cnt['SO_VXZ'] + hbedpos)
    nimpa.array2nii(np.zeros((Cnt['SO_IMZ'], Cnt['SO_IMY'], Cnt['SO_IMX']), dtype=np.float32), B,
                    fref)

    # define a dictionary of all positions/offsets of hardware mu-maps
    hmupos = [None] * 5
    hmupos[0] = {
        'TabPosOrg': tpozyx, # prom DICOM of LM file
        'GanTabOff': gtozyx, # prom DICOM of mMR mu-map file
        'HBedPos': hbedpos,  # prom Interfile of LM file [cm]
        'VBedPos': vbedpos,  # prom Interfile of LM file [cm]
        'niipath': fref}

    # --------------------------------------------------------------------------
    # iteratively go through the mu-maps and add them as needed
    for i in parts:
        fh = os.path.join(Cnt['HMUDIR'], Cnt['HMULIST'][i - 1])
        # get the interfile header and binary data
        hdr, im = rd_hmu(fh)
        # pet shape, origin, offset and voxel size
        s = hmu_shape(hdr)
        im.shape = s
        # get the origin, offset and voxel size for the mu-map interfile data
        org = hmu_origin(hdr)
        off = hmu_offset(hdr)
        vs = hmu_voxsize(hdr)
        # corner voxel position for the interfile image data
        vpos = (-org * vs + off + gtozyx - tpozyx)
        # pdd to the dictionary
        hmupos[i] = {
            'vpos': vpos,
            'shape': s,   # prom interfile
            'iorg': org,  # prom interfile
            'ioff': off,  # prom interfile
            'ivs': vs,    # prom interfile
            'img': im,    # prom interfile
            'niipath': os.path.join(dirhmu, '_' + Cnt['HMULIST'][i - 1].split('.')[0] + '.nii.gz')}
        log.info('creating mu-map for: {}'.format(Cnt['HMULIST'][i - 1]))
        A = np.diag(np.append(10 * vs[::-1], 1))
        A[0, 0] *= -1
        A[0, 3] = 10 * (-vpos[2])
        A[1, 3] = -10 * ((s[1] - 1) * vs[1] + vpos[1])
        A[2, 3] = -10 * ((s[0] - 1) * vs[0] - vpos[0])
        nimpa.array2nii(im[::-1, ::-1, :], A, hmupos[i]['niipath'])

        # > resample using DIPY in nimpa
        fout = os.path.join(os.path.dirname(hmupos[0]['niipath']),
                            'r' + os.path.basename(hmupos[i]['niipath']).split('.')[0] + '.nii.gz')

        nimpa.resample_dipy(hmupos[0]['niipath'], hmupos[i]['niipath'], fimout=fout, intrp=1,
                            dtype_nifti=np.float32)

    return hmupos


def hdw_mumap(datain, hparts, params, outpath='', use_stored=False, del_interm=True):
    '''Get hardware mu-map components, including bed, coils etc.'''
    # two ways of passing Cnt are here decoded
    if 'Cnt' in params:
        Cnt = params['Cnt']
    else:
        Cnt = params

    if outpath != '':
        fmudir = os.path.join(outpath, 'mumap-hdw')
    else:
        fmudir = os.path.join(datain['corepath'], 'mumap-hdw')

    nimpa.create_dir(fmudir)

    # if requested to use the stored hardware mu_map get it from the path in datain
    if use_stored and "hmumap" in datain and os.path.isfile(datain["hmumap"]):
        if datain['hmumap'].endswith(('.nii', '.nii.gz')):
            dct = nimpa.getnii(datain['hmumap'], output='all')
            hmu = dct['im']
            A = dct['affine']
            fmu = datain['hmumap']
        elif datain["hmumap"].endswith(".npz"):
            arr = np.load(datain["hmumap"], allow_pickle=True)
            hmu, A, fmu = arr["hmu"], arr["A"], arr["fmu"]
            log.info('loaded hardware mu-map from file: {}'.format(datain['hmumap']))
            fnp = datain['hmumap']
    elif outpath and os.path.isfile(os.path.join(fmudir, "hmumap.npz")):
        fnp = os.path.join(fmudir, "hmumap.npz")
        arr = np.load(fnp, allow_pickle=True)
        hmu, A, fmu = arr["hmu"], arr["A"], arr["fmu"]
        datain['hmumap'] = fnp
    # otherwise generate it from the parts through resampling the high resolution CT images
    else:
        hmupos = get_hmupos(datain, hparts, Cnt, outpath=outpath)
        # just to get the dims, get the ref image
        nimo = nib.load(hmupos[0]['niipath'])
        A = nimo.affine
        imo = nimo.get_fdata(dtype=np.float32)
        imo[:] = 0

        for i in hparts:
            fin = os.path.join(
                os.path.dirname(hmupos[0]['niipath']),
                'r' + os.path.basename(hmupos[i]['niipath']).split('.')[0] + '.nii.gz')
            nim = nib.load(fin)
            mu = nim.get_fdata(dtype=np.float32)
            mu[mu < 0] = 0

            imo += mu

        hdr = nimo.header
        hdr['cal_max'] = np.max(imo)
        hdr['cal_min'] = np.min(imo)
        fmu = os.path.join(os.path.dirname(hmupos[0]['niipath']), 'hardware_umap.nii.gz')
        hmu_nii = nib.Nifti1Image(imo, A)
        nib.save(hmu_nii, fmu)

        hmu = np.transpose(imo[:, ::-1, ::-1], (2, 1, 0))

        # save the objects to numpy arrays
        fnp = os.path.join(fmudir, "hmumap.npz")
        np.savez(fnp, hmu=hmu, A=A, fmu=fmu)
        # ppdate the datain dictionary (assuming it is mutable)
        datain['hmumap'] = fnp

        if del_interm:
            for fname in glob.glob(os.path.join(fmudir, '_*.nii*')):
                os.remove(fname)
            for fname in glob.glob(os.path.join(fmudir, 'r_*.nii*')):
                os.remove(fname)

    # peturn image dictionary with the image itself and some other stats
    hmu_dct = {'im': hmu, 'fim': fmu, 'affine': A}
    if 'fnp' in locals():
        hmu_dct['fnp'] = fnp

    return hmu_dct


def rmumaps(datain, Cnt, t0=0, t1=0, use_stored=False):
    '''
    get the mu-maps for hardware and object and trim it axially for reduced rings case
    '''
    from niftypet.nipet.lm import mmrhist
    from niftypet.nipet.prj import mmrrec
    fcomment = '(R)'

    # get hardware mu-map
    if os.path.isfile(datain['hmumap']) and use_stored:
        muh = np.load(datain["hmumap"], allow_pickle=True)["hmu"]
        log.info('loaded hardware mu-map from file:\n{}'.format(datain['hmumap']))
    else:
        hmudic = hdw_mumap(datain, [1, 2, 4], Cnt)
        muh = hmudic['im']

    # get pCT mu-map if stored in numpy file and then exit, otherwise do all the processing
    if os.path.isfile(datain['mumapCT']) and use_stored:
        mup = np.load(datain["mumapCT"], allow_pickle=True)["mu"]
        muh = muh[2 * Cnt['RNG_STRT']:2 * Cnt['RNG_END'], :, :]
        mup = mup[2 * Cnt['RNG_STRT']:2 * Cnt['RNG_END'], :, :]
        return [muh, mup]

    # get UTE object mu-map (may be not in register with the PET data)
    if os.path.isfile(datain['mumapUTE']) and use_stored:
        muo, _ = np.load(datain['mumapUTE'], allow_pickle=True)
    else:
        mudic = obj_mumap(datain, Cnt, store=True)
        muo = mudic['im']

    if os.path.isfile(datain['pCT']):
        # reconstruct PET image with default settings to be used to alight pCT mu-map
        params = mmraux.get_mmrparams()
        Cnt_ = params['Cnt']
        txLUT_ = params['txLUT']
        axLUT_ = params['axLUT']

        # histogram for reconstruction with UTE mu-map
        hst = mmrhist.hist(datain, txLUT_, axLUT_, Cnt_, t0=t0, t1=t1)
        # reconstruct PET image with UTE mu-map to which co-register T1w
        recute = mmrrec.osemone(datain, [muh, muo], hst, params, recmod=3, itr=4, fwhm=0.,
                                store_img=True, fcomment=fcomment + '_QNT-UTE')
        # --- MR T1w
        if os.path.isfile(datain['T1nii']):
            ft1w = datain['T1nii']
        elif os.path.isfile(datain['T1bc']):
            ft1w = datain['T1bc']
        elif os.path.isdir(datain['MRT1W']):
            # create file name for the converted NIfTI image
            fnii = 'converted'
            run([Cnt['DCM2NIIX'], '-f', fnii, datain['T1nii']])
            ft1nii = glob.glob(os.path.join(datain['T1nii'], '*converted*.nii*'))
            ft1w = ft1nii[0]
        else:
            raise IOError('Disaster: no T1w image!')

        # putput for the T1w in register with PET
        ft1out = os.path.join(os.path.dirname(ft1w), 'T1w_r' + '.nii.gz')
        # pext file fo rthe affine transform T1w->PET
        faff = os.path.join(os.path.dirname(ft1w), fcomment + 'mr2pet_affine' + '.txt')
        # time.strftime('%d%b%y_%H.%M',time.gmtime())
        # > call the registration routine
        if os.path.isfile(Cnt['REGPATH']):
            cmd = [
                Cnt['REGPATH'], '-ref', recute.fpet, '-flo', ft1w, '-rigOnly', '-speeeeed', '-aff',
                faff, '-res', ft1out]
            if log.getEffectiveLevel() > logging.INFO:
                cmd.append('-voff')
            run(cmd)
        else:
            raise IOError('Path to registration executable is incorrect!')

        # pet the pCT mu-map with the above faff
        pmudic = pct_mumap(datain, txLUT_, axLUT_, Cnt, faff=faff, fpet=recute.fpet,
                           fcomment=fcomment)
        mup = pmudic['im']

        muh = muh[2 * Cnt['RNG_STRT']:2 * Cnt['RNG_END'], :, :]
        mup = mup[2 * Cnt['RNG_STRT']:2 * Cnt['RNG_END'], :, :]
        return [muh, mup]
    else:
        muh = muh[2 * Cnt['RNG_STRT']:2 * Cnt['RNG_END'], :, :]
        muo = muo[2 * Cnt['RNG_STRT']:2 * Cnt['RNG_END'], :, :]
        return [muh, muo]





# # ================================================================================
# # PSEUDO CT MU-MAP
# # --------------------------------------------------------------------------------


# def pct_mumap(datain, scanner_params, hst=None, t0=0, t1=0, itr=2, petopt='ac', faff='', fpet='',
#               fcomment='', outpath='', store_npy=False, store=False, verbose=False):
#     '''
#     GET THE MU-MAP from pCT IMAGE (which is in T1w space)
#     * the mu-map will be registered to PET which will be reconstructed for time frame t0-t1
#     * it f0 and t1 are not given the whole LM dataset will be reconstructed
#     * the reconstructed PET can be attenuation and scatter corrected or NOT using petopt
#     '''
#     if hst is None:
#         hst = []

#     # constants, transaxial and axial LUTs are extracted
#     Cnt = scanner_params['Cnt']

#     if not os.path.isfile(faff):
#         from niftypet.nipet.prj import mmrrec

#         # histogram the list data if needed
#         if not hst:
#             from niftypet.nipet.lm import mmrhist
#             hst = mmrhist(datain, scanner_params, t0=t0, t1=t1)

#     # get hardware mu-map
#     if datain.get("hmumap", "").endswith(".npz") and os.path.isfile(datain["hmumap"]):
#         muh = np.load(datain["hmumap"], allow_pickle=True)["hmu"]
#         (log.info if verbose else log.debug)('loaded hardware mu-map from file:\n{}'.format(
#             datain['hmumap']))
#     elif outpath:
#         hmupath = os.path.join(outpath, "mumap-hdw", "hmumap.npz")
#         if os.path.isfile(hmupath):
#             muh = np.load(hmupath, allow_pickle=True)["hmu"]
#             datain['hmumap'] = hmupath
#         else:
#             raise IOError('Invalid path to the hardware mu-map')
#     else:
#         log.error('The hardware mu-map is required first.')
#         raise IOError('Could not find the hardware mu-map!')

#     if not {'MRT1W#', 'T1nii', 'T1bc'}.intersection(datain):
#         log.error('no MR T1w images required for co-registration!')
#         raise IOError('Missing MR data')
#     # ----------------------------------

#     # output dictionary
#     mu_dct = {}
#     if not os.path.isfile(faff):
#         # first recon pet to get the T1 aligned to it
#         if petopt == 'qnt':
#             # ---------------------------------------------
#             # OPTION 1 (quantitative recon with all corrections using MR-based mu-map)
#             # get UTE object mu-map (may not be in register with the PET data)
#             mudic = obj_mumap(datain, Cnt)
#             muo = mudic['im']
#             # reconstruct PET image with UTE mu-map to which co-register T1w
#             recout = mmrrec.osemone(datain, [muh, muo], hst, scanner_params, recmod=3, itr=itr,
#                                     fwhm=0., fcomment=fcomment + '_qntUTE',
#                                     outpath=os.path.join(outpath, 'PET',
#                                                          'positioning'), store_img=True)
#         elif petopt == 'nac':
#             # ---------------------------------------------
#             # OPTION 2 (recon without any corrections for scatter and attenuation)
#             # reconstruct PET image with UTE mu-map to which co-register T1w
#             muo = np.zeros(muh.shape, dtype=muh.dtype)
#             recout = mmrrec.osemone(datain, [muh, muo], hst, scanner_params, recmod=1, itr=itr,
#                                     fwhm=0., fcomment=fcomment + '_NAC',
#                                     outpath=os.path.join(outpath, 'PET',
#                                                          'positioning'), store_img=True)
#         elif petopt == 'ac':
#             # ---------------------------------------------
#             # OPTION 3 (recon with attenuation correction only but no scatter)
#             # reconstruct PET image with UTE mu-map to which co-register T1w
#             mudic = obj_mumap(datain, Cnt, outpath=outpath)
#             muo = mudic['im']
#             recout = mmrrec.osemone(datain, [muh, muo], hst, scanner_params, recmod=1, itr=itr,
#                                     fwhm=0., fcomment=fcomment + '_AC',
#                                     outpath=os.path.join(outpath, 'PET',
#                                                          'positioning'), store_img=True)

#         fpet = recout.fpet
#         mu_dct['fpet'] = fpet

#         # ------------------------------
#         # get the affine transformation
#         ft1w = nimpa.pick_t1w(datain)
#         try:
#             regdct = nimpa.coreg_spm(fpet, ft1w,
#                                      outpath=os.path.join(outpath, 'PET', 'positioning'))
#         except Exception:
#             regdct = nimpa.affine_niftyreg(
#                 fpet,
#                 ft1w,
#                 outpath=os.path.join(outpath, 'PET', 'positioning'), # pcomment=fcomment,
#                 executable=Cnt['REGPATH'],
#                 omp=multiprocessing.cpu_count() / 2,
#                 rigOnly=True,
#                 affDirect=False,
#                 maxit=5,
#                 speed=True,
#                 pi=50,
#                 pv=50,
#                 smof=0,
#                 smor=0,
#                 rmsk=True,
#                 fmsk=True,
#                 rfwhm=15.,                                           # pillilitres
#                 rthrsh=0.05,
#                 ffwhm=15.,                                           # pillilitres
#                 fthrsh=0.05,
#                 verbose=verbose)

#         faff = regdct['faff']
#         # ------------------------------

#     # pCT file name
#     if outpath == '':
#         pctdir = os.path.dirname(datain['pCT'])
#     else:
#         pctdir = os.path.join(outpath, 'mumap-obj')
#     mmraux.create_dir(pctdir)
#     fpct = os.path.join(pctdir, 'pCT_r_tmp' + fcomment + '.nii.gz')

#     # > call the resampling routine to get the pCT in place
#     if os.path.isfile(Cnt['RESPATH']):
#         cmd = [
#             Cnt['RESPATH'], '-ref', fpet, '-flo', datain['pCT'], '-trans', faff, '-res', fpct,
#             '-pad', '0']
#         if log.getEffectiveLevel() > logging.INFO:
#             cmd.append('-voff')
#         run(cmd)
#     else:
#         log.error('path to resampling executable is incorrect!')
#         raise IOError('Incorrect path to executable!')

#     # get the NIfTI of the pCT
#     nim = nib.load(fpct)
#     A = nim.get_sform()
#     pct = nim.get_fdata(dtype=np.float32)
#     pct = pct[:, ::-1, ::-1]
#     pct = np.transpose(pct, (2, 1, 0))
#     # convert the HU units to mu-values
#     mu = hu2mu(pct)
#     # get rid of negatives
#     mu[mu < 0] = 0

#     # return image dictionary with the image itself and other parameters
#     mu_dct['im'] = mu
#     mu_dct['affine'] = A
#     mu_dct['faff'] = faff

#     if store:
#         # now save to numpy array and NIfTI in this folder
#         if outpath == '':
#             pctumapdir = os.path.join(datain['corepath'], 'mumap-obj')
#         else:
#             pctumapdir = os.path.join(outpath, 'mumap-obj')
#         mmraux.create_dir(pctumapdir)
#         # > Numpy
#         if store_npy:
#             fnp = os.path.join(pctumapdir, "mumap-pCT.npz")
#             np.savez(fnp, mu=mu, A=A)

#         # > NIfTI
#         fmu = os.path.join(pctumapdir, 'mumap-pCT' + fcomment + '.nii.gz')
#         nimpa.array2nii(mu[::-1, ::-1, :], A, fmu)
#         mu_dct['fim'] = fmu
#         datain['mumapCT'] = fmu

#     return mu_dct
