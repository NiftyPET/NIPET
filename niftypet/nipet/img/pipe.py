"""module for pipelined image reconstruction and analysis"""
import logging
import os
from numbers import Integral
from subprocess import call
from textwrap import dedent

import numpy as np

from niftypet import nimpa

from ..lm import dynamic_timings
from ..lm.mmrhist import mmrhist
from ..prj import mmrrec
from . import obtain_image
from .mmrimg import image_affine

log = logging.getLogger(__name__)


def mmrchain(
    datain,                 # all input data in a dictionary
    scanner_params,         # all scanner parameters in one dictionary
                            # containing constants, transaxial and axial
                            # LUTs.
    outpath=None,           # output path for results
    fout=None,              # full file name (any folders and extensions are disregarded)
    frames=None,            # definition of time frames, default: ['fluid', [0, 0]]
    mu_h=None,              # hardware mu-map.
    mu_o=None,              # object mu-map.
    tAffine=None,           # affine transformations for the mu-map for
                            # each time frame separately.
    itr=4,                  # number of OSEM iterations
    fwhm=0.,                # Gaussian Post-Smoothing FWHM
    psf=None,               # Resolution Modelling
    recmod=-1,              # reconstruction mode: -1: undefined, chosen
                            # automatically. 3: attenuation and scatter
                            # correction, 1: attenuation correction
                            # only, 0: no correction (randoms only).
    histo=None,             # input histogram (from list-mode data);
                            # if not given, it will be performed.
    decay_ref_time=None,    # decay corrects relative to the reference
                            # time provided; otherwise corrects to the scan
                            # start time.
    trim=False,
    trim_scale=2,
    trim_interp=0,          # interpolation for upsampling used in PVC
    trim_memlim=True,       # reduced use of memory for machines
                            # with limited memory (slow though)
    trim_store=True,
    pvcroi=None,            # ROI used for PVC.  If undefined no PVC
                            # is performed.
    pvcreg_tool='niftyreg', # the registration tool used in PVC
    store_rois=False,       # stores the image of PVC ROIs
                            # as defined in pvcroi.
    pvcpsf=None,
    pvcitr=5,
    fcomment='',            # text comment used in the file name of
                            # generated image files
    ret_sinos=False,        # return prompt, scatter and randoms
                            # sinograms for each reconstruction
    ret_histo=False,        # return histogram (LM processing output) for
                            # each image frame
    store_img=True,
    store_img_intrmd=False,
    store_itr=None,         # store any reconstruction iteration in
                            # the list.  ignored if the list is empty.
    del_img_intrmd=False,
):
    if frames is None:
        frames = ['fluid', [0, 0]]
    if mu_h is None:
        mu_h = []
    if mu_o is None:
        mu_o = []
    if pvcroi is None:
        pvcroi = []
    if pvcpsf is None:
        pvcpsf = []
    if store_itr is None:
        store_itr = []

    # decompose all the scanner parameters and constants
    Cnt = scanner_params['Cnt']

    # -------------------------------------------------------------------------
    # HISOTGRAM PRECEEDS FRAMES
    if histo is not None and 'psino' in histo:
        frames = ['fluid', [histo['t0'], histo['t1']]]
    else:
        histo = None
        log.warning(
            'the given histogram does not contain a prompt sinogram--will generate a histogram.')

    # FRAMES
    # check for the provided dynamic frames
    if isinstance(frames, list):
        # Can be given in three ways:
        # * a 1D list (duration of each frame is listed)
        # * a more concise 2D list--repetition and duration lists in
        #   each entry.  Must start with the 'def' entry.
        # * a 2D list with fluid timings: must start with the string
        #   'fluid' or 'timings'.  a 2D list with consecutive lists
        #   describing start and end of the time frame, [t0, t1];
        #   The number of time frames for this option is unlimited,
        #   provided the t0 and t1 are within the acquisition times.

        # 2D starting with entry 'fluid' or 'timings'
        if (isinstance(frames[0], str) and frames[0] in ('fluid', 'timings')
                and all(isinstance(t, list) and len(t) == 2 for t in frames[1:])):
            t_frms = frames[1:]
        # if 2D definitions, starting with entry 'def':
        elif (isinstance(frames[0], str) and frames[0] == 'def'
              and all(isinstance(t, list) and len(t) == 2 for t in frames[1:])):
            # get total time and list of all time frames
            t_frms = dynamic_timings(frames)['timings'][1:]
        # if 1D:
        elif all(isinstance(t, Integral) for t in frames):
            # get total time and list of all time frames
            t_frms = dynamic_timings(frames)['timings'][1:]
        else:
            log.error('osemdyn: frames definitions are not given in the correct list format:'
                      ' 1D [15,15,30,30,...] or 2D list [[2,15], [2,30], ...]')
    else:
        log.error(
            'provided dynamic frames definitions are incorrect (should be a list of definitions).')
        raise TypeError('Wrong data type for dynamic frames')
    # number of dynamic time frames
    nfrm = len(t_frms)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # create folders for results
    if outpath is None:
        petdir = os.path.join(datain['corepath'], 'reconstructed')
        fmudir = os.path.join(datain['corepath'], 'mumap-obj')
        pvcdir = os.path.join(datain['corepath'], 'PRCL')
    else:
        petdir = os.path.join(outpath, 'PET')
        fmudir = os.path.join(outpath, 'mumap-obj')
        pvcdir = os.path.join(outpath, 'PRCL')

    if fout is not None:
        # > get rid of folders
        fout = os.path.basename(fout)
        # > get rid of extension
        fout = fout.split('.')[0]

    # folder for co-registered mu-maps (for motion compensation)
    fmureg = os.path.join(fmudir, 'registered')
    # folder for affine transformation MR/CT->PET
    petaff = os.path.join(petdir, 'faffine')

    # folder for reconstructed images (dynamic or static depending on number of frames).
    if nfrm > 1:
        petimg = os.path.join(petdir, 'multiple-frames')
        pvcdir = os.path.join(pvcdir, 'multiple-frames')
    elif nfrm == 1:
        petimg = os.path.join(petdir, 'single-frame')
        pvcdir = os.path.join(pvcdir, 'single-frame')
    else:
        raise TypeError('Unrecognised/confusing time frames!')
    # create now the folder
    nimpa.create_dir(petimg)
    # create folder
    nimpa.create_dir(petdir)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # MU-MAPS
    # get the mu-maps, if given;  otherwise will use blank mu-maps.
    if tAffine is not None:
        muod = obtain_image(mu_o, imtype='object mu-map')
    else:
        muod = obtain_image(mu_o, Cnt=Cnt, imtype='object mu-map')

    # hardware mu-map
    muhd = obtain_image(mu_h, Cnt, imtype='hardware mu-map')

    # choose the mode of reconstruction based on the provided (or not) mu-maps
    if muod['exists'] and muhd['exists'] and recmod == -1:
        recmod = 3
    elif (muod['exists'] or muhd['exists']) and recmod == -1:
        recmod = 1
        log.warning('partial mu-map:  scatter correction is switched off.')
    else:
        if recmod == -1:
            recmod = 0
            log.warning(
                'no mu-map provided: scatter and attenuation corrections are switched off.')
    # -------------------------------------------------------------------------

    # import pdb; pdb.set_trace()

    # output dictionary
    output = {}
    output['recmod'] = recmod
    output['frames'] = t_frms
    output['#frames'] = nfrm

    # if affine transformation is given
    # the baseline mu-map in NIfTI file or dictionary has to be given
    if tAffine is None:
        log.info('using the provided mu-map the same way for all frames.')
    else:
        if len(tAffine) != nfrm:
            raise ValueError("the number of affine transformations in the list"
                             " has to be the same as the number of dynamic frames")
        elif not isinstance(tAffine, list):
            raise ValueError("tAffine has to be a list of either 4x4 numpy arrays"
                             " of affine transformations or a list of file path strings")
        elif 'fim' not in muod:
            raise NameError("when tAffine is given, the object mu-map has to be"
                            " provided either as a dictionary or NIfTI file")

        # check if all are file path strings to the existing files
        if all(isinstance(t, str) for t in tAffine):
            if all(os.path.isfile(t) for t in tAffine):
                # the internal list of affine transformations
                faff_frms = tAffine
                log.info('using provided paths to affine transformations for each dynamic frame.')
            else:
                raise IOError('not all provided paths are valid!')
        # check if all are numpy arrays
        elif all(isinstance(t, (np.ndarray, np.generic)) for t in tAffine):
            # create the folder for dynamic affine transformations
            nimpa.create_dir(petaff)
            faff_frms = []
            for i in range(nfrm):
                fout_ = os.path.join(petaff, 'affine_frame(' + str(i) + ').txt')
                np.savetxt(fout_, tAffine[i], fmt='%3.9f')
                faff_frms.append(fout_)
            log.info('using provided numpy arrays affine transformations for each dynamic frame.')
        else:
            raise ValueError(
                'Affine transformations for each dynamic frame could not be established.')

        # -------------------------------------------------------------------------------------
        # get ref image for mu-map resampling
        # -------------------------------------------------------------------------------------
        if 'fmuref' in muod:
            fmuref = muod['fmuref']
            log.info('reusing the reference mu-map from the object mu-map dictionary.')
        else:
            # create folder if doesn't exists
            nimpa.create_dir(fmudir)
            # ref file name
            fmuref = os.path.join(fmudir, 'muref.nii.gz')
            # ref affine
            B = image_affine(datain, Cnt, gantry_offset=False)
            # ref image (blank)
            im = np.zeros((Cnt['SO_IMZ'], Cnt['SO_IMY'], Cnt['SO_IMX']), dtype=np.float32)
            # store ref image
            nimpa.array2nii(im, B, fmuref)
            log.info('generated a reference mu-map in:\n{}'.format(fmuref))
        # -------------------------------------------------------------------------------------

        output['fmuref'] = fmuref
        output['faffine'] = faff_frms

    # output list of intermediate file names for mu-maps and PET images
    # (useful for dynamic imaging)
    if tAffine is not None: output['fmureg'] = []

    if store_img_intrmd:
        output['fpeti'] = []
        if fwhm > 0:
            output['fsmoi'] = []

    # > number of3D  sinograms
    if Cnt['SPN'] == 1:
        snno = Cnt['NSN1']
    elif Cnt['SPN'] == 11:
        snno = Cnt['NSN11']
    else:
        raise ValueError('unrecognised span: {}'.format(Cnt['SPN']))

    # dynamic images in one numpy array
    dynim = np.zeros((nfrm, Cnt['SO_IMZ'], Cnt['SO_IMY'], Cnt['SO_IMY']), dtype=np.float32)
    # if asked, output only scatter+randoms sinogram for each frame
    if ret_sinos and itr > 1 and recmod > 2:
        dynmsk = np.zeros((nfrm, Cnt['NSEG0'], Cnt['NSANGLES'], Cnt['NSBINS']), dtype=np.float32)
        dynrsn = np.zeros((nfrm, snno, Cnt['NSANGLES'], Cnt['NSBINS']), dtype=np.float32)
        dynssn = np.zeros((nfrm, snno, Cnt['NSANGLES'], Cnt['NSBINS']), dtype=np.float32)
        dynpsn = np.zeros((nfrm, snno, Cnt['NSANGLES'], Cnt['NSBINS']), dtype=np.float32)

    # > returning dictionary of histograms if requested
    if ret_histo:
        hsts = {}

    # import pdb; pdb.set_trace()

    # starting frame index with reasonable prompt data
    ifrmP = 0
    # iterate over frame index
    for ifrm in range(nfrm):
        # start time of a current (ifrm-th) dynamic frame
        t0 = int(t_frms[ifrm][0])
        # end time of a current (ifrm-th) dynamic frame
        t1 = int(t_frms[ifrm][1])
        # --------------
        # check if there is enough prompt data to do a reconstruction
        # --------------
        log.info('dynamic frame times t0={}, t1={}:'.format(t0, t1))
        if histo is None:
            hst = mmrhist(datain, scanner_params, t0=t0, t1=t1)
        else:
            hst = histo
            log.info(
                dedent('''\
                ------------------------------------------------------
                using provided histogram
                ------------------------------------------------------'''))

        if ret_histo:
            hsts[str(t0) + '-' + str(t1)] = hst

        if np.sum(hst['dhc']) > 0.99 * np.sum(hst['phc']):
            log.warning(
                dedent('''\
                ===========================================================================
                amount of randoms is the greater part of prompts => omitting reconstruction
                ==========================================================================='''))
            ifrmP = ifrm + 1
            continue
        # --------------------
        # transform the mu-map if given the affine transformation for each frame
        if tAffine is not None:
            # create the folder for aligned (registered for motion compensation) mu-maps
            nimpa.create_dir(fmureg)
            # the converted nii image resample to the reference size
            fmu = os.path.join(fmureg, 'mumap_dyn_frm' + str(ifrm) + fcomment + '.nii.gz')
            # command for resampling
            if os.path.isfile(Cnt['RESPATH']):
                cmd = [
                    Cnt['RESPATH'], '-ref', fmuref, '-flo', muod['fim'], '-trans', faff_frms[ifrm],
                    '-res', fmu, '-pad', '0']
                if log.getEffectiveLevel() > log.INFO:
                    cmd.append('-voff')
                call(cmd)
            else:
                raise IOError('Incorrect path to NiftyReg (resampling) executable.')
            # get the new mu-map from the just resampled file
            muodct = nimpa.getnii(fmu, output='all')
            muo = muodct['im']
            muo[muo < 0] = 0
            output['fmureg'].append(fmu)
        else:
            muo = muod['im']
        # ---------------------

        # output image file name
        if nfrm > 1:
            frmno = '_frm' + str(ifrm)
        else:
            frmno = ''

        # run OSEM reconstruction of a single time frame
        recimg = mmrrec.osemone(datain, [muhd['im'], muo], hst, scanner_params,
                                decay_ref_time=decay_ref_time, recmod=recmod, itr=itr, fwhm=fwhm,
                                psf=psf, outpath=petimg, frmno=frmno, fcomment=fcomment + '_i',
                                store_img=store_img_intrmd, store_itr=store_itr, fout=fout,
                                ret_sinos=ret_sinos)

        # form dynamic Numpy array
        if fwhm > 0:
            dynim[ifrm, :, :, :] = recimg.imsmo
        else:
            dynim[ifrm, :, :, :] = recimg.im

        if ret_sinos and itr > 1 and recmod > 2:
            dynpsn[ifrm, :, :, :] = np.squeeze(hst['psino'])
            dynssn[ifrm, :, :, :] = np.squeeze(recimg.ssn)
            dynrsn[ifrm, :, :, :] = np.squeeze(recimg.rsn)
            dynmsk[ifrm, :, :, :] = np.squeeze(recimg.amsk)

        if store_img_intrmd:
            output['fpeti'].append(recimg.fpet)
            if fwhm > 0:
                output['fsmoi'].append(recimg.fsmo)

        if nfrm == 1: output['tuple'] = recimg

    output['im'] = np.squeeze(dynim)

    if ret_sinos and itr > 1 and recmod > 2:
        output['sinos'] = {
            'psino': np.squeeze(dynpsn), 'ssino': np.squeeze(dynssn), 'rsino': np.squeeze(dynrsn),
            'amask': np.squeeze(dynmsk)}

    if ret_histo:
        output['hst'] = hsts

    # ----------------------------------------------------------------------
    # trim the PET image
    # images have to be stored for PVC
    if pvcroi: store_img_intrmd = True
    if trim:
        # create file name
        if 'lm_dcm' in datain:
            fnm = os.path.basename(datain['lm_dcm'])[:20]
        elif 'lm_ima' in datain:
            fnm = os.path.basename(datain['lm_ima'])[:20]
        # trim PET and upsample
        petu = nimpa.imtrimup(dynim, affine=image_affine(datain, Cnt), scale=trim_scale,
                              int_order=trim_interp, outpath=petimg, fname=fnm, fcomment=fcomment,
                              fcomment_pfx=fout + '_', store_img=trim_store,
                              store_img_intrmd=store_img_intrmd, memlim=trim_memlim,
                              verbose=log.getEffectiveLevel())

        output.update({
            'trimmed': {'im': petu['im'], 'fpet': petu['fimi'], 'affine': petu['affine']}})
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # run PVC if requested and required input given
    if pvcroi:
        if not os.path.isfile(datain['T1lbl']):
            raise Exception('No labels and/or ROIs image definitions found!')
        else:
            # get the PSF kernel for PVC
            if not pvcpsf:
                pvcpsf = nimpa.psf_measured(scanner='mmr', scale=trim_scale)
            else:
                if (
                    isinstance(pvcpsf, (np.ndarray, np.generic)) and
                    pvcpsf.shape != (3, 2 * Cnt['RSZ_PSF_KRNL'] + 1)
                ):  # yapf: disable
                    raise ValueError(
                        'the PSF kernel has to be an numpy array with the shape of ({},{})'.format(
                            3, 2 * Cnt['RSZ_PSF_KRNL'] + 1))

        # > file names for NIfTI images of PVC ROIs and PVC corrected PET
        froi = []
        fpvc = []

        # > perform PVC for each time frame
        dynpvc = np.zeros(petu['im'].shape, dtype=np.float32)
        for i in range(ifrmP, nfrm):
            # transform the parcellations (ROIs) if given the affine transformation for each frame
            if tAffine is None:
                log.warning(
                    'affine transformation are not provided: will generate for the time frame.')
                faffpvc = None
                # raise StandardError('No affine transformation')
            else:
                faffpvc = faff_frms[i]

            # chose file name of individual PVC images
            if nfrm > 1:
                fcomment_pvc = '_frm' + str(i) + fcomment
            else:
                fcomment_pvc = fcomment
            # ===========================
            # perform PVC
            petpvc_dic = nimpa.pvc_iyang(petu['fimi'][i], datain, Cnt, pvcroi, pvcpsf,
                                         tool=pvcreg_tool, itr=pvcitr, faff=faffpvc,
                                         fcomment=fcomment_pvc, outpath=pvcdir,
                                         store_rois=store_rois, store_img=store_img_intrmd)
            # ===========================
            if nfrm > 1:
                dynpvc[i, :, :, :] = petpvc_dic['im']
            else:
                dynpvc = petpvc_dic['im']
            fpvc.append(petpvc_dic['fpet'])

            if store_rois: froi.append(petpvc_dic['froi'])

        # > update output dictionary
        output.update({'impvc': dynpvc})
        output['fprc'] = petpvc_dic['fprc']
        output['imprc'] = petpvc_dic['imprc']

        if store_img_intrmd: output.update({'fpvc': fpvc})
        if store_rois: output.update({'froi': froi})
    # ----------------------------------------------------------------------

    if store_img:
        # description for saving NIFTI image
        # attenuation number: if only bed present then it is 0.5
        attnum = (1 * muhd['exists'] + 1 * muod['exists']) / 2.
        descrip = (f"alg=osem"
                   f";att={attnum*(recmod>0)}"
                   f";sct={1*(recmod>1)}"
                   f";spn={Cnt['SPN']}"
                   f";sub=14"
                   f";itr={itr}"
                   f";fwhm={fwhm}"
                   f";psf={psf}"
                   f";nfrm={nfrm}")

        # squeeze the not needed dimensions
        dynim = np.squeeze(dynim)

        # NIfTI file name for the full PET image (single or multiple frame)

        # save the image to NIfTI file
        if nfrm == 1:
            t0 = hst['t0']
            t1 = hst['t1']
            if t1 == t0:
                t0 = 0
                t1 = hst['dur']
            # > --- file naming and saving ---
            if fout is None:
                fpet = os.path.join(
                    petimg,
                    os.path.basename(recimg.fpet)[:8] + f'_t-{t0}-{t1}sec_itr-{itr}')
                fpeto = f"{fpet}{fcomment}.nii.gz"
            else:
                fpeto = os.path.join(petimg, os.path.basename(fout) + '.nii.gz')

            nimpa.prc.array2nii(dynim[::-1, ::-1, :], recimg.affine, fpeto, descrip=descrip)
            # > --- ---
        else:
            if fout is None:
                fpet = os.path.join(petimg,
                                    os.path.basename(recimg.fpet)[:8] + f'_nfrm-{nfrm}_itr-{itr}')
                fpeto = f"{fpet}{fcomment}.nii.gz"
            else:
                fpeto = os.path.join(petimg, os.path.basename(fout) + f'_nfrm-{nfrm}.nii.gz')

            nimpa.prc.array2nii(dynim[:, ::-1, ::-1, :], recimg.affine, fpeto, descrip=descrip)

        output['fpet'] = fpeto

        # get output file names for trimmed/PVC images
        if trim:
            # folder for trimmed and dynamic
            pettrim = os.path.join(petimg, 'trimmed')
            # make folder
            nimpa.create_dir(pettrim)
            # trimming scale added to NIfTI descritoption
            descrip_trim = f'{descrip};trim_scale={trim_scale}'
            # file name for saving the trimmed image
            if fout is None:
                fpetu = os.path.join(
                    pettrim,
                    os.path.basename(fpet) + f'_recon-trimmed-upsampled-scale-{trim_scale}')
            else:
                fpetu = os.path.join(
                    pettrim,
                    os.path.basename(fout) + f'_recon-trimmed-upsampled-scale-{trim_scale}')
            # in case of PVC
            if pvcroi:
                # itertive Yang (iY) added to NIfTI descritoption
                descrip_pvc = f'{descrip_trim};pvc=iY'
                # file name for saving the PVC NIfTI image
                fpvc = f"{fpetu}_PVC{fcomment}.nii.gz"
                output['trimmed']['fpvc'] = fpvc

            # update the trimmed image file name
            fpetu += f'{fcomment}.nii.gz'
            # store the file name in the output dictionary
            output['trimmed']['fpet'] = fpetu

        # save images
        if nfrm == 1:
            if trim:
                nimpa.prc.array2nii(petu['im'][::-1, ::-1, :], petu['affine'], fpetu,
                                    descrip=descrip_trim)
            if pvcroi:
                nimpa.prc.array2nii(dynpvc[::-1, ::-1, :], petu['affine'], fpvc,
                                    descrip=descrip_pvc)
        elif nfrm > 1:
            if trim:
                nimpa.prc.array2nii(petu['im'][:, ::-1, ::-1, :], petu['affine'], fpetu,
                                    descrip=descrip_trim)
            if pvcroi:
                nimpa.prc.array2nii(dynpvc[:, ::-1, ::-1, :], petu['affine'], fpvc,
                                    descrip=descrip_pvc)

    if del_img_intrmd:
        if pvcroi:
            for fi in fpvc:
                os.remove(fi)
        if trim:
            for fi in petu['fimi']:
                os.remove(fi)

    return output
