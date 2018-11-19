"""module for pipelined image reconstruction and analysis"""
__author__      = "Pawel Markiewicz"
__copyright__   = "Copyright 2018"
#-------------------------------------------------------------------------------

import numpy as np
import sys
import os
import scipy.ndimage as ndi
from subprocess import call

from niftypet import nimpa
from niftypet.nipet.prj import mmrrec
from niftypet.nipet import mmraux

#-------------------------------------------------------------------------------
def mmrchain(datain,        # all input data in a dictionary
            scanner_params, # all scanner parameters in one dictionary containing constants, transaxial and axial LUTs
            outpath='',     # output path for results
            frames=['fluid', [0,0]], # definition of time frames
            mu_h = [],      # hardware mu-map
            mu_o = [],      # object mu-map
            tAffine = [],   # affine transformations for the mu-map for each time frame separately
            
            itr=4,          # number of OSEM iterations
            fwhm=0.,        # Gaussian Smoothing FWHM
            recmod = -1,    # reconstruction mode: -1: undefined, chosen automatically. 3: attenuation and scatter correction, 1: attenuation correction only, 0: no correction (randoms only)
            histo=[],         # input histogram (from list-mode data).  if not given, it will be performed.

            trim=False,
            trim_scale=2,
            trim_interp=1,  # interpolation for upsampling used in PVC
            trim_memlim=True,   # reduced use of memory for machines with limited memory (makes it slower though)
            pvcroi=[],      # ROI used for PVC.  If undefined no PVC is performed. 
            psfkernel=[],
            pvcitr=5, 
            
            fcomment='',    # text comment used in the file name of generated image files
            ret_sinos=False,  # return prompt, scatter and randoms sinograms for each reconstruction
            store_img = True,
            store_img_intrmd=False,
            store_itr=[],   # store any reconstruction iteration in the list.  ignored if the list is empty.
            del_img_intrmd=False):


    from niftypet import nipet

    # decompose all the scanner parameters and constants
    Cnt   = scanner_params['Cnt']
    txLUT = scanner_params['txLUT']
    axLUT = scanner_params['axLUT']


    # -------------------------------------------------------------------------
    # FRAMES
    # check for the provided dynamic frames
    if isinstance(frames, list):
        # Can be given in three ways:
        # a 1D list (duration of each frame is listed)
        # a more concise 2D list (lists repetition and duration in each list entry)
        # or 'fluid', a 2D list with first entry 'fluid' meaning that the concecutive lists are start and end time [t0, t1].
        # for the last option the number of time frames is unlimited, provided that the t0 and t1 are within the acquisition times
        # Starting with the 'fluid':
        # if 'fluid'
        if isinstance(frames[0], basestring) and frames[0]=='fluid' and all([isinstance(t,list) and len(t)==2 for t in frames[1:]]):
            t_frms = frames[1:]
        # if 2D:
        elif all([isinstance(t,list) and len(t)==2 for t in frames]):
            # get total time and list of all time frames
            dfrms = mmraux.timings_from_list(frames)
            t_frms = dfrms['timings']
        # if 1D:
        elif all([isinstance(t,(int, np.int32, np.int16, np.int8, np.uint8, np.uint16, np.uint32)) for t in frames]):
            # get total time and list of all time frames
            dfrms = mmraux.timings_from_list(frames)
            t_frms = dfrms['timings']
        else:
            print 'e> osemdyn: frames definitions are not given in the correct list format: 1D [15,15,30,30,...] or 2D list [[2,15], [2,30], ...]'
    elif isinstance(frames, (np.ndarray, np.generic) ):
        dfrms = mmraux.timings_from_list( list(frames) )
        t_frms = dfrms['timings']
    else:
        print 'e> osemdyn: provided dynamic frames definitions are not in either Python list or nympy array.'
        raise TypeError('Wrong data type for dynamic frames')
    # number of dynamic time frames
    nfrm = len(t_frms)
    # -------------------------------------------------------------------------


    # -------------------------------------------------------------------------
    # create folders for results
    if outpath=='':     
        petdir = os.path.join(datain['corepath'], 'reconstructed')
        fmudir = os.path.join(datain['corepath'], 'mumap-obj')
        pvcdir = os.path.join(datain['corepath'], 'PRCL')
    else:               
        petdir = os.path.join(outpath, 'PET')
        fmudir = os.path.join(outpath, 'mumap-obj')
        pvcdir = os.path.join(outpath, 'PRCL')

    # folder for coregistered mu-maps (for motion compensation)
    fmureg = os.path.join( fmudir, 'registered')
    # folder for affine transformation MR/CT->PET
    petaff = os.path.join( petdir, 'faffine')

    # folder for reconstructed images (dynamic or static depending on number of frames).
    if nfrm>1:
        petimg = os.path.join(petdir, 'multiple-frames')
        pvcdir = os.path.join(pvcdir, 'multiple-frames')
    elif nfrm==1:
        petimg = os.path.join(petdir, 'single-frame')
        pvcdir = os.path.join(pvcdir, 'single-frame')
    else:
        print 'e> confused!'
        raise TypeError('Unrecognised time frames!')
    # create now the folder
    mmraux.create_dir(petimg)
    # create folder
    mmraux.create_dir(petdir)
    # -------------------------------------------------------------------------
    
    
    # -------------------------------------------------------------------------
    # MU-MAPS
    # get the mu-maps, if given.  otherwise will use blank mu-maps
    if tAffine:
        muod = nipet.img.mmrimg.obtain_image(mu_o, imtype='object mu-map', verbose=Cnt['VERBOSE'])
    else:
        muod = nipet.img.mmrimg.obtain_image(mu_o, Cnt=Cnt, imtype='object mu-map')
    muhd = nipet.img.mmrimg.obtain_image(mu_h, Cnt, imtype='hardware mu-map')

    # choose the mode of reconstruction based on the provided (or not) mu-maps
    if muod['exists'] and muhd['exists'] and recmod==-1:
        recmod = 3
    elif  (muod['exists'] or muhd['exists']) and recmod==-1:
        recmod = 1
        print 'w> partial mu-map:  scatter correction is switched off.'
    else:
        if recmod==-1:
            recmod = 0
            print 'w> no mu-map provided: scatter and attenuation corrections are switched off.'
    # -------------------------------------------------------------------------
    

    # output dictionary
    output = {}
    output['recmod'] = recmod
    output['frames'] = t_frms
    output['#frames'] = nfrm

    # if affine transformation is given the baseline mu-map in NIfTI file or dictionary has to be given
    if not tAffine:
        if Cnt['VERBOSE']: print 'i> using the provided mu-map the same way for all frames.'
    else:
        if len(tAffine)!=nfrm:
            print 'e> the number of affine transformations in the list has to be the same as the number of dynamic frames!'
            raise IndexError('Inconsistent number of frames.')
        elif not isinstance(tAffine, list):
            print 'e> tAffine has to be a list of either 4x4 numpy arrays of affine transformations or a list of file path strings!'
            raise IndexError('Expecting a list.')
        elif not 'fim' in muod:
            print 'e> when tAffine is given, the object mu-map has to be provided either as a dictionary or NIfTI file!'
            raise NameError('No path to object mu-map.')

        # check if all are file path strings to the existing files
        if all([isinstance(t, basestring) for t in tAffine]):
            if all([os.path.isfile(t) for t in tAffine]):
                # the internal list of affine transformations
                faff_frms = tAffine
                if Cnt['VERBOSE']: print 'i> using provided paths to affine transformations for each dynamic frame.'
            else:
                print 'e> not all provided paths are valid!'
                raise IOError('Wrong paths.')
        # check if all are numpy arrays
        elif all([isinstance(t, (np.ndarray, np.generic)) for t in tAffine]):
            # create the folder for dynamic affine transformations
            mmraux.create_dir(petaff)
            faff_frms = []
            for i in range(nfrm):
                fout = os.path.join(petaff, 'affine_frame('+str(i)+').txt')
                np.savetxt(fout, tAffine[i], fmt='%3.9f')
                faff_frms.append(fout)
            if Cnt['VERBOSE']: print 'i> using provided numpy arrays affine transformations for each dynamic frame.'
        else:
            raise StandardError('Affine transformations for each dynamic frame could not be established.')

        # -------------------------------------------------------------------------------------
        # get ref image for mu-map resampling
        # -------------------------------------------------------------------------------------
        if 'fmuref' in muod:
            fmuref = muod['fmuref']
            if Cnt['VERBOSE']: print 'i> reusing the reference mu-map from the object mu-map dictionary.'
        else:
            # create folder if doesn't exists
            mmraux.create_dir(fmudir)
            # ref file name
            fmuref = os.path.join(fmudir, 'muref.nii.gz')
            # ref affine
            B = nipet.img.mmrimg.image_affine(datain, Cnt, gantry_offset=False)
            # ref image (blank)
            im = np.zeros((Cnt['SO_IMZ'], Cnt['SO_IMY'], Cnt['SO_IMX']), dtype=np.float32)
            # store ref image
            nimpa.array2nii(im, B, fmuref)
            if Cnt['VERBOSE']: print 'i> generated a reference mu-map in', fmuref
        # -------------------------------------------------------------------------------------

        output['fmuref'] = fmuref
        output['faffine'] = faff_frms

    # output list of intermidiate file names for mu-maps and PET images (useful for dynamic imaging)
    if tAffine: output['fmureg'] = []
    if store_img_intrmd: output['fpeti'] = []

    # dynamic images in one numpy array
    dynim = np.zeros((nfrm, Cnt['SO_IMZ'], Cnt['SO_IMY'], Cnt['SO_IMY']), dtype=np.float32)
    #if asked, output only scatter+randoms sinogram for each frame
    if ret_sinos and itr>1 and recmod>2:
        dynrsn = np.zeros((nfrm, Cnt['NSN11'], Cnt['NSANGLES'], Cnt['NSBINS']), dtype=np.float32)
        dynssn = np.zeros((nfrm, Cnt['NSN11'], Cnt['NSANGLES'], Cnt['NSBINS']), dtype=np.float32)
        dynpsn = np.zeros((nfrm, Cnt['NSN11'], Cnt['NSANGLES'], Cnt['NSBINS']), dtype=np.float32)


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
        print 'i> dynamic frame times t0, t1:', t0, t1
        if not histo:
            hst = nipet.lm.mmrhist.hist(datain, txLUT, axLUT, Cnt, t0=t0, t1=t1)
        else:
            hst = histo
            print ''
            print 'i> using provided histogram'
            print ''
        if np.sum(hst['dhc'])>0.99*np.sum(hst['phc']):
            print '==============================================================================================='
            print 'w> the amount of random events is the greatest part of prompt events => omitting reconstruction'
            print '==============================================================================================='
            ifrmP = ifrm+1
            continue
        # --------------------
        # transform the mu-map if given the affine transformation for each frame
        if tAffine:
            # create the folder for aligned (registered for motion compensation) mu-maps
            mmraux.create_dir(fmureg)
            # the converted nii image resample to the reference size
            fmu = os.path.join(fmureg, 'mumap_dyn_frm'+str(ifrm)+fcomment+'.nii.gz')
            # command for resampling
            if os.path.isfile( Cnt['RESPATH'] ):
                cmd = [Cnt['RESPATH'],
                '-ref', fmuref,
                '-flo', muod['fim'],
                '-trans', faff_frms[ifrm],
                '-res', fmu,
                '-pad', '0']
                if not Cnt['VERBOSE']: cmd.append('-voff')
                call(cmd)
            else:
                print 'e> path to the executable for resampling is incorrect!'
                raise IOError('Incorrect NiftyReg (resampling) executable.')
            # get the new mu-map from the just resampled file
            muodct = nimpa.getnii(fmu, output='all')
            muo = muodct['im']
            A = muodct['affine']
            muo[muo<0] = 0
            output['fmureg'].append(fmu)
        else:
            muo = muod['im']
        #---------------------

        # output image file name
        if nfrm>1:
            frmno = '_frm'+str(ifrm)
        else:
            frmno = ''

        # run OSEM reconstruction of a single time frame
        recimg = mmrrec.osemone(datain, [muhd['im'], muo], 
                                hst, scanner_params,
                                recmod=recmod, itr=itr, fwhm=fwhm,
                                outpath=petimg,
                                frmno=frmno,
                                fcomment=fcomment+'_i',
                                store_img=store_img_intrmd,
                                store_itr=store_itr,
                                ret_sinos=ret_sinos)
        # form dynamic numpy array
        dynim[ifrm,:,:,:] = recimg.im
        if ret_sinos and itr>1 and recmod>2:
            dynpsn[ifrm,:,:,:] = hst['psino']
            dynssn[ifrm,:,:,:] = recimg.ssn 
            dynrsn[ifrm,:,:,:] = recimg.rsn

        if store_img_intrmd: output['fpeti'].append(recimg.fpet)
        if nfrm==1: output['tuple'] = recimg

    output['im'] = np.squeeze(dynim)
    if ret_sinos and itr>1 and recmod>2:
        output['sinos'] = {'psino':dynpsn, 'ssino':dynssn, 'rsino':dynrsn}

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
        petu = nimpa.trimim(
            dynim,
            affine=nipet.img.mmrimg.image_affine(datain, Cnt),
            scale=trim_scale,
            int_order=trim_interp,
            outpath=petimg,
            fname = fnm,
            fcomment=fcomment,
            store_img_intrmd=store_img_intrmd,
            memlim=trim_memlim,
            verbose=Cnt['VERBOSE']
        )

        output.update({'trimmed': { 'im':petu['im'],
                                    'fpet':petu['fimi'],
                                    'affine':petu['affine']}}
        )
    # ----------------------------------------------------------------------


    # ----------------------------------------------------------------------
    #run PVC if requested and required input given
    if pvcroi:
        if not os.path.isfile(datain['T1lbl']):
            print 'e> no label image from T1 parcellations and/or ROI definitions!'
            raise StandardError('No ROIs')
        else:
            # get the PSF kernel for PVC
            if not psfkernel:
                psfkernel = nimpa.psf_measured(scanner='mmr', scale=trim_scale)
            else:
                if isinstance(psfkernel, (np.ndarray, np.generic)) and psfkernel.shape!=(3, 17):
                    print 'e> the PSF kernel has to be an numpy array with the shape of (3, 17)!'
                    raise IndexError('PSF: wrong shape or not a matrix')
        # perform PVC for each time frame
        froi2 = []
        fpvc = []
        dynpvc = np.zeros(petu['im'].shape, dtype=np.float32)
        for i in range(ifrmP,nfrm):
            # transform the parcelations (ROIs) if given the affine transformation for each frame
            if not tAffine:
                print 'w> affine transformation are not provided: will generate for the time frame.'
                faffpvc = ''
                #raise StandardError('No affine transformation')
            else:
                faffpvc = faff_frms[i]
            # chose file name of individual PVC images
            if nfrm>1:
                fcomment_pvc = '_frm'+str(i)+fcomment
            else:
                fcomment_pvc = fcomment
            #============================
            # perform PVC
            petpvc_dic = nimpa.pvc_iyang(
                petu['fimi'][i],
                datain,
                Cnt,
                pvcroi,
                psfkernel,
                itr=pvcitr,
                faff=faffpvc,
                fcomment=fcomment_pvc,
                outpath=pvcdir,
                store_img=store_img_intrmd)
            #============================
            if nfrm>1: 
                dynpvc[i,:,:,:] = petpvc_dic['im']
            else:
                dynpvc = petpvc_dic['im']

            froi2.append(petpvc_dic['froi'])
            fpvc.append(petpvc_dic['fpet'])
        # update output dictionary
        output.update({'impvc':dynpvc, 'froi':froi2, 'fpvc':fpvc})
    # ----------------------------------------------------------------------

    if store_img:
        # description for saving NIFTI image
        # attenuation number: if only bed present then it is 0.5
        attnum =  ( 1*muhd['exists'] + 1*muod['exists'] ) / 2.
        descrip =    'alg=osem'                     \
                    +';att='+str(attnum*(recmod>0)) \
                    +';sct='+str(1*(recmod>1))      \
                    +';spn='+str(Cnt['SPN'])        \
                    +';sub=14'                      \
                    +';itr='+str(itr)               \
                    +';fwhm='+str(fwhm)             \
                    +';nfrm='+str(nfrm)

        # squeeze the not needed dimensions
        dynim = np.squeeze(dynim)

        # NIfTI file name for the full PET image (single or multiple frame)
        
        # save the image to NIfTI file
        if nfrm==1:
            t0 = hst['t0']
            t1 = hst['t1']
            if t1==t0:
                t0 = 0
                t1 = hst['dur']
            fpet = os.path.join( petimg, os.path.basename(recimg.fpet)[:8]+'_itr-'+str(itr)+'_t-'+str(t0)+'-'+str(t1)+'sec' )
            fpeto = fpet+fcomment+'.nii.gz'
            nimpa.prc.array2nii( dynim[::-1,::-1,:], recimg.affine, fpeto, descrip=descrip)
        else:
            fpet = os.path.join( petimg, os.path.basename(recimg.fpet)[:8]+'_itr-'+str(itr)+'_nfrm-'+str(nfrm) )
            fpeto = fpet+fcomment+'.nii.gz'
            nimpa.prc.array2nii( dynim[:,::-1,::-1,:], recimg.affine, fpeto, descrip=descrip)

        # get output file names for trimmed/PVC images
        if trim:
            # folder for trimmed and dynamic
            pettrim = os.path.join( petimg, 'trimmed')
            # make folder
            mmraux.create_dir(pettrim)
            # trimming scale added to NIfTI descritoption
            descrip_trim = descrip + ';trim_scale='+str(trim_scale)
            # file name for saving the trimmed image
            fpetu = os.path.join(pettrim, os.path.basename(fpet) + '_trimmed-upsampled-scale-'+str(trim_scale))
            # in case of PVC
            if pvcroi:
                # itertive Yang (iY) added to NIfTI descritoption
                descrip_pvc = descrip_trim + ';pvc=iY'
                # file name for saving the PVC NIfTI image
                fpvc = fpetu + '_PVC' + fcomment + '.nii.gz'
                output['trimmed']['fpvc'] = fpvc

            # update the trimmed image file name
            fpetu += fcomment+'.nii.gz'
            # store the file name in the output dictionary
            output['trimmed']['fpet'] = fpetu

        output['fpet'] = fpeto

        # save images
        if nfrm==1:
            if trim:
                nimpa.prc.array2nii( petu['im'][::-1,::-1,:], petu['affine'], fpetu, descrip=descrip_trim)
            if pvcroi:
                nimpa.prc.array2nii( dynpvc[::-1,::-1,:], petu['affine'], fpvc, descrip=descrip_pvc)
        elif nfrm>1:
            if trim:
                nimpa.prc.array2nii( petu['im'][:,::-1,::-1,:], petu['affine'], fpetu, descrip=descrip_trim)
            if pvcroi:
                nimpa.prc.array2nii( dynpvc[:,::-1,::-1,:], petu['affine'], fpvc, descrip=descrip_pvc)


    if del_img_intrmd:
        if pvcroi:
            for fi in fpvc:
                os.remove(fi)
        if trim:
            for fi in petu['fimi']:
                os.remove(fi)




    return output 