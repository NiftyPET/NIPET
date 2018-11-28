"""Image reconstruction from raw PET data"""
__author__      = "Pawel Markiewicz"
__copyright__   = "Copyright 2018"
#------------------------------------------------------------------------------

import numpy as np
import random
import sys
import os
import scipy.ndimage as ndi
from collections import namedtuple

import petprj

from niftypet.nipet.img import mmrimg
from niftypet.nipet import mmrnorm
from niftypet.nipet import mmraux
from niftypet import nimpa

# for isotope info
import resources

#reconstruction mode:
# 0 - no attenuation  and  no scatter
# 1 - attenuation  and   no scatter
# 2 - attenuation and scatter given as input parameter
# 3 - attenuation  and  scatter
recModeStr = ['_noatt_nosct_', '_nosct_', '_noatt_', '_', '_ute_']


# fwhm in [mm]
def fwhm2sig(fwhm, Cnt):
    return (0.1*fwhm/Cnt['SZ_VOXY']) / (2*(2*np.log(2))**.5)


#=========================================================================
# OSEM RECON
#-------------------------------------------------------------------------
def get_subsets14(n, params):
    ''' Define the n-th subset out of 14 in the transaxial projection space
    '''
    Cnt = params['Cnt']
    txLUT = params['txLUT']

    # just for check of sums (have to be equal for all subsets to make them balanced)
    aisum = np.sum(txLUT['msino'], axis=0)
    # number of subsets
    N = 14
    # projections per subset
    P = Cnt['NSANGLES']/N
    # the remaining projections which have to be spread over the N subsets with a given frequency
    fs = N/float(P-N)
    # generate sampling pattern for subsets up to N out of P
    sp = np.array([np.arange(i,Cnt['NSANGLES'],P) for i in range(N)])
    # ======================================
    S = np.zeros((N,P),dtype=np.int16)
    # ======================================
    # sum of sino angle projections
    totsum = np.zeros(N, dtype=np.int32)
    # iterate subset (which is also the angle iterator within block b)
    for s in range(N):
        # list of sino angular indexes for a given subset
        si = []
        #::::: iterate sino blocks.  This bit may be unnecessary, it can be taken directly from sp array
        for b in range(N):
            #--angle index within a sino block depending on subset s
            ai = (s+b)%N
            #--angle index for whole sino
            sai = sp[ai, b]
            si.append(sai)
            totsum[s] += aisum[sai]
        #:::::
        # deal with the remaining part, ie, P-N per block
        rai = np.int16( np.floor( np.arange(s,2*N,fs)[:4]%N ) )
        for i in range(P-N):
            sai = sp[-1,rai[i]]+i+1
            totsum[s] += aisum[sai]
            si.append(sai)
        # print si
        S[s] = np.array((si))

    # get the projection bin index for transaxial gpu sinos
    tmsk = txLUT['msino']>0
    Smsk = -1*np.ones(tmsk.shape, dtype=np.int32)
    Smsk[tmsk] = range(Cnt['Naw'])

    iprj = Smsk[:,S[n]]
    iprj = iprj[iprj>=0]

    # n=0; plot(S[n,:-4],ones(14), '*'); plot(S[n,-4:],ones(4), 'o')

    # Smsk = -1*np.ones(tmsk.shape, dtype=np.int32)
    # q=-1*ones(Cnt['Naw'])
    # q[iprj] = 3
    # Smsk[tmsk] = q

    return iprj, S
#---------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------#
#=== OSEM image reconstruction with several modes (with/without scatter and/or attenuation correction) ===#
def osemone(datain, mumaps, hst, scanner_params,
            recmod=3, itr=4, fwhm=0., mask_radious=29.,
            sctsino=np.array([]),
            outpath='',
            store_img=False, frmno='', fcomment='',
            store_itr=[],
            ret_sinos=False, emmskS=False, randsino = None, normcomp = None):


    #---------- sort out OUTPUT ------------
    #-output file name for the reconstructed image, initially assume n/a
    fout = 'n/a'
    if store_img or store_itr:
        if outpath=='':
            opth = os.path.join( datain['corepath'], 'reconstructed' )
        else:
            opth = outpath
        mmraux.create_dir(opth)
    #----------

    # Get particular scanner parameters: Constants, transaxial and axial LUTs
    Cnt   = scanner_params['Cnt']
    txLUT = scanner_params['txLUT']
    axLUT = scanner_params['axLUT']

    import time
    from niftypet import nipet
    # from niftypet.nipet.sct import mmrsct
    # from niftypet.nipet.prj import mmrhist

    if Cnt['VERBOSE']: print 'i> reconstruction in mode', recmod

    # get object and hardware mu-maps
    muh, muo = mumaps

    # get the GPU version of the image dims
    mus = mmrimg.convert2dev(muo+muh, Cnt)

    if Cnt['SPN']==1:
        snno = Cnt['NSN1']
    elif Cnt['SPN']==11:
        snno = Cnt['NSN11']

    # remove gaps from the prompt sino
    psng = mmraux.remgaps(hst['psino'], txLUT, Cnt)

    #=========================================================================
    # GET NORM
    #-------------------------------------------------------------------------
    if normcomp == None:
        ncmp, _ = mmrnorm.get_components(datain, Cnt)
    else:
        ncmp = normcomp
        print 'w> using user-defined normalisation components'
    nsng = mmrnorm.get_sinog(datain, hst, axLUT, txLUT, Cnt, normcomp=ncmp)
    #=========================================================================

    #=========================================================================
    # ATTENUATION FACTORS FOR COMBINED OBJECT AND BED MU-MAP
    #-------------------------------------------------------------------------
    # combine attenuation and norm together depending on reconstruction mode
    if recmod==0:
        asng = np.ones(psng.shape, dtype=np.float32)
    else:
        asng = np.zeros(psng.shape, dtype=np.float32)
        petprj.fprj(asng, mus, txLUT, axLUT, np.array([-1], dtype=np.int32), Cnt, 1)
    # combine attenuation and normalisation
    ansng = asng*nsng
    #=========================================================================

    #=========================================================================
    # Randoms
    #-------------------------------------------------------------------------
    if isinstance(randsino, np.ndarray):
        rsino = randsino
        rsng = mmraux.remgaps(randsino, txLUT, Cnt)
    else:
        rsino, snglmap = nipet.lm.mmrhist.rand(hst['fansums'], txLUT, axLUT, Cnt)
        rsng = mmraux.remgaps(rsino, txLUT, Cnt)
    #=========================================================================

    #=========================================================================
    # SCAT
    #-------------------------------------------------------------------------
    if recmod==2:
        if sctsino.size>0:
            ssng = mmraux.remgaps(sctsino, txLUT, Cnt)
        elif sctsino.size==0 and os.path.isfile(datain['em_crr']):
            emd = nimpa.getnii(datain['em_crr'])
            ssn, sssr, amsk = nipet.sct.mmrsct.vsm(mumaps, emd['im'], datain, hst, rsino, 0.1, txLUT, axLUT, Cnt)
            ssng = mmraux.remgaps(ssn, txLUT, Cnt)
        else:
            print 'e> no emission image available for scatter estimation!  check if it''s present or the path is correct.'
            sys.exit()
    else:
        ssng = np.zeros(rsng.shape, dtype=rsng.dtype)
    #=========================================================================

    if Cnt['VERBOSE']:
        print ''
        print '>------ OSEM (', itr,  ') -------'
    #------------------------------------
    Sn = 14 # number of subsets
    #-get one subset to get number of projection bins in a subset
    Sprj, s = get_subsets14(0,scanner_params)
    Nprj = len(Sprj)
    #-init subset array and sensitivity image for a given subset
    sinoTIdx = np.zeros((Sn, Nprj+1), dtype=np.int32)
    #-init sensitivity images for each subset
    imgsens = np.zeros((Sn, Cnt['SZ_IMY'], Cnt['SZ_IMX'], Cnt['SZ_IMZ']), dtype=np.float32)
    for n in range(Sn):
        sinoTIdx[n,0] = Nprj #first number of projection for the given subset
        sinoTIdx[n,1:], s = get_subsets14(n,scanner_params)
        # sensitivity image
        petprj.bprj(imgsens[n,:,:,:], ansng[sinoTIdx[n,1:],:], txLUT, axLUT,  sinoTIdx[n,1:], Cnt )
    #-------------------------------------

    #-mask for reconstructed image.  anything outside it is set to zero
    msk = mmrimg.get_cylinder(Cnt, rad=mask_radious, xo=0, yo=0, unival=1, gpu_dim=True)>0.9

    #-init image
    img = np.ones((Cnt['SZ_IMY'], Cnt['SZ_IMX'], Cnt['SZ_IMZ']), dtype=np.float32)

    #-decay correction
    lmbd = np.log(2)/resources.riLUT[Cnt['ISOTOPE']]['thalf']
    if 't0' in hst and 'dur' in hst:
        dcycrr = np.exp(lmbd*hst['t0'])*lmbd*hst['dur'] / (1-np.exp(-lmbd*hst['dur']))
        # apply quantitative correction to the image
        qf = ncmp['qf'] / resources.riLUT[Cnt['ISOTOPE']]['BF'] / float(hst['dur'])
        qf_loc = ncmp['qf_loc']
    else:
        dcycrr = 1
        qf = 1
        qf_loc = 1

    #-affine matrix for the reconstructed images
    B = mmrimg.image_affine(datain, Cnt)

    #-time it
    stime = time.time()
    #=========================================================================
    # OSEM RECONSTRUCTION
    #-------------------------------------------------------------------------
    for k in range(itr):
        if Cnt['VERBOSE']:
            print ''
            print '--------------- itr-{}/{} ---------------'.format(k,itr)
        petprj.osem(img, msk, psng, rsng, ssng, nsng, asng, imgsens, txLUT, axLUT, sinoTIdx, Cnt)
        if np.nansum(img)<0.1:
            print '---------------------------------------------------------------------'
            print 'w> it seems there is not enough true data to render reasonable image.'
            print '---------------------------------------------------------------------'
            #img[:]=0
            itr = k
            break
        if recmod>=3 and ( ((k<itr-1) and (itr>1)) ): # or (itr==1)
            sct_time = time.time()
            ssn, sssr, amsk = nipet.sct.mmrsct.vsm(
                mumaps,
                mmrimg.convert2e7(img, Cnt),
                datain,
                hst,
                rsino,
                scanner_params,
                prcntScl=0.1,
                emmsk=emmskS)
            ssng = mmraux.remgaps(ssn, txLUT, Cnt)
            if Cnt['VERBOSE']: print 'i> scatter time:', (time.time() - sct_time)
        # save images during reconstruction if requested
        if store_itr and k in store_itr:
            im = mmrimg.convert2e7(img * (dcycrr*qf*qf_loc), Cnt)
            fout =  os.path.join(opth, os.path.basename(datain['lm_bf'])[:8] \
                + frmno +'_t'+str(hst['t0'])+'-'+str(hst['t1'])+'sec' \
                +'_itr'+str(k)+fcomment+'_inrecon.nii.gz')
            nimpa.array2nii( im[::-1,::-1,:], B, fout)


    if Cnt['VERBOSE']: print 'i> recon time:', (time.time() - stime)
    #=========================================================================

    
    if Cnt['VERBOSE']: 
        print 'i> applying decay correction', dcycrr
        print 'i> applying quantification factor', qf, 'to the whole image for the frame duration of :', hst['dur']
    
    img *= dcycrr * qf * qf_loc #additional factor for making it quantitative in absolute terms (derived from measurements)

    #---- save images -----
    #-first convert to standard mMR image size
    im = mmrimg.convert2e7(img, Cnt)

    #-description text to NIfTI
    #-attenuation number: if only bed present then it is 0.5
    attnum =  ( 1*(np.sum(muh)>0.5)+1*(np.sum(muo)>0.5) ) / 2.
    descrip =   'alg=osem'+ \
                ';sub=14'+ \
                ';att='+str(attnum*(recmod>0))+ \
                ';sct='+str(1*(recmod>1))+ \
                ';spn='+str(Cnt['SPN'])+ \
                ';itr='+str(itr) +\
                ';fwhm='+str(fwhm) +\
                ';t0='+str(hst['t0']) +\
                ';t1='+str(hst['t1']) +\
                ';dur='+str(hst['dur']) +\
                ';qf='+str(qf)

    if fwhm>0:
        im = ndi.filters.gaussian_filter(im, fwhm2sig(fwhm, Cnt), mode='mirror')
    if store_img:
        fout =  os.path.join(opth, os.path.basename(datain['lm_bf'])[:8] \
                + frmno +'_t'+str(hst['t0'])+'-'+str(hst['t1'])+'sec' \
                +'_itr'+str(itr)+fcomment+'.nii.gz')
        if Cnt['VERBOSE']: print 'i> saving image to: ', fout
        nimpa.array2nii( im[::-1,::-1,:], B, fout, descrip=descrip)


    # returning:
    # (0) E7 image [can be smoothed];
    # (1) file name of saved E7 image
    # (2) [optional] scatter sino
    # (3) [optional] single slice rebinned scatter
    # (4) [optional] mask for scatter scaling based on attenuation data
    # (5) [optional] random sino
    # if ret_sinos and recmod>=3:
    #     recout = namedtuple('recout', 'im, fpet, ssn, sssr, amsk, rsn')
    #     recout.im   = im
    #     recout.fpet = fout
    #     recout.ssn  = ssn
    #     recout.sssr = sssr
    #     recout.amsk = amsk
    #     recout.rsn  = rsino
    # else:
    #     recout = namedtuple('recout', 'im, fpet')
    #     recout.im   = im
    #     recout.fpet = fout
    if ret_sinos and recmod>=3 and itr>1:
        RecOut = namedtuple('RecOut', 'im, fpet, affine, ssn, sssr, amsk, rsn')
        recout = RecOut(im, fout, B, ssn, sssr, amsk, rsino)
    else:
        RecOut = namedtuple('RecOut', 'im, fpet, affine')
        recout = RecOut(im, fout, B)
        
    return recout










#===============================================================================
# EMML
# def emml(   datain, mumaps, hst, txLUT, axLUT, Cnt, 
#             recmod=3, itr=10, fwhm=0., mask_radious=29., store_img=True, ret_sinos=False, sctsino = None, randsino = None, normcomp = None):

#     #subsets (when not used)
#     sbs = np.array([-1], dtype=np.int32)

#     # get object and hardware mu-maps
#     muh, muo = mumaps

#     # get the GPU version of the image dims
#     mus = mmrimg.convert2dev(muo+muh, Cnt)

#     # remove gaps from the prompt sinogram
#     psng = mmraux.remgaps(hst['psino'], txLUT, Cnt)

#     #=========================================================================
#     # GET NORM
#     #-------------------------------------------------------------------------
#     if normcomp == None:
#         ncmp, _ = mmrnorm.get_components(datain, Cnt)
#     else:
#         ncmp = normcomp
#         print 'w> using user-defined normalisation components'
#     nrmsng = mmrnorm.get_sinog(datain, hst, axLUT, txLUT, Cnt, normcomp=ncmp)
#     #=========================================================================

    
#     #=========================================================================
#     # Randoms
#     #-------------------------------------------------------------------------
#     if randsino == None:
#         rsino, snglmap = mmrhist.rand(hst['fansums'], txLUT, axLUT, Cnt)
#         rsng = mmraux.remgaps(rsino, txLUT, Cnt)
#     else:
#         rsino = randsino
#         rsng = mmraux.remgaps(randsino, txLUT, Cnt)
#     #=========================================================================


#     #=========================================================================
#     # ATTENUATION FACTORS FOR COMBINED OBJECT AND BED MU-MAP
#     #-------------------------------------------------------------------------
#     # combine attenuation and norm together depending on reconstruction mode
#     if recmod==0:
#         asng = np.ones(psng.shape, dtype=np.float32)
#     else:
#         asng = np.zeros(psng.shape, dtype=np.float32)
#         petprj.fprj(asng, mus, txLUT, axLUT, sbs, Cnt, 1)
#     attnrmsng = asng*nrmsng
#     #=========================================================================

    
#     #=========================================================================
#     # SCATTER and the additive term
#     #-------------------------------------------------------------------------
#     if recmod==2:
#         if sctsino != None:
#             # remove the gaps from the provided scatter sinogram
#             ssng = mmraux.remgaps(sctsino, txLUT, Cnt)
#         elif sctsino == None and os.path.isfile(datain['em_crr']):
#             # estimate scatter from already reconstructed and corrected emission image
#             emd = nimpa.prc.getnii(datain['em_crr'], Cnt)
#             ssn, sssr, amsk = mmrsct.vsm(mumaps, emd['im'], datain, hst, rsn, 0.1, txLUT, axLUT, Cnt)
#             ssng = mmraux.remgaps(ssn, txLUT, Cnt)
#         else:
#             print 'e> no emission image available for scatter estimation!  check if it''s present or the path is correct.'
#             sys.exit()
#     else:
#         ssng = np.zeros(rsng.shape, dtype=rsng.dtype)
#     # form the additive term
#     rssng = (rsng + ssng) / attnrmsng
#     #=========================================================================

#     #mask for reconstructed image
#     msk = mmrimg.get_cylinder(Cnt, rad=mask_radious, xo=0, yo=0, unival=1, gpu_dim=True)>0.9
#     # estimated image
#     imrec = np.ones((Cnt['SZ_IMY'], Cnt['SZ_IMX'], Cnt['SZ_IMZ']), dtype=np.float32)
#     # backprj image
#     bim = np.zeros((Cnt['SZ_IMY'], Cnt['SZ_IMX'], Cnt['SZ_IMZ']), dtype=np.float32)
#     # Get sensitivity image by backprojection
#     sim = np.zeros((Cnt['SZ_IMY'], Cnt['SZ_IMX'], Cnt['SZ_IMZ']), dtype=np.float32)
#     petprj.bprj(sim, attnrmsng, txLUT, axLUT, sbs, Cnt)
#     #init estimate sino
#     esng = np.zeros((Cnt['Naw'], Cnt['NSN11']), dtype=np.float32)


#     for k in range(itr):
#         print '>--------- ITERATION', k, '-----------<'
#         esng[:] = 0
#         petprj.fprj(esng, imrec, txLUT, axLUT, sbs, Cnt, 0)
#         # esng *= attnrmsng
#         esng += (rssng+ssng)
#         # crr = attnrmsng*(psng/esng)
#         crr = psng/esng
#         bim[:] = 0
#         petprj.bprj(bim, crr, txLUT, axLUT,  sbs, Cnt)
#         bim /= sim
#         imrec *= msk*bim
#         imrec[np.isnan(imrec)] = 0

#         if recmod>=3 and ( ((k<itr-1)and(itr>1))):
#             sct_time = time.time()
#             ssn, sssr, amsk = mmrsct.vsm(mumaps, mmrimg.convert2e7(img, Cnt), datain, hst, rsn, scanner_params, prcntScl=0.1, emmsk=emmskS)
#             ssng = mmraux.remgaps(ssn, txLUT, Cnt) / attnrmsng
#             if Cnt['VERBOSE']: print 'i> scatter time:', (time.time() - sct_time)

#     # decay correction
#     lmbd = np.log(2)/resources.riLUT[Cnt['ISOTOPE']]['thalf']
#     dcycrr = np.exp(lmbd*hst['t0'])*lmbd*hst['dur'] / (1-np.exp(-lmbd*hst['dur']))
#     # apply quantitative correction to the image
#     qf = ncmp['qf'] / resources.riLUT[Cnt['ISOTOPE']]['BF'] / float(hst['dur'])
#     if Cnt['VERBOSE']: print 'i> applying quantification factor', qf, 'to the whole image for the frame duration of :', hst['dur']
#     imrec *= dcycrr * qf * 0.205 #additional factor for making it quantitative in absolute terms (derived from measurements)

#     # convert to standard mMR image size
#     im = mmrimg.convert2e7(imrec, Cnt)

#     if fwhm>0:
#         im = ndi.filters.gaussian_filter(im, fwhm2sig(fwhm, Cnt), mode='mirror')

#     #save images
#     B = mmrimg.image_affine(datain, Cnt)
#     fout = ''

#     if store_img:
#         # description text to NIfTI
#         # attenuation number: if only bed present then it is 0.5
#         attnum =  ( 1*(np.sum(muh)>0.5)+1*(np.sum(muo)>0.5) ) / 2.
#         descrip =   'alg=emml'+ \
#                     ';sub=0'+ \
#                     ';att='+str(attnum*(recmod>0))+ \
#                     ';sct='+str(1*(recmod>1))+ \
#                     ';spn='+str(Cnt['SPN'])+ \
#                     ';itr='+str(itr)+ \
#                     ';fwhm='+str(fwhm) +\
#                     ';t0='+str(hst['t0']) +\
#                     ';t1='+str(hst['t1']) +\
#                     ';dur='+str(hst['dur']) +\
#                     ';qf='+str(qf)
#         fout =  os.path.join(datain['corepath'], os.path.basename(datain['lm_dcm'])[:8]+'_emml_'+str(itr)+'.nii.gz')
#         nimpa.array2nii( im[::-1,::-1,:], B, fout, descrip=descrip)
            
#     if ret_sinos and recmod>=3 and itr>1:
#         RecOut = namedtuple('RecOut', 'im, fpet, affine, ssn, sssr, amsk, rsn')
#         recout = RecOut(im, fout, B, ssn, sssr, amsk, rsn)
#     else:
#         RecOut = namedtuple('RecOut', 'im, fpet, affine')
#         recout = RecOut(im, fout, B)
    
#     return recout





#=============================================================================
# OSEM

# def osem14(datain, mumaps, hst, txLUT, axLUT, Cnt,
#             recmod=3, itr=4, fwhm=0., mask_radious=29.):

#     muh, muo = mumaps
#     mus = mmrimg.convert2dev(muo+muh, Cnt)
    
#     if Cnt['SPN']==1:
#         snno = Cnt['NSN1']
#     elif Cnt['SPN']==11:
#         snno = Cnt['NSN11']

#     #subsets (when not used)
#     sbs = np.array([-1], dtype=np.int32)

#     # remove gaps from the prompt sino
#     psng = mmraux.remgaps(hst['psino'], txLUT, Cnt)

#     #=========================================================================
#     # GET NORM
#     #-------------------------------------------------------------------------
#     nrmsng = mmrnorm.get_sinog(datain, hst, axLUT, txLUT, Cnt)
#     #=========================================================================

#     #=========================================================================
#     # RANDOMS ESTIMATION
#     #-------------------------------------------------------------------------
#     rsino, snglmap = mmrhist.rand(hst['fansums'], txLUT, axLUT, Cnt)
#     rndsng = mmraux.remgaps(rsino, txLUT, Cnt)
#     #=========================================================================

#     #=========================================================================
#     # FORM THE ADDITIVE TERM
#     #-------------------------------------------------------------------------
#     if recmod==0 or recmod==1 or recmod==3 or recmod==4:
#         rssng = rndsng
#     elif recmod==2:
#         if os.path.isfile(datain['em_crr']):
#             emd = nimpa.getnii(datain['em_crr'])
#             ssn, sssr, amsk = mmrsct.vsm(mumaps, emd['im'], datain, hst, rsino, 0.1, txLUT, axLUT, Cnt)
#             rssng = rndsng + mmraux.remgaps(ssn, txLUT, Cnt)
#         else:
#             print 'e> no emission image availble for scatter estimation!  check if it''s present or the path is correct.'
#             sys.exit()
#     #=========================================================================

#     #=========================================================================
#     # ATTENUATION FACTORS FOR COMBINED OBJECT AND BED MU-MAP
#     #-------------------------------------------------------------------------
#     # combine attenuation and norm together depending on reconstruction mode
#     if recmod==0 or recmod==2:
#         attnrmsng = nrmsng
#     else:
#         attnrmsng = np.zeros(psng.shape, dtype=np.float32)
#         petprj.fprj(attnrmsng, mus, txLUT, axLUT, sbs, Cnt, 1)
#         attnrmsng *= nrmsng
#     #=========================================================================

#     #mask for reconstructed image
#     rcnmsk = mmrimg.get_cylinder(Cnt, rad=mask_radious, xo=0, yo=0, unival=1, gpu_dim=True)
#     #-------------------------------------------------------------------------
#     # number of subsets
#     Sn = 14
#     # get one subset to get number of projection bins in a subset
#     Sprj, s = get_subsets14(0,txLUT,Cnt)
#     # init subset array and sensitivity image for a given subset
#     sinoTIdx = np.zeros((Sn, len(Sprj)), dtype=np.int32)
#     sim = np.zeros((Sn, Cnt['SZ_IMY'], Cnt['SZ_IMX'], Cnt['SZ_IMZ']), dtype=np.float32)
#     for n in range(Sn):
#         sinoTIdx[n,:], s = get_subsets14(n,txLUT,Cnt)
#         petprj.bprj(sim[n,:,:,:], attnrmsng, txLUT, axLUT, sinoTIdx[n,:], Cnt)
#     #--------------------------------------------------------------------------

#     # estimated image
#     xim = np.ones((Cnt['SZ_IMY'], Cnt['SZ_IMX'], Cnt['SZ_IMZ']), dtype=np.float32)
#     # backprj image
#     bim = np.ones((Cnt['SZ_IMY'], Cnt['SZ_IMX'], Cnt['SZ_IMZ']), dtype=np.float32)
#     # init scatter sino (zeros)
#     ssng  = np.zeros((Cnt['Naw'],   snno), dtype=np.float32)
#     # sinogram subset mask
#     sbmsk = np.zeros((txLUT['Naw'], snno), dtype=np.bool)
#     # estimated sinogram (forward model)
#     esng  = np.zeros((txLUT['Naw'], snno), dtype=np.float32)

#     for k in range(itr):
#         # randomly go through subsets + ssng
#         sn = range(Sn)
#         random.shuffle(sn)
#         s = 0
#         for n in sn:
#             print ' '
#             print k, '>--------- SUBSET', s, n, '-----------'
#             s+=1
#             sbmsk[:] = False
#             sbmsk[sinoTIdx[n,:],:] = True
#             esng[:] = 0
#             petprj.fprj(esng, xim, txLUT, axLUT, sinoTIdx[n,:], Cnt, 0)
#             esng *= attnrmsng
#             if (recmod==3  or recmod==4):
#                 esng += (rssng+ssng)*sbmsk
#             else:
#                 esng += rssng*sbmsk

#             # corrections to be backprojected to the image space
#             crr = attnrmsng*(np.float32(psng)/esng)
#             crr[np.isnan(crr)] = 0
#             crr[np.isinf(crr)] = 0
#             petprj.bprj(bim, crr, txLUT, axLUT, sinoTIdx[n,:], Cnt)
#             # devide the backprojected image by the corresponding subset sensitivity image
#             bim /= sim[n,:,:,:]
#             # apply the reconstruction mask
#             xim *= rcnmsk*bim
#             # get rid of any NaN values, if any
#             xim[np.isnan(xim)]=0

#             # plt.figure(); plt.imshow(xim[:,:,70], interpolation='none', cmap='gray'); plt.show()

#         #plt.figure(); plt.imshow(xim[:,:,70], interpolation='none', cmap='gray'); plt.show()
#         if (recmod==3  or recmod==4) and k<itr-1:
#             ssn, sssr, amsk = mmrsct.vsm(mumaps, mmrimg.convert2e7(xim, Cnt), datain, hst, rsino, txLUT, axLUT, Cnt, prcntScl=0.1, emmsk=True)
#             ssng = mmraux.remgaps(ssn, txLUT, Cnt)

#     #---- save images -----
#     #first convert to standard mMR image size
#     im = mmrimg.convert2e7(xim, Cnt)
#     B = mmrimg.image_affine(datain, Cnt)
#     #save the nii image
#     fout = os.path.dirname(datain['lm_dcm'])+'/'+os.path.basename(datain['lm_dcm'])[:8]+'_osem14_i'+str(itr)+'_s'+str(Cnt['SPN'])+'_r'+str(recmod)+'.nii'
#     nimpa.array2nii( im[::-1,::-1,:], B, fout)
#     #do smoothing and save the image
#     if fwhm>0:
#         imsmo = ndi.filters.gaussian_filter(im, fwhm2sig(fwhm, Cnt), mode='mirror')
#         nimpa.array2nii( imsmo[::-1,::-1,:], B,
#             os.path.dirname(datain['lm_dcm'])+'/'+os.path.basename(datain['lm_dcm'])[:8]+'_osem14_i'+str(itr)+'_s'+str(Cnt['SPN'])+'_r'+str(recmod)+'_smo'+str(fwhm)+'.nii')

#     if recmod==3:
#         datain['em_crr'] = fout

#     return im, fout