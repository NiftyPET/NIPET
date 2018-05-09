"""hist.py: processing of PET list-mode data: histogramming and randoms estimation."""
__author__      = "Pawel Markiewicz"
__copyright__   = "Copyright 2018"
#-------------------------------------------------------------------------------
import numpy as np
from math import pi
import sys
import os
import scipy.ndimage as ndi
import nibabel as nib
import cPickle as pickle

#CUDA extension module
import mmr_lmproc

#================================================================================
# HISTOGRAM THE LIST-MODE DATA
#--------------------------------------------------------------------------------
def mmrhist(
        datain,
        scanner_params,
        t0=0, t1=0,
        outpath='',
        frms=np.array([0], dtype=np.uint16),
        use_stored=False,
        store=False,
        cmass_sig=5):
    '''Process the list-mode data and return histogram, head curves, and centre of mass for motion detection.
    '''

    # constants, transaxial and axial LUTs are extracted
    Cnt   = scanner_params['Cnt']
    txLUT = scanner_params['txLUT']
    axLUT = scanner_params['axLUT']

    hst = hist(
            datain, txLUT, axLUT, Cnt,
            frms=frms,
            use_stored=use_stored,
            store=store,
            outpath=outpath,
            t0=t0, t1=t1,
            cmass_sig=cmass_sig)

    return hst


def hist(datain, txLUT, axLUT, Cnt, frms=np.array([0], dtype=np.uint16), use_stored=False, store=False, outpath='', t0=0, t1=0, cmass_sig=5 ):

    # histogramming with bootstrapping:
    # Cnt['BTP'] = 0: no bootstrapping [default];
    # Cnt['BTP'] = 1: non-parametric bootstrapping;
    # Cnt['BTP'] = 2: parametric bootstrapping (using Poisson distribution with mean = 1)

    #number of dynamic frames
    nfrm = len(frms)
    if    Cnt['SPN']==1:  nsinos=Cnt['NSN1']
    elif  Cnt['SPN']==11: nsinos=Cnt['NSN11']
    elif  Cnt['SPN']==0:  nsinos=Cnt['NSEG0']

    if Cnt['VERBOSE']: print 'i> histograming with span', Cnt['SPN'], 'and', nfrm, 'dynamic frames.'


    if use_stored==True and 'sinos' in datain and os.path.basename(datain['sinos'])=='sinos_s'+str(Cnt['SPN'])+'_n'+str(nfrm)+'_frm-'+str(t0)+'-'+str(t1)+'.npy' :

        # nele, ttags, tpos = mmr_lmproc.lminfo(datain['lm_bf'])
        # nitag = (ttags[1]-ttags[0]+999)/1000

        hstout = {}
        (hstout['phc'], hstout['dhc'], hstout['mss'], hstout['pvs'],
         hstout['bck'], hstout['fan'], hstout['psn'], hstout['dsn'],
         hstout['ssr']) =   np.load( datain['sinos'] )

        nitag = len(hstout['phc'])
        if Cnt['VERBOSE']: print 'i> duration by integrating time tags [s]:', nitag

    elif os.path.isfile(datain['lm_bf']):
        # gather info about the LM time tags
        nele, ttags, tpos = mmr_lmproc.lminfo(datain['lm_bf'])
        nitag = (ttags[1]-ttags[0]+999)/1000
        if Cnt['VERBOSE']: print 'i> duration by integrating time tags [s]:', nitag

        # adjust frame time if outside the limit
        if t1>nitag: t1 = nitag
        # check if the time point is allowed
        if t0>=nitag:
            print 'e> time frame definition outside the list-mode data acquisition time!'
            raise ValueError('Not allowed time frame definition')

        # ---------------------------------------
        # preallocate all the output arrays
        VTIME = 2
        MXNITAG = 5400 #limit to 1hr and 30mins
        if (nitag>MXNITAG): 
            tn = MXNITAG/(1<<VTIME)
        else: 
            tn = (nitag+(1<<VTIME)-1)/(1<<VTIME)

        pvs = np.zeros((tn, Cnt['NSEG0'], Cnt['NSBINS']), dtype=np.uint32)
        phc = np.zeros((nitag), dtype=np.uint32)
        dhc = np.zeros((nitag), dtype=np.uint32)
        mss = np.zeros((nitag), dtype=np.float32)

        bck = np.zeros((2, nitag, Cnt['NBCKT']), dtype=np.uint32)
        fan = np.zeros((nfrm, Cnt['NRNG'], Cnt['NCRS']), dtype=np.uint32)

        if nfrm==1:
            psino = np.zeros((nsinos, Cnt['NSANGLES'], Cnt['NSBINS']), dtype=np.uint16)
            dsino = np.zeros((nsinos, Cnt['NSANGLES'], Cnt['NSBINS']), dtype=np.uint16)
        elif nfrm>1:
            psino = np.zeros((nfrm, nsinos, Cnt['NSANGLES'], Cnt['NSBINS']), dtype=np.uint8)
            dsino = np.zeros((nfrm, nsinos, Cnt['NSANGLES'], Cnt['NSBINS']), dtype=np.uint8)

        # single slice rebinned prompots
        ssr = np.zeros((Cnt['NSEG0'], Cnt['NSANGLES'], Cnt['NSBINS']), dtype=np.uint32)

        hstout = {
            'phc':phc,
            'dhc':dhc,
            'mss':mss,
            'pvs':pvs,

            'bck':bck,
            'fan':fan,

            'psn':psino,
            'dsn':dsino,
            'ssr':ssr

        }
        # ---------------------------------------

        # do the histogramming and processing
        mmr_lmproc.hist(
                    hstout,
                    datain['lm_bf'],
                    frms, 
                    t0, t1,
                    txLUT, axLUT, Cnt)
        if store:
            if outpath=='':
                fsino = os.path.dirname(datain['lm_bf'])
            else:
                fsino = os.path.join(outpath, 'sino')
                nipet.mmraux.create_dir(fsino)
            # complete the path with the file name
            fsino = os.path.join(fsino, 'sinos_s'+str(Cnt['SPN'])+'_n'+str(nfrm)+'_frm-'+str(t0)+'-'+str(t1)+'.npy')
            # store to the above path
            np.save(fsino,  (hstout['phc'], hstout['dhc'], hstout['mss'], hstout['pvs'],
                    hstout['bck'], hstout['fan'], hstout['psn'], hstout['dsn'], hstout['ssr']))

    else:
        print 'e> input list-mode data not defined.'
        return

    #short (interval) projection views
    pvs_sgtl = np.array( hstout['pvs']>>8, dtype=float)
    pvs_crnl = np.array( np.bitwise_and(hstout['pvs'], 255), dtype=float )

    cmass = Cnt['SO_VXZ']*ndi.filters.gaussian_filter(hstout['mss'], cmass_sig, mode='mirror')
    if Cnt['VERBOSE']: print 'i> centre of mass of axial radiodistribution (filtered with Gaussian of SD =', cmass_sig, '):  COMPLETED.'

    #========================== BUCKET SINGLES ==============================
    #number of single rates reported for the given second
    nsr = (hstout['bck'][1,:,:]>>30)
    #average in a second period
    hstout['bck'][0,nsr>0] /= nsr[nsr>0]
    #time indeces when single rates given
    tmsk = np.sum(nsr,axis=1)>0
    single_rate = np.copy(hstout['bck'][0,tmsk,:])
    #time
    t = np.arange(nitag)
    t = t[tmsk]
    #get the average bucket singles:
    buckets = np.int32( np.sum(single_rate,axis=0)/single_rate.shape[0] )
    if Cnt['VERBOSE']: print 'i> dynamic and static buckets single rates:  COMPLETED.'
    #=========================================================================

    # account for the fact that when t0==t1 that means that full dataset is processed
    if t0==t1: t1 = t0+nitag

    pdata={
        't0':t0,
        't1':t1,
        'dur':t1-t0,                #duration
        'phc':hstout['phc'],        #prompts head curve
        'dhc':hstout['dhc'],        #delayeds head curve
        'cmass':cmass,              #centre of mass of the radiodistribution in axial direction
        'pvs_sgtl':pvs_sgtl,        #sagittal projection views in short intervals
        'pvs_crnl':pvs_crnl,        #coronal projection views in short intervals
        
        'fansums':hstout['fan'],    #fan sums of delayeds for variance reduction of random event sinograms
        'sngl_rate':single_rate,    #bucket singles over time
        'tsngl':t,                  #time points of singles measurements in list-mode data
        'buckets':buckets,          #average bucket singles

        'psino':hstout['psn'],      #prompt sinogram
        'dsino':hstout['dsn'],      #delayeds sinogram
        'pssr' :hstout['ssr']       #single-slice rebinned sinogram of prompts
    }

    return pdata

#================================================================================
# GET REDUCED VARIANCE RANDOMS
#--------------------------------------------------------------------------------
def rand(fansums, txLUT, axLUT, Cnt):

    if    Cnt['SPN']==1:  nsinos=Cnt['NSN1']
    elif  Cnt['SPN']==11: nsinos=Cnt['NSN11']
    elif  Cnt['SPN']==0:  nsinos=Cnt['NSEG0']

    #number of frames
    nfrm = fansums.shape[0]
    if Cnt['VERBOSE']: print 'i> # of dynamic frames:', nfrm

    #random sino and estimated crystal map of singles put into a dictionary
    rsn  = np.zeros((nsinos, Cnt['NSANGLES'], Cnt['NSBINS']), dtype=np.float32)
    cmap = np.zeros((Cnt['NCRS'], Cnt['NRNG']), dtype=np.float32)
    rndout = {
        'rsn': rsn,
        'cmap':cmap,
    }

    #save results for each frame
    
    rsino = np.zeros((nfrm, nsinos, Cnt['NSANGLES'], Cnt['NSBINS']), dtype=np.float32)
    crmap = np.zeros((nfrm, Cnt['NCRS'], Cnt['NRNG']), dtype=np.float32)
    
    for i in range(nfrm):
        rndout['rsn'][:,:,:] = 0
        rndout['cmap'][:,:]  = 0
        mmr_lmproc.rand(rndout, fansums[i,:,:], txLUT, axLUT, Cnt)
        rsino[i,:,:,:] = rndout['rsn']
        crmap[i,:,:] = rndout['cmap']

    if nfrm==1:
        rsino = rsino[0,:,:,:]
        crmap = crmap[0,:,:]

    return rsino, crmap


#================================================================================
# NEW!! GET REDUCED VARIANCE RANDOMS (BASED ON PROMPTS)
#--------------------------------------------------------------------------------
def prand(fansums, pmsk, txLUT, axLUT, Cnt):

    if    Cnt['SPN']==1:  nsinos=Cnt['NSN1']
    elif  Cnt['SPN']==11: nsinos=Cnt['NSN11']
    elif  Cnt['SPN']==0:  nsinos=Cnt['NSEG0']

    #number of frames
    nfrm = fansums.shape[0]
    if Cnt['VERBOSE']: print 'i> # of dynamic frames:', nfrm

    #random sino and estimated crystal map of singles put into a dictionary
    rsn  = np.zeros((nsinos, Cnt['NSANGLES'], Cnt['NSBINS']), dtype=np.float32)
    cmap = np.zeros((Cnt['NCRS'], Cnt['NRNG']), dtype=np.float32)
    rndout = {
        'rsn': rsn,
        'cmap':cmap,
    }

    #save results for each frame
    
    rsino = np.zeros((nfrm, nsinos, Cnt['NSANGLES'], Cnt['NSBINS']), dtype=np.float32)
    crmap = np.zeros((nfrm, Cnt['NCRS'], Cnt['NRNG']), dtype=np.float32)
        
    for i in range(nfrm):
        rndout['rsn'][:,:,:] = 0
        rndout['cmap'][:,:]  = 0
        mmr_lmproc.prand(rndout, pmsk, fansums[i,:,:], txLUT, axLUT, Cnt)
        rsino[i,:,:,:] = rndout['rsn']
        crmap[i,:,:] = rndout['cmap']

    if nfrm==1:
        rsino = rsino[0,:,:,:]
        crmap = crmap[0,:,:]

    return rsino, crmap







#================================================================================
def sino2nii(sino, Cnt, fpth):
    '''save sinogram in span-11 into NIfTI file'''
    #number of segments
    segn = len(Cnt['SEG'])
    cumseg = np.cumsum(Cnt['SEG'])
    cumseg = np.append([0], cumseg)

    #plane offset (relative to 127 planes of seg 0) for each segment
    OFF = np.min( abs( np.append([Cnt['MNRD']], [Cnt['MXRD']], axis=0) ), axis=0 )
    niisn = np.zeros(( Cnt['SEG'][0], Cnt['NSANGLES'], Cnt['NSBINS'], segn), dtype=sino.dtype)

    #first segment (with direct planes)
    # tmp = 
    niisn[:,:,:,0] = sino[Cnt['SEG'][0]-1::-1, ::-1, ::-1]

    for iseg in range(1,segn):
        niisn[OFF[iseg]:OFF[iseg]+Cnt['SEG'][iseg], :, :, iseg] = sino[cumseg[iseg]+Cnt['SEG'][iseg]-1:cumseg[iseg]-1:-1, ::-1, ::-1 ]

    niisn = np.transpose(niisn, (2, 1, 0, 3))

    nim = nib.Nifti1Image(niisn, np.eye(4))
    nib.save(nim, fpth)


#=================================================================================
# create michelogram map for emission data, only when the input sino in in span-1
def get_michem(sino, axLUT, Cnt):
    # span:
    spn = -1

    if Cnt['SPN']==1:
        slut = np.arange(Cnt['NSN1']) #for span 1, one-to-one mapping
    elif Cnt['SPN']==11:
        slut = axLUT['sn1_sn11']
    else:
        print 'e> sino is not in span-1 neither span-11'
        sys.exit()

    #acitivity michelogram
    Mem = np.zeros((Cnt['NRNG'],Cnt['NRNG']), dtype=np.float32)
    #sino to ring number & sino-1 to sino-11 index:
    sn1_rno  = axLUT['sn1_rno']
    #sum all the sinograms inside 
    ssm = np.sum(sino, axis=(1,2))

    for sni in range(len(sn1_rno)):
        r0 = sn1_rno[sni,0]
        r1 = sn1_rno[sni,1]
        Mem[r1,r0] = ssm[slut[sni]]

    return Mem

#-------------------------------------------------------------------------------------------------
def get_time_offset(hst):
    '''
    Detects when the signal is stronger than the randoms (noise) in the list-mode data stream.
    '''
    # detect when the signal (here prompt data) is almost as strong as randoms
    s = hst['dhc']>0.98*hst['phc']
    # return index, which will constitute time in seconds, for this offset
    return  len(s)-np.argmax(s[::-1])-1

def split_frames(hst, Tref=60):
    '''
    Splits the whole acquisition data into approximately statistically equivalent frames
    relative to the first frame whose duration is Tref.  The next frames will have a similar
    count level.
    hst: histogram dictionary
    Tref: reference duration in seconds
    '''
    # get the offset
    ioff = get_time_offset(hst)
    # difference between prompts and randoms
    diff = np.int64(hst['phc']) - np.int64(hst['dhc'])
    # follow up index
    i = ioff
    j = i+Tref
    # reference count level
    cref = np.sum(diff[i:j])
    # cumulative sum of the difference
    csum = np.cumsum(diff)

    # threshold to be achieved
    thrsh = csum[j-1] + cref
    fdur = [j-i]
    frms = [[i,j]]
    clvl = [cref]
    print 'counts t(%d,%d) = %d. diff=%d' % ( i,j,clvl[-1] , np.sum(diff[i:j])-cref )
    while thrsh<csum[-1]:
        i = j
        j = np.argmax(csum>thrsh)
        fdur.append(j-i)
        frms.append([i,j])
        clvl.append(np.sum(diff[i:j]))
        print 'counts t(%d,%d) = %d. diff=%d' % ( i,j,clvl[-1] , np.sum(diff[i:j])-cref )
        thrsh += cref
    # last remianing frame
    i=j
    j=hst['dur']
    # if last frame is short, include it in the last one.
    if np.sum(diff[i:])>.5*cref:
        fdur.append(j-i)
        frms.append([i,j])
        clvl.append(np.sum(diff[i:]))
    else:
        fdur[-1] += j-i
        frms[-1][-1] += j-i
        clvl[-1] += np.sum(diff[i:])
        i = frms[-1][0]
    print 'counts t(%d,%d) = %d. diff=%d' % ( i,j,clvl[-1] , np.sum(diff[i:j])-cref )
    return {'timings':frms, 'fdur':fdur, 'fcnts':clvl, 'offset':ioff}


#-------------------------------------------------------------------------------------------------

def frame_position(hst, tposition, Cref=0, tr0=0, tr1=15, verbose = True):
    ''' hst: histogram data
        tposition: time position (middle point) of the frame to be defined
        Cref: reference count level to be in the frame (prompts - delays)
        tr0, tr1: time definitions of reference frame whose count level
        will be used as the reference Cref.  If Cref is not defined (i.e. = 0)
        then the tr0 and tr1 will be used.
    '''
    
    # claculate the difference between the prompts and delays (more realistic count level)
    diff = np.int64(hst['phc']) - np.int64(hst['dhc'])
    # cumulative sum for calculating count levels in arbitrary time windows
    cumdiff = np.cumsum(diff)

    if Cref==0:
        Cref = cumdiff[tr1]-cumdiff[tr0-1]

    if Cref<0:
        raise ValueError('The reference count level has to be non-negative')

    if verbose: print 'i> reference count level:', Cref


    stp0 = 0
    stp1 = 0
    Cw = 0
    while Cw<Cref:
        # check if it is possible to widen the sampling window both ways
        if (tposition-stp0-1)>0: stp0 += 1
        if (tposition+stp1+1)<=len(cumdiff)-1: stp1 += 1
        Cw = cumdiff[tposition+stp1] - cumdiff[tposition-stp0-1]

    tw0 = tposition-stp0
    tw1 = tposition+stp1
    Tw = tw1 - tw0
    if verbose:
        print 'i> time window t[{}, {}] of duration T={} and count level Cw={}'.format(tw0, tw1, Tw, Cw)

    return (tw0, tw1)


def auxilary_frames(hst, t_frms, Cref=0, tr0=0, tr1=15, verbose = True):
    ''' Get auxilary time frames with equal count levels for constant precision in
        the estimation of subject motion based on PET data. 
    '''
    
    # claculate the difference between the prompts and delays (more realistic count level)
    diff = np.int64(hst['phc']) - np.int64(hst['dhc'])

    # previous frame (time tuple)
    prev_frm = (0,0)
    # previous frame index
    prev_i = -1
    # look up table to the auxilary frames from the regular ones
    timings = []
    fi2afi = []
    for i in range(len(t_frms)):
        # time point as an average between the frame end points
        tp = int(np.mean([t_frms[i][0],t_frms[i][1]]))
        # alternative (more accurate) average through centre of mass
        t0 = t_frms[i][0]
        t1 = t_frms[i][1]
        if t1>=hst['dur']: t1 = hst['dur']-1
        t = np.arange(t0,t1)
        tcm = np.sum(diff[t]*t)/np.sum(diff[t])
        # get the tuple of the equivalent count level frame
        frm = frame_position(hst, tcm, tr0=tr0, tr1=tr1, verbose=False)
        # form the LUT
        if frm!=prev_frm:
            prev_frm = frm
            prev_i += 1
            timings.append(list(frm))
        fi2afi.append(prev_i)
        if verbose:
            print 't[{}, {}]; tp={}, tcm={} => frm id:{}, timings:{}'.format(t_frms[i][0], t_frms[i][1], tp, tcm, fi2afi[-1], timings[-1])
    # form the list of auxilary dynamic frames of equivalent count level (as in Cref) for reconstruction
    mfrm = ['fluid'] + timings 
    return {'timings':mfrm, 'frame_idx':fi2afi}