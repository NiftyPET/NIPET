import os
import sys

from textwrap import dedent

import numpy as np
import h5py

from math import pi
import scipy.ndimage as ndi

# import the C-extension with CUDA
from . import lmproc_sig

#-------------------------------------------------------------------------------
# LOGGING
#-------------------------------------------------------------------------------
import logging

#> console handler
ch = logging.StreamHandler()
formatter = logging.Formatter(
    '\n%(levelname)s> %(asctime)s - %(name)s - %(funcName)s\n> %(message)s'
    )
ch.setFormatter(formatter)

def get_logger(name):
    return logging.getLogger(name)
#-------------------------------------------------------------------------------



def lminfo_sig(datain, Cnt, t0=0, t1=0):

    #> set verbose and its level
    log = get_logger(__name__)
    log.setLevel(Cnt['LOG'])

    if not os.path.isfile(datain['lm_h5']):
        raise IOError('LM HDF5 file not found!')

    f = h5py.File(datain['lm_h5'],'r')

    if (f['HeaderData']['ListHeader']['isListCompressed'][0])>0:
        raise IOError(
            'The list mode data is compressed \
            and has to be first decompressed using GE proprietary software!'
        )

    else:
        log.debug('the list mode is decompressed [OK]')

    lm = f['ListData']['listData']

    # find first time marker by reading first k=1 time markers
    # event offset
    eoff = 0
    # direction of time search: 1-forward
    dsearch = 1
    # how many t-markers forward?
    k_markers = 1

    #> first time marker
    eoff_start, tstart, _ = lmproc_sig.nxtmrkr(datain['lm_h5'], Cnt['BPE'], eoff, k_markers, dsearch)

    #> last time marker
    eoff_end, tend, _ = lmproc_sig.nxtmrkr(datain['lm_h5'], Cnt['BPE'], (lm.shape[0]//Cnt['BPE'])-Cnt['BPE'], 1, -1)


    #> total number of elements in the list mode data
    totele = lm.shape[0]//Cnt['BPE']

    #> offset for first events
    eoff_first = 0

    #> last event offset
    eoff_last = totele-1


    if not t0==t1==0:

        #> update the times by the offset if it is greater than 0
        t1 += tstart//Cnt['ITIME']
        t0 += tstart//Cnt['ITIME']

        if (t1*Cnt['ITIME'])>tend:
            t1 = (tend+Cnt['ITIME']-1)//Cnt['ITIME']

        if (t0*Cnt['ITIME'])<=tstart:
            t0 = tstart//Cnt['ITIME']
        
        log.debug('t0 = {}, t1 = {}'.format(t0, t1))

        
        def find_tmark(t, tstart, tend, eoff_start, eoff_end, lmpth, bpe):
            ''' 
            find the event offsets for time index t
            to be used for list mode data processing
            '''

            trgt = int(t*Cnt['ITIME'])

            if trgt<tstart:
                trgt = tstart

            if trgt>tend:
                trgt = tend

            log.debug('target t_marker: {}'.format(trgt))

            k_markers = 100
            eoff, tmrk, counts = lmproc_sig.nxtmrkr(lmpth, bpe, 0, k_markers, 1)

            #> average recorded events per ms
            epm = eoff/k_markers


            flg_done = False
            while (abs(tmrk-trgt)>10) or flg_done:
                
                skip_off = int(eoff + (trgt-tmrk)*epm) #+ eoff_start
                if skip_off>=eoff_end:
                    skip_off = int(totele-0.25*epm*bpe)
                    log.debug('corrected offset to: {}'.format(skip_off))

                if skip_off<eoff_start:
                    skip_off = int(eoff_start+bpe)
                    log.debug('corrected offset to: {}'.format(skip_off))

                eoff_n, tmrk_n, _ = lmproc_sig.nxtmrkr(lmpth, bpe, skip_off, 1, np.sign(trgt-tmrk))
                
                if (tmrk_n==tmrk):
                    flg_done = True
                else:
                    epm = (eoff_n-eoff)/(tmrk_n-tmrk)
                
                eoff = eoff_n
                tmrk = tmrk_n
                
                log.debug('t_mark: {}'.format(tmrk))

            # import pdb; pdb.set_trace()
            if tmrk!=trgt:
                eoff, tmrk, _ = lmproc_sig.nxtmrkr(lmpth, bpe, eoff, abs(trgt-tmrk), np.sign(trgt-tmrk)) #+1*((trgt-tmrk)<0)

            return eoff, tmrk


        #> start
        eoff0, tmrk0 = find_tmark(t0, tstart, tend, eoff_start, eoff_end, datain['lm_h5'], Cnt['BPE'])
        #> stop
        eoff1, tmrk1 = find_tmark(t1, tstart, tend, eoff_start, eoff_end, datain['lm_h5'], Cnt['BPE'])

        #> number of elements to be considered in the list mode data
        ele = eoff1 - eoff0

        

    else:

        eoff0 = eoff_first
        eoff1 = eoff_last

        tmrk0 = tstart
        tmrk1 = tend
            
        # number of elements to be considered in the list mode data
        ele = totele

    #> integration time tags (+1 for the end)
    nitag = ((tmrk1-tmrk0)+Cnt['ITIME']-1)//Cnt['ITIME']

    #> update real time markers in seconds
    t0 = tmrk0//Cnt['ITIME']
    t1 = tmrk1//Cnt['ITIME']


    log.info(dedent('''\
        -----------------------------------------------
        > the first time is: {}s at event address: {}
        > the last  time is: {}s at event address: {}
        ------------------------------------------------
        > the start time is: {}s at event address: {} (used as offset)
        > the stop  time is: {}s at event address: {}
        > the number of report itags is: {}
        > -----------------------------------------------
        '''.format(
            tstart/Cnt['ITIME'], eoff_start,
            tend/Cnt['ITIME'], eoff_end,
            tmrk0/Cnt['ITIME'], eoff0,
            tmrk1/Cnt['ITIME'], eoff1,
            nitag)))

    f.close()


    return dict(
        nitag=nitag,
        nele=ele,
        totele=totele,
        tm0=tmrk0,
        tm1=tmrk1,
        evnt_addr0=eoff0,
        evnt_addr1=eoff1,
        toff=tstart,
        tend=tend)


#================================================================================
# HISTOGRAM THE LIST-MODE DATA
#--------------------------------------------------------------------------------
def hist(datain, txLUT, axLUT, Cnt, frms=np.array([0], dtype=np.uint16), use_stored=False, hst_store=False, t0=0, t1=0, cmass_sig=5 ):

    # histogramming with bootstrapping:
    # Cnt['BTP'] = 0: no bootstrapping [default];
    # Cnt['BTP'] = 2: parametric bootstrapping (using Poisson distribution with mean = 1)

    #> set verbose and its level
    log = get_logger(__name__)
    log.setLevel(Cnt['LOG'])

    # gather info about the LM time tags
    lmdct = lminfo_sig(datain, Cnt, t0, t1)

    # ====================================================================
    # SETTING UP CHUNKS
    # divide the data into data chunks
    # the default is to read around 1GB to be dealt with all streams (default: 32)
    nchnk = (lmdct['nele']+Cnt['ELECHNK']-1)//Cnt['ELECHNK']
    log.info('''\
        \r> duration by integrating time tags [s]: {}
        \r> # chunks of data (initial): {}
        \r> # elechnk: {}', 
        '''.format(lmdct['nitag'], nchnk, Cnt['ELECHNK']))

    # divide the list mode data into chunks in terms of addresses of selected time tags
    # break time tag
    btag = np.zeros((nchnk+1), dtype=np.int32)

    # address (position) in file (in bpe-bytes unit)
    atag = np.zeros((nchnk+1), dtype=np.uint64)

    # elements per thread to be dealt with
    ele4thrd = np.zeros((nchnk), dtype=np.int32)

    # elements per data chunk
    ele4chnk = np.zeros((nchnk), dtype=np.int32)

    # byte values for the whole event
    bval = np.zeros(Cnt['BPE'], dtype=int)

    atag[0] = lmdct['evnt_addr0']
    btag[0] = 0


    # LM data properties in a dictionary
    lmprop = {
        'lmfn':datain['lm_h5'],
        'bpe' :Cnt['BPE'],
        'nele':lmdct['nele'],
        'nchk':nchnk,
        'nitg':lmdct['nitag'],
        'toff':lmdct['toff'],
        'tend':lmdct['tend'],
        'tm0' :lmdct['tm0'],
        'tm1' :lmdct['tm1'],
        'atag':atag,
        'btag':btag,
        'ethr':ele4thrd,
        'echk':ele4chnk,
        'LOG':Cnt['LOG']
        }

    

    # get the setup into <lmprop>
    lmproc_sig.lminfo(lmprop)

    # import pdb; pdb.set_trace()

    # ---------------------------------------
    # preallocate all the output arrays
    if (lmdct['nitag']>Cnt['MXNITAG']): tn = Cnt['MXNITAG']//(1<<Cnt['VTIME'])
    else: tn = lmdct['nitag']//(1<<Cnt['VTIME'])

    # sinogram projection views (sort timre frames govern by VTIME)
    pvs = np.zeros((tn, 2*Cnt['NRNG']-1, Cnt['NSBINS']), dtype=np.uint32)
    # prompt head curve (counts per second)
    phc = np.zeros((lmdct['nitag']), dtype=np.uint32)
    # centre of mass of radiodistribution (axially only)
    mss = np.zeros((lmdct['nitag']), dtype=np.float32)
    # prompt sinogram
    psino = np.zeros((Cnt['NRNG']*Cnt['NRNG'], Cnt['NSBINS'], Cnt['NSANGLES']), dtype=np.uint32)
    hstout = {  'phc':phc,
                'mss':mss,
                'pvs':pvs,
                'psn':psino}


    # do the histogramming and processing
    lmproc_sig.hist(hstout, lmprop, frms, txLUT, axLUT, Cnt)

    #unpack short (interval) sinogram projection views
    pvs_sgtl = np.array( hstout['pvs']>>8, dtype=float)
    pvs_sgtl = pvs_sgtl[:,::-1,:]
    pvs_crnl = np.array( np.bitwise_and(hstout['pvs'], 255), dtype=float )
    pvs_crnl = pvs_crnl[:,::-1,:]

    cmass = 1*ndi.filters.gaussian_filter(hstout['mss'], cmass_sig, mode='mirror')
    #> apply the axial dimensions in [cm] to the centre of mass
    cmass = cmass*Cnt['AXFOV']/Cnt['NSEG0']


    hst = {
        'pvs_sgtl':pvs_sgtl,
        'pvs_crnl':pvs_crnl,
        'cmass':cmass,
        'phc':hstout['phc'],
        'psino':np.transpose(hstout['psn'], (0,2,1)),
        'dur':lmdct['nitag'],
        'lmprop':lmprop
    }
    return hst

