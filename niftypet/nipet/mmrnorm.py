"""mmraux.py: auxilary functions for PET list-mode data processing."""

__author__      = "Pawel Markiewicz"
__copyright__   = "Copyright 2018"

#-------------------------------------------------------------------------------
import numpy as np
import sys
import os
import pydicom as dcm
import re
from pkg_resources import resource_filename
import logging

#auxiliary functions through Python extensions in CUDA
import mmr_auxe


#=================================================================================================
# GET NORM COMPONENTS
#=================================================================================================
def get_components(datain, Cnt):
    "Return the normalisation components from provided file."
    log = logging.getLogger(__name__)

    if 'nrm_ima' in datain and os.path.isfile(datain['nrm_ima']):
        fnrm_dat = datain['nrm_bf']
        fnrm_hdr = datain['nrm_ima']

    elif 'nrm_dcm' in datain and os.path.isfile(datain['nrm_dcm']) and os.path.isfile(datain['nrm_bf']):
        fnrm_dat = datain['nrm_bf']
        fnrm_hdr = datain['nrm_dcm']
    else:
        log.error('norm file does not exist or it is incomplete.')
        raise NameError('NoNorm')

    f = open(fnrm_dat, 'rb')
    #geometric effects
    geo       = np.fromfile(f, np.float32, Cnt['NSBINS']*Cnt['NSEG0'])
    geo.shape = (Cnt['NSEG0'], Cnt['NSBINS'])
    #crystal interference
    crs_intf  = np.fromfile(f, np.float32, 9*Cnt['NSBINS'])
    crs_intf.shape = (Cnt['NSBINS'],9)
    #crystal efficiencies
    crs_eff   = np.fromfile(f, np.float32, Cnt['NCRS']*Cnt['NRNG'])
    crs_eff.shape  = (Cnt['NRNG'], Cnt['NCRS'])
    #axial effects
    ax_eff1   = np.fromfile(f, np.float32, Cnt['NSN11'])
    #paralyzing ring DT parameters
    rng_dtp   = np.fromfile(f, np.float32, Cnt['NRNG'])
    #non-paralyzing ring DT parameters
    rng_dtnp  = np.fromfile(f, np.float32, Cnt['NRNG'])
    #TX crystal DT parameter
    crs_dt    = np.fromfile(f, np.float32, 9)
    #additional axial effects
    ax_eff2   = np.fromfile(f, np.float32, Cnt['NSN11'])
    f.close()

    #-------------------------------------------------
    #the files below are found based on a 24hr scan of germanium-68 phantom
    # axial effects for span-1
    fnm = 'AxialFactorForSpan1.npy'
    fpth = os.path.join(resource_filename(__name__, '../auxdata'), fnm)#'niftypet'
    ax_f1 = np.load(fpth, allow_pickle=True)
    # relative scale factors for axial scatter deriving span-11 scale factors from SSR scale factors
    fnm = 'RelativeScaleFactors_scatter_axial_ssrTOspan11.f32'
    fpth = os.path.join(resource_filename(__name__, '../auxdata'), fnm)
    f = open(fpth, 'rb')
    sax_f11 = np.fromfile(f, np.float32, Cnt['NSN11'])
    f.close()
    # relative scale factors for axial scatter deriving span-1 scale factors from SSR scale factors
    fnm = 'RelativeScaleFactors_scatter_axial_ssrTOspan1.f32'
    fpth = os.path.join(resource_filename(__name__, '../auxdata'), fnm)
    f = open(fpth, 'rb')
    sax_f1 = np.fromfile(f, np.float32, Cnt['NSN1'])
    f.close()
    #-------------------------------------------------

    #-------------------------------------------------
    # HEADER FILE
    # possible DICOM locations for the Interfile header
    nhdr_locations = [[0x29,0x1010], [0x29,0x1110]]
    # read the DICOM file
    d = dcm.read_file(fnrm_hdr)
    if   d[0x0018, 0x1020].value == 'syngo MR B20P' or d[0x0018, 0x1020].value == 'syngo MR E11':
        nhdr = d[0x29,0x1010].value
    elif d[0x0018, 0x1020].value == 'syngo MR B18P':
        found_nhdr = False
        for loc in nhdr_locations:
            if loc in d:
                nhdr = d[loc].value
                if '!INTERFILE' in nhdr and 'scanner quantification factor' in nhdr:
                    log.debug('got the normalisation interfile header from [%#x, %#x]' % (loc[0], loc[1]))
                    found_nhdr = True
                    break
        if not found_nhdr:
            log.error('DICOM field with normalisation interfile header has not been found!')
            return None, None
    else: log.error('unknown scanner software version!');  return None, None

    f0 = nhdr.find('scanner quantification factor')
    f1 = f0+nhdr[f0:].find('\n')
    #regular expression for the needed three numbers
    p = re.compile(r'(?<=:=)\s*\d{1,5}[.]\d{3,10}[e][+-]\d{1,4}')
    #-quantification factor:
    qf = float(p.findall(nhdr[f0:f1])[0])
    #-local quantification correction factor
    qf_loc = 0.205
    #-------------------------------------------------

    nrmcmp = {'qf':qf, 'qf_loc':qf_loc, 'geo':geo, 'cinf':crs_intf, 'ceff':crs_eff,
                'axe1':ax_eff1, 'dtp':rng_dtp, 'dtnp':rng_dtnp,
                'dtc':crs_dt, 'axe2':ax_eff2, 'axf1':ax_f1,
                'sax_f11':sax_f11, 'sax_f1':sax_f1}


    return nrmcmp, nhdr


def get_sinog(datain, hst, axLUT, txLUT, Cnt, normcomp=None):

    #get the normalisation components
    if normcomp==None:
        normcomp, _ = get_components(datain, Cnt)

    #number of sino planes (2D sinos) depends on the span used
    if Cnt['SPN']==1:
        nsinos = Cnt['NSN1']
    elif Cnt['SPN']==11:
        nsinos = Cnt['NSN11']

    #predefine the sinogram
    sinog = np.zeros((txLUT['Naw'], nsinos), dtype=np.float32)

    #get the sino in the GPU-optimised shape
    mmr_auxe.norm(sinog, normcomp, hst['buckets'], axLUT, txLUT['aw2ali'], Cnt)

    return sinog


def get_sino(datain, hst, axLUT, txLUT, Cnt):

    #number of sino planes (2D sinos) depends on the span used
    if Cnt['SPN']==1:
        nsinos = Cnt['NSN1']
    elif Cnt['SPN']==11:
        nsinos = Cnt['NSN11']

    #get sino with no gaps
    s = get_sinog(datain, hst, axLUT, txLUT, Cnt)
    #preallocate sino with gaps
    sino = np.zeros((Cnt['NSANGLES'], Cnt['NSBINS'], nsinos), dtype=np.float32)
    #fill the sino with gaps
    mmr_auxe.pgaps(sino, s, txLUT, Cnt)
    sino = np.transpose(sino, (2,0,1))

    return sino

def get_norm_sino(datain, scanner_params, hst):
    Cnt = scanner_params['Cnt']
    txLUT = scanner_params['txLUT']
    axLUT = scanner_params['axLUT']

    # if not hst:
    #     hst = mmrhist.mmrhist(datain, scanner_params)

    #number of sino planes (2D sinos) depends on the span used
    if Cnt['SPN']==1:
        nsinos = Cnt['NSN1']
    elif Cnt['SPN']==11:
        nsinos = Cnt['NSN11']

    #get sino with no gaps
    s = get_sinog(datain, hst, axLUT, txLUT, Cnt)
    #preallocate sino with gaps
    sino = np.zeros((Cnt['NSANGLES'], Cnt['NSBINS'], nsinos), dtype=np.float32)
    #fill the sino with gaps
    mmr_auxe.pgaps(sino, s, txLUT, Cnt)
    sino = np.transpose(sino, (2,0,1))

    return sino
