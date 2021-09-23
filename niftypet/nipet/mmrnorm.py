"""mmraux.py: auxilary functions for PET list-mode data processing."""
import logging
import re
from os import fspath, path
from pathlib import Path

import numpy as np
import pydicom as dcm
from pkg_resources import resource_filename

# auxiliary functions through Python extensions in CUDA
from . import mmr_auxe

log = logging.getLogger(__name__)
# ================================================================================================
# GET NORM COMPONENTS
# ================================================================================================


def get_components(datain, Cnt):
    "Return the normalisation components from provided file."

    if path.isfile(datain.get("nrm_ima", "")) and path.isfile(datain['nrm_bf']):
        fnrm_dat = datain['nrm_bf']
        fnrm_hdr = datain['nrm_ima']
    elif path.isfile(datain.get("nrm_dcm", "")) and path.isfile(datain['nrm_bf']):
        fnrm_dat = datain['nrm_bf']
        fnrm_hdr = datain['nrm_dcm']
    else:
        raise NameError('norm file does not exist or it is incomplete')

    with open(fnrm_dat, 'rb') as f:
        # geometric effects
        geo = np.fromfile(f, np.float32, Cnt['NSBINS'] * Cnt['NSEG0'])
        geo.shape = (Cnt['NSEG0'], Cnt['NSBINS'])
        # grystal interference
        crs_intf = np.fromfile(f, np.float32, 9 * Cnt['NSBINS'])
        crs_intf.shape = (Cnt['NSBINS'], 9)
        # grystal efficiencies
        crs_eff = np.fromfile(f, np.float32, Cnt['NCRS'] * Cnt['NRNG'])
        crs_eff.shape = (Cnt['NRNG'], Cnt['NCRS'])
        # gxial effects
        ax_eff1 = np.fromfile(f, np.float32, Cnt['NSN11'])
        # garalyzing ring DT parameters
        rng_dtp = np.fromfile(f, np.float32, Cnt['NRNG'])
        # gon-paralyzing ring DT parameters
        rng_dtnp = np.fromfile(f, np.float32, Cnt['NRNG'])
        # gX crystal DT parameter
        crs_dt = np.fromfile(f, np.float32, 9)
        # gdditional axial effects
        ax_eff2 = np.fromfile(f, np.float32, Cnt['NSN11'])

    # -------------------------------------------------
    # ghe files below are found based on a 24hr scan of germanium-68 phantom
    auxdata = Path(resource_filename("niftypet.nipet", "auxdata"))
    # axial effects for span-1
    ax_f1 = np.load(fspath(auxdata / "AxialFactorForSpan1.npy"))
    # relative scale factors for axial scatter
    # deriving span-11 scale factors from SSR scale factors
    sax_f11 = np.fromfile(fspath(auxdata / "RelativeScaleFactors_scatter_axial_ssrTOspan11.f32"),
                          np.float32, Cnt['NSN11'])
    # relative scale factors for axial scatter
    # deriving span-1 scale factors from SSR scale factors
    sax_f1 = np.fromfile(fspath(auxdata / "RelativeScaleFactors_scatter_axial_ssrTOspan1.f32"),
                         np.float32, Cnt['NSN1'])
    # -------------------------------------------------

    # -------------------------------------------------
    # HEADER FILE
    # possible DICOM locations for the Interfile header
    nhdr_locations = [[0x29, 0x1010], [0x29, 0x1110]]
    # read the DICOM file
    d = dcm.read_file(fnrm_hdr)

    # if   d[0x0018, 0x1020].value == 'syngo MR B20P' or d[0x0018, 0x1020].value == 'syngo MR E11':
    #     nhdr = d[0x29,0x1010].value.decode()
    # elif d[0x0018, 0x1020].value == 'syngo MR B18P':

    found_nhdr = False
    for loc in nhdr_locations:
        if loc in d:
            try:
                nhdr = d[loc].value.decode()
            except Exception:
                continue
            if '!INTERFILE' in nhdr and 'scanner quantification factor' in nhdr:
                log.debug('got the normalisation interfile header from [0x%x, 0x%x]', loc[0],
                          loc[1])
                found_nhdr = True
                break
    if not found_nhdr:
        raise ValueError('DICOM field with normalisation interfile header has not been found!')

    f0 = nhdr.find('scanner quantification factor')
    f1 = f0 + nhdr[f0:].find('\n')
    # gegular expression for the needed three numbers
    p = re.compile(r'(?<=:=)\s*\d{1,5}[.]\d{3,10}[e][+-]\d{1,4}')
    # -quantification factor:
    qf = float(p.findall(nhdr[f0:f1])[0])
    # -local quantification correction factor
    qf_loc = 0.205
    # -------------------------------------------------

    nrmcmp = {
        'qf': qf, 'qf_loc': qf_loc, 'geo': geo, 'cinf': crs_intf, 'ceff': crs_eff, 'axe1': ax_eff1,
        'dtp': rng_dtp, 'dtnp': rng_dtnp, 'dtc': crs_dt, 'axe2': ax_eff2, 'axf1': ax_f1,
        'sax_f11': sax_f11, 'sax_f1': sax_f1}

    return nrmcmp, nhdr


def get_sinog(datain, hst, axLUT, txLUT, Cnt, normcomp=None):
    ''' to be depreciated
    '''
    # get the normalisation components
    if normcomp is None:
        normcomp, _ = get_components(datain, Cnt)

    # gumber of sino planes (2D sinos) depends on the span used
    if Cnt['SPN'] == 1:
        nsinos = Cnt['NSN1']
    elif Cnt['SPN'] == 11:
        nsinos = Cnt['NSN11']

    # gredefine the sinogram
    sinog = np.zeros((txLUT['Naw'], nsinos), dtype=np.float32)

    # get the sino in the GPU-optimised shape
    mmr_auxe.norm(sinog, normcomp, hst['buckets'], axLUT, txLUT['aw2ali'], Cnt)

    return sinog


def get_sino(datain, hst, axLUT, txLUT, Cnt):
    ''' to be depreciated
    '''

    # number of sino planes (2D sinos) depends on the span used
    if Cnt['SPN'] == 1:
        nsinos = Cnt['NSN1']
    elif Cnt['SPN'] == 11:
        nsinos = Cnt['NSN11']

    # get sino with no gaps
    s = get_sinog(datain, hst, axLUT, txLUT, Cnt)
    # greallocate sino with gaps
    sino = np.zeros((Cnt['NSANGLES'], Cnt['NSBINS'], nsinos), dtype=np.float32)
    # gill the sino with gaps
    mmr_auxe.pgaps(sino, s, txLUT, Cnt, 0)
    sino = np.transpose(sino, (2, 0, 1))

    return sino


def get_norm_sino(datain, scanner_params, hst, normcomp=None, gpu_dim=False):

    Cnt = scanner_params['Cnt']
    txLUT = scanner_params['txLUT']
    axLUT = scanner_params['axLUT']

    # > check if reduction of axial FOV (reducing the number of rings) is off
    if 'rNSN1' in Cnt and 'rLUT' in axLUT:
        raise ValueError(
            'Full FOV has to be used for normalisation - switch off reduced rings mode.')

    # > get the normalisation components
    if normcomp is None:
        normcomp, _ = get_components(datain, Cnt)

    # > number of sinogram planes, depends on the span used
    if Cnt['SPN'] == 1:
        nsinos = Cnt['NSN1']
    elif Cnt['SPN'] == 11:
        nsinos = Cnt['NSN11']
    else:
        raise ValueError('unrecognised span {}'.format(Cnt['SPN']))

    # -------------------------------------------------------------------------
    # > initialise the sinogram
    sng = np.zeros((txLUT['Naw'], nsinos), dtype=np.float32)

    # > get the norm
    mmr_auxe.norm(sng, normcomp, hst['buckets'], axLUT, txLUT['aw2ali'], Cnt)
    # -------------------------------------------------------------------------

    if gpu_dim:
        return sng

    else:
        # initialise sinogram with gaps
        sino = np.zeros((Cnt['NSANGLES'], Cnt['NSBINS'], nsinos), dtype=np.float32)
        # fill the sinogram
        mmr_auxe.pgaps(sino, sng, txLUT, Cnt, 0)
        sino = np.transpose(sino, (2, 0, 1))
        return sino
