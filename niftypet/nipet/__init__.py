#!/usr/bin/env python
"""initialise the NiftyPET NIPET package"""
__author__ = "Pawel J. Markiewicz", "Casper O. da Costa-Luis"
__copyright__ = "Copyright 2021"
# version detector. Precedence: installed dist, git, 'UNKNOWN'
try:
    from ._dist_ver import __version__
except ImportError:
    try:
        from setuptools_scm import get_version

        __version__ = get_version(root="../..", relative_to=__file__)
    except (ImportError, LookupError):
        __version__ = "UNKNOWN"
__all__ = [
    # GPU utils
    'resource_filename', 'cs', 'dev_info', 'gpuinfo',
    # utils
    'LOG_FORMAT', 'LogHandler', 'path_resources', 'resources',
    # package
    'img', 'lm', 'mmr_auxe', 'mmraux', 'mmrnorm', 'prj',
    # img
    'align_mumap', 'im_e72dev', 'im_dev2e7', 'hdw_mumap', 'obj_mumap',
    'pct_mumap', 'mmrchain',
    # lm
    'dynamic_timings', 'mmrhist', 'randoms',
    # mmraux
    'classify_input', 'get_mmrparams',
    # prj
    'back_prj', 'frwd_prj', 'simulate_recon', 'simulate_sino',
    # sct
    'vsm',
    # optional
    'video_dyn', 'video_frm', 'xnat']  # yapf: disable
from pkg_resources import resource_filename

from niftypet.ninst import cudasetup as cs
from niftypet.ninst.dinf import dev_info, gpuinfo
from niftypet.ninst.tools import LOG_FORMAT, LogHandler, path_resources, resources

# shared CUDA C library for extended auxiliary functions for the mMR
# > Siemens Biograph mMR
from . import img, lm, mmr_auxe, mmraux, mmrnorm, prj
from .img.mmrimg import align_mumap
from .img.mmrimg import convert2dev as im_e72dev
from .img.mmrimg import convert2e7 as im_dev2e7
from .img.mmrimg import get_cylinder, hdw_mumap, obj_mumap, pct_mumap
from .img.pipe import mmrchain
from .lm.mmrhist import dynamic_timings, mmrhist, randoms
from .mmraux import explore_input as classify_input
from .mmraux import mMR_params as get_mmrparams
from .mmraux import sino2ssr
from .prj.mmrprj import back_prj, frwd_prj
from .prj.mmrsim import simulate_recon, simulate_sino
from .sct.mmrsct import vsm

from .mmrnorm import get_norm_sino

# log = logging.getLogger(__name__)
# technically bad practice to add handlers
# https://docs.python.org/3/howto/logging.html#library-config
# log.addHandler(LogHandler())  # do it anyway for convenience

if resources.ENBLAGG:
    from .lm.pviews import video_dyn, video_frm
else:
    video_dyn, video_frm = None, None

if resources.ENBLXNAT:
    from xnat import xnat
else:
    xnat = None

# > GE Signa
# from . import aux_sig

# from . import lm_sig
# from .lm_sig.hst_sig import lminfo_sig

# for use in `cmake -DCMAKE_PREFIX_PATH=...`
cmake_prefix = resource_filename(__name__, "cmake")
