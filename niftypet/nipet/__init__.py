#!/usr/bin/env python
"""initialise the NiftyPET NIPET package"""
__author__      = ("Pawel J. Markiewicz", "Casper O. da Costa-Luis")
__copyright__   = "Copyright 2020"

import logging
import os
import platform
import re
import sys
from textwrap import dedent

from pkg_resources import resource_filename
from tqdm.auto import tqdm

from niftypet.ninst import cudasetup as cs
from niftypet.ninst.dinf import dev_info, gpuinfo
from niftypet.ninst.tools import LOG_FORMAT, LogHandler, path_resources, resources

# shared CUDA C library for extended auxiliary functions for the mMR
#> Siemens Biograph mMR
from . import img, lm, mmr_auxe, mmraux, mmrnorm, prj
from .img.mmrimg import align_mumap
from .img.mmrimg import convert2dev as im_e72dev
from .img.mmrimg import convert2e7 as im_dev2e7
from .img.mmrimg import hdw_mumap, obj_mumap, pct_mumap
from .img.pipe import mmrchain
from .lm.mmrhist import dynamic_timings, mmrhist, randoms
from .mmraux import explore_input as classify_input
from .mmraux import mMR_params as get_mmrparams
from .prj.mmrprj import back_prj, frwd_prj
from .prj.mmrsim import simulate_recon, simulate_sino
from .sct.mmrsct import vsm

# log = logging.getLogger(__name__)
# technically bad practice to add handlers
# https://docs.python.org/3/howto/logging.html#library-config
# log.addHandler(LogHandler())  # do it anyway for convenience










if resources.ENBLAGG:
    from .lm.pviews import video_dyn, video_frm

if resources.ENBLXNAT:
    from xnat import xnat


#> GE Signa
#from . import aux_sig

#from . import lm_sig
#from .lm_sig.hst_sig import lminfo_sig

# for use in `cmake -DCMAKE_PREFIX_PATH=...`
cmake_prefix = resource_filename(__name__, "cmake")
