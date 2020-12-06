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

from tqdm.auto import tqdm

from niftypet.ninst.tools import LogHandler
from niftypet.ninst import cudasetup as cs

log = logging.getLogger(__name__)
# technically bad practice to add handlers
# https://docs.python.org/3/howto/logging.html#library-config
# log.addHandler(LogHandler())  # do it anyway for convenience

path_resources = cs.path_niftypet_local()
resources = cs.get_resources()

from niftypet.ninst.dinf import gpuinfo, dev_info

from . import prj
from . import img

#> Siemens Biograph mMR
from . import mmrnorm
from . import mmraux
from .mmraux import explore_input as classify_input
from .mmraux import mMR_params as get_mmrparams

# shared CUDA C library for extended auxiliary functions for the mMR
from . import mmr_auxe

from . import lm
from .lm.mmrhist import dynamic_timings, mmrhist, randoms

from .img.mmrimg import hdw_mumap, obj_mumap, pct_mumap, align_mumap
from .img.mmrimg import convert2e7 as im_dev2e7
from .img.mmrimg import convert2dev as im_e72dev
from .img.pipe import mmrchain

from .sct.mmrsct import vsm

from .prj.mmrprj import frwd_prj, back_prj

from .prj.mmrsim import simulate_sino, simulate_recon

if resources.ENBLAGG:
    from .lm.pviews import video_frm, video_dyn

if resources.ENBLXNAT:
    from xnat import xnat


#> GE Signa
#from . import aux_sig

#from . import lm_sig
#from .lm_sig.hst_sig import lminfo_sig
