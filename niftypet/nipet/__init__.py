#!/usr/bin/env python
"""init the NiftyPET package"""
__author__      = "Pawel Markiewicz"
__copyright__   = "Copyright 2018"
# ---------------------------------------------------------------------------------

import os
import sys
import platform
import logging

log = logging.getLogger(__name__)

# if using conda put the resources in the folder with the environment name
if 'CONDA_DEFAULT_ENV' in os.environ:
    env = os.environ['CONDA_DEFAULT_ENV']
    log.debug('conda environment found:' + env)
else:
    env = ''
# create the path for the resources files according to the OS platform
if platform.system() in ['Linux', 'Darwin']:
    path_resources = os.path.join( os.path.join(os.path.expanduser('~'),   '.niftypet'), env )
elif platform.system() == 'Windows':
    path_resources = os.path.join( os.path.join(os.getenv('LOCALAPPDATA'), '.niftypet'), env )
else:
    log.error('unrecognised operating system!')
    
sys.path.append(path_resources)
try:
    import resources
except ImportError:
    log.error("""
    NiftyPET's resources file <resources.py> could not be imported.
    It should be in '~/.niftypet/resources.py' (Linux) or
    '/Users/USERNAME/AppData/Local/niftypet/resources.py' (Windows)
    but likely it does not exist.

    Tried to find in '%s'
    """ % path_resources)
    raise
#===========================


import mmraux
from mmraux import explore_input as classify_input
from mmraux import mMR_params as get_mmrparams

# shared CUDA C library for extended auxiliary functions for the mMR
import mmr_auxe

import mmrnorm
import lm
import prj
import sct
import img

from dinf import gpuinfo, dev_info

from img.mmrimg import hdw_mumap
from img.mmrimg import obj_mumap
from img.mmrimg import pct_mumap
from img.mmrimg import align_mumap
from img.mmrimg import convert2e7 as im_dev2e7
from img.mmrimg import convert2dev as im_e72dev

from img.pipe import mmrchain

from lm.mmrhist import dynamic_timings
from lm.mmrhist import mmrhist
from lm.mmrhist import randoms

from sct.mmrsct import vsm

from prj.mmrprj import frwd_prj
from prj.mmrprj import back_prj

from prj.mmrsim import simulate_sino, simulate_recon

if resources.ENBLAGG:
    from lm.pviews import video_frm, video_dyn

if resources.ENBLXNAT:
    from xnat import xnat

# import sigaux
# import lms
