#!/usr/bin/env python
"""init the NiftyPET package"""
__author__      = "Pawel Markiewicz"
__copyright__   = "Copyright 2018"
# ---------------------------------------------------------------------------------

import os
import sys
import platform

# if using conda put the resources in the folder with the environment name
if 'CONDA_DEFAULT_ENV' in os.environ:
	env = os.environ['CONDA_DEFAULT_ENV']
	print 'i> conda environment found:', env
else:
	env = ''
# create the path for the resources files according to the OS platform
if platform.system() == 'Linux' :
	path_resources = os.path.join( os.path.join(os.path.expanduser('~'),   '.niftypet'), env )
elif platform.system() == 'Windows' :
	path_resources = os.path.join( os.path.join(os.getenv('LOCALAPPDATA'), '.niftypet'), env )
else:
	print 'e> unrecognised operating system!  Linux and Windows operating systems only are supported.'
	
sys.path.append(path_resources)
try:
    import resources
except ImportError as ie:
    print '----------------------------'
    print 'e> Import Error: NiftyPET''s resources file <resources.py> could not be imported.  It should be in ''~/.niftypet/resources.py'' (Linux) or ''//Users//USERNAME//AppData//Local//niftypet//resources.py'' (Windows) but likely it does not exists.'
    print '----------------------------'
    raise ImportError
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

from lm.mmrhist import mmrhist
from img.pipe import mmrchain

from prj.mmrprj import frwd_prj
from prj.mmrprj import back_prj

from prj.mmrsim import simulate_sino, simulate_recon

if resources.ENBLAGG:
	from lm.pviews import video_frm, video_dyn

if resources.ENBLXNAT:
	from xnat import xnat

# import sigaux
# import lms