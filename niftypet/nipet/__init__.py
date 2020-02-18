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


class LogHandler(logging.StreamHandler):
    """Custom formatting and tqdm-compatibility"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        fmt = logging.Formatter(
            '%(levelname)s:%(asctime)s:%(name)s:%(funcName)s\n> %(message)s')
        self.setFormatter(fmt)

    def handleError(self, record):
        super().handleError(record)
        raise IOError(record)

    def emit(self, record):
        """Write to tqdm's stream so as to not break progress-bars"""
        try:
            msg = self.format(record)
            tqdm.write(
                msg, file=self.stream, end=getattr(self, "terminator", "\n"))
            self.flush()
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)


log = logging.getLogger(__name__)
# technically bad practice to add handlers
# https://docs.python.org/3/howto/logging.html#library-config
# but we'll do it anyway for convenience
log.addHandler(LogHandler())

# if using conda put the resources in the folder with the environment name
if 'CONDA_DEFAULT_ENV' in os.environ:
    try:
        env = re.findall('envs/(.*)/bin/python', sys.executable)[0]
    except IndexError:
        env = os.environ['CONDA_DEFAULT_ENV']
    log.info('conda environment found:' + env)
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
except ImportError as ie:
    raise ImportError(dedent('''\
        --------------------------------------------------------------------------
        NiftyPET resources file <resources.py> could not be imported.
        It should be in ~/.niftypet/resources.py (Linux) or
        in //Users//USERNAME//AppData//Local//niftypet//resources.py (Windows)
        but likely it does not exists.
        --------------------------------------------------------------------------'''))



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

from .dinf import gpuinfo, dev_info

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
from . import aux_sig

from . import lm_sig
from .lm_sig.hst_sig import lminfo_sig
