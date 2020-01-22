#!/usr/bin/env python
""" Compile CUDA source code and setup Python package 'nipet'
    for namespace package 'niftypet'.
"""
__author__      = "Pawel J. Markiewicz"
__copyright__   = "Copyright 2018"
# ---------------------------------------------------------------------------------

from setuptools import setup, find_packages

import os
import sys
import platform
from subprocess import Popen, PIPE
import logging

if 'DISPLAY' in os.environ:
    from Tkinter import Tk
    from tkFileDialog import askdirectory
else:
    def askdirectory(title='Folder: ', initialdir=os.path.expanduser('~'), name=''):
        """
        decreasing precedence: os.environ[name], raw_input, initialdir
        """
        path = os.environ.get(name, None)
        if path is None:
            path = raw_input(title)
        if path == '':
            return initialdir
        return path


import cudasetup_hdr as cs

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()

#-------------------------------------------------------------------------
# The below function is a copy of the same function in install_tools.py
# in NIMPA
def update_resources(Cnt):
    '''Update resources.py with the paths to the new installed apps.
    '''
    # list of path names which will be saved
    key_list = ['PATHTOOLS', 'RESPATH', 'REGPATH', 'DCM2NIIX', 'HMUDIR']

    # get the local path to NiftyPET resources.py
    path_resources = cs.path_niftypet_local()
    resources_file = os.path.join(path_resources,'resources.py')

    # update resources.py
    if os.path.isfile(resources_file):
        f = open(resources_file, 'r')
        rsrc = f.read()
        f.close()
        # get the region of keeping in synch with Python
        i0 = rsrc.find('### start NiftyPET tools ###')
        i1 = rsrc.find('### end NiftyPET tools ###')
        pth_list = []
        for k in key_list:
            if k in Cnt:
                pth_list.append('\'' + Cnt[k].replace("\\","/") + '\'')
            else:
                pth_list.append('\'\'')

        # modify resources.py with the new paths
        strNew = '### start NiftyPET tools ###\n'
        for i in range(len(key_list)):
            if pth_list[i] != '\'\'':
                strNew += key_list[i]+' = '+pth_list[i] + '\n'
        rsrcNew = rsrc[:i0] + strNew + rsrc[i1:]
        f = open(resources_file, 'w')
        f.write(rsrcNew)
        f.close()

    return Cnt
#-------------------------------------------------------------------------

if 'Windows' not in platform.system() and 'Linux' not in platform.system():
    log.error('the current operating system is not supported.')
    raise SystemError('OS: Unknown Sysytem.')

#----------------------------------------------------
# select the supported GPU device and install resources.py
log.info('setting up CUDA...')
gpuarch = cs.resources_setup()
#----------------------------------------------------



#===============================================================
# Hardware mu-maps
log.info('indicate the location of hardware mu-maps:')

#---------------------------------------------------------------
# get the local path to NiftyPET resources.py
path_resources = cs.path_niftypet_local()
# if exists, import the resources and get the constants
if os.path.isfile(os.path.join(path_resources,'resources.py')):
    sys.path.append(path_resources)
    try:
        import resources
    except ImportError:
        log.error("NiftyPET's resources file <resources.py> could not be imported")
        raise
    # get the current setup, if any
    Cnt = resources.get_setup()

    # assume the hardware mu-maps are not installed
    hmu_flg = False
    # go through each piece of the hardware components
    if 'HMUDIR' in Cnt and Cnt['HMUDIR']!='':
        for hi in Cnt['HMULIST']:
            if os.path.isfile(os.path.join(Cnt['HMUDIR'],hi)):
                hmu_flg = True
            else:
                hmu_flg = False
                break
    # if not installed ask for the folder through GUI
    # otherwise the path will have to be filled manually
    if not hmu_flg:
        prompt = dict(title='Folder for hardware mu-maps',
                      initialdir=os.path.expanduser('~'))
        if 'DISPLAY' in os.environ:
            Tk().withdraw()
        else:
            prompt['name'] = 'HMUDIR'
        Cnt['HMUDIR'] = askdirectory(**prompt)
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # update the path in resources.py
    update_resources(Cnt)
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
else:
    raise SystemError('Missing file: resources.py')

log.info('hardware mu-maps have been located')
#===============================================================



#===============================================================
log.info('CUDA compilation for NIPET')

path_current = os.path.dirname( os.path.realpath(__file__) )
path_build = os.path.join(path_current, 'build')
if not os.path.isdir(path_build): os.makedirs(path_build)
os.chdir(path_build)

# cmake installation commands
cmd = []
cmd.append([
    'cmake',
    os.path.join('..','niftypet'),
    '-DPYTHON_INCLUDE_DIRS=' + cs.pyhdr,
    '-DPYTHON_PREFIX_PATH='+cs.prefix,
    '-DCUDA_NVCC_FLAGS='+gpuarch
])
cmd.append(['cmake', '--build', './'])

if platform.system()=='Windows':
    cmd[0] += ['-G', Cnt['MSVC_VRSN']]
    cmd[1] += ['--config', 'Release']

# error string for later reporting
errstr = []
# the log files the cmake results are written
cmakelog = ['py_cmake_config.log', 'py_cmake_build.log']
# run commands with logging
for ci in range(len(cmd)):
    with open(cmakelog[ci], 'w') as f:
        p = Popen(cmd[ci], stdout=PIPE, stderr=PIPE)
        for c in iter(lambda: p.stdout.read(1), ''):
            sys.stdout.write(c)
            f.write(c)
    # get the pipes outputs
    stdout, stderr = p.communicate()
    ei = stderr.find('error')
    if ei>=0:
        errstr.append(stderr[ei:ei+60]+'...')
    else:
        errstr.append('_')

    if stderr:
        log.warning('-------- reports -----------')
        log.warning(stderr)
        log.warning('------------ end ---------------')

    log.info(stdout)


log.info('--- error report ---')
for ci in range(len(cmd)):
    if errstr[ci] != '_':
        log.error('found error(s) in ' + ' '.join(cmd[ci]) + ' >> ' + errstr[ci])
log.info('--- end ---')

# come back from build folder
os.chdir(path_current)
#===============================================================





#===============================================================
# PYTHON SETUP
#===============================================================

log.info('found those packages:')
log.info(find_packages(exclude=['docs']))

with open('README.rst') as file:
    long_description = file.read()

#---- for setup logging -----
stdout = sys.stdout
stderr = sys.stderr
log_file = open('setup_nipet.log', 'w')
sys.stdout = log_file
sys.stderr = log_file
#----------------------------

if platform.system() == 'Linux' :
    fex = '*.so'
elif platform.system() == 'Windows' :
    fex = '*.pyd'
#----------------------------
setup(
    name='nipet',
    license = 'Apache 2.0',
    version='1.1.20',
    description='CUDA-accelerated Python utilities for high-throughput PET/MR image reconstruction and analysis.',
    long_description=long_description,
    author='Pawel J. Markiewicz',
    author_email='p.markiewicz@ucl.ac.uk',
    url='https://github.com/NiftyPET/NiftyPET',
    keywords='PET image reconstruction and analysis',
    install_requires=['nimpa>=1.1.0', 'pydicom>=1.0.2,<=1.2.2',
      'nibabel>=2.2.1, <=2.3.1', 'tqdm>=4.27'],
    python_requires='<3.0.0',
    packages=find_packages(exclude=['docs']),
    package_data={
        'niftypet': ['auxdata/*'],
        'niftypet.nipet.dinf': [fex],
        'niftypet.nipet.lm' : [fex],
        'niftypet.nipet.prj' : [fex],
        'niftypet.nipet.sct' : [fex],
        'niftypet.nipet' : [fex],
    },
    zip_safe=False,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Education',
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: C',
        'Programming Language :: C++',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
    ],
    # namespace_packages=['niftypet'],
)
#===============================================================
