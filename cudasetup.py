#!/usr/bin/env python
"""Tools for CUDA compilation and set-up for Python 3."""
from distutils.sysconfig import get_python_inc
import logging
import os
from pkg_resources import resource_filename
import platform
import re
import shutil
import subprocess
import sys
from textwrap import dedent

import numpy as np
__author__      = ("Pawel J. Markiewicz", "Casper O. da Costa-Luis")
__copyright__   = "Copyright 2020"
log = logging.getLogger('nipet.cudasetup')

prefix = sys.prefix
pyhdr = get_python_inc()  # Python header paths
nphdr = np.get_include()  # numpy header path
mincc = 35  # minimum required CUDA compute capability


def path_niftypet_local():
    '''Get the path to the local (home) folder for NiftyPET resources.'''
    # if using conda put the resources in the folder with the environment name
    if 'CONDA_DEFAULT_ENV' in os.environ:
        env = os.environ['CONDA_DEFAULT_ENV']
        log.info('install> conda environment found: {}'.format(env))
    else:
        env = ''
    # create the path for the resources files according to the OS platform
    if platform.system() in ('Linux', 'Darwin'):
        path_resources = os.path.expanduser('~')
    elif platform.system() == 'Windows' :
        path_resources = os.getenv('LOCALAPPDATA')
    else:
        raise ValueError('Unknown operating system: {}'.format(platform.system()))
    path_resources = os.path.join(path_resources, '.niftypet', env)

    return path_resources


def find_cuda():
    '''Locate the CUDA environment on the system.'''
    # search the PATH for NVCC
    for fldr in os.environ['PATH'].split(os.pathsep):
        cuda_path = os.path.join(fldr, 'nvcc')
        if os.path.exists(cuda_path):
            cuda_path = os.path.dirname(os.path.dirname(cuda_path))
            break
        cuda_path = None

    if cuda_path is None:
        log.warning('nvcc compiler could not be found from the PATH!')
        return None

    # serach for the CUDA library path
    lcuda_path = os.path.join(cuda_path, 'lib64')
    if 'LD_LIBRARY_PATH' in os.environ:
        if lcuda_path in os.environ['LD_LIBRARY_PATH'].split(os.pathsep):
            log.info('found CUDA lib64 in LD_LIBRARY_PATH: {}'.format(lcuda_path))
    elif os.path.isdir(lcuda_path):
        log.info('found CUDA lib64 in: {}'.format(lcuda_path))
    else:
        log.warning('folder for CUDA library (64-bit) could not be found!')

    return cuda_path, lcuda_path


def dev_setup():
    '''figure out what GPU devices are available and choose the supported ones.'''
    # check first if NiftyPET was already installed and use the choice of GPU
    path_resources = path_niftypet_local()
    # if so, import the resources and get the constants
    if os.path.isfile(os.path.join(path_resources,'resources.py')):
        sys.path.append(path_resources)
        try:
            import resources
        except ImportError as ie:
            log.error(dedent('''\
                --------------------------------------------------------------------------
                NiftyPET resources file <resources.py> could not be imported.
                It should be in ~/.niftypet/resources.py (Linux) or
                in //Users//USERNAME//AppData//Local//niftypet//resources.py (Windows)
                but likely it does not exists.
                --------------------------------------------------------------------------'''))
    else:
        log.error('resources file not found/installed.')
        return None

    # get all constants and check if device is already chosen
    Cnt = resources.get_setup()
    if 'CCARCH' in Cnt and 'DEVID' in Cnt:
        log.info('using this CUDA architecture(s): {}'.format(Cnt['CCARCH']))
        return Cnt['CCARCH']

    # get the current locations
    path_current = os.path.dirname( os.path.realpath(__file__) )
    path_resins = os.path.join(path_current, 'resources')
    path_dinf = os.path.join(path_current, 'niftypet')
    path_dinf = os.path.join(path_dinf, 'nipet')
    path_dinf = os.path.join(path_dinf, 'dinf')
    # temporary installation location for identifying the CUDA devices
    path_tmp_dinf = os.path.join(path_resins,'dinf')
    # if the folder 'path_tmp_dinf' exists, delete it
    if os.path.isdir(path_tmp_dinf):
        shutil.rmtree(path_tmp_dinf)
    # copy the device_info module to the resources folder within the installation package
    shutil.copytree( path_dinf, path_tmp_dinf)
    # create a build using cmake
    path_tmp_build = os.path.join(path_tmp_dinf, 'build')
    os.makedirs(path_tmp_build)
    os.chdir(path_tmp_build)
    if platform.system()=='Windows':
        subprocess.call(
            ['cmake', '../', '-DPYTHON_INCLUDE_DIRS='+pyhdr,
             '-DPYTHON_PREFIX_PATH='+prefix, '-G', Cnt['MSVC_VRSN']]
        )
        subprocess.call(['cmake', '--build', './', '--config', 'Release'])
        path_tmp_build = os.path.join(path_tmp_build, 'Release')

    elif platform.system() in ['Linux', 'Darwin']:
        subprocess.call(
            ['cmake', '../', '-DPYTHON_INCLUDE_DIRS='+pyhdr,
             '-DPYTHON_PREFIX_PATH='+prefix]
        )
        subprocess.call(['cmake', '--build', './'])
    else:
        log.error('Unknown operating system: {}'.format(platform.system()))
        return None

    # import the new module for device properties
    sys.path.insert(0, path_tmp_build)
    import dinf

    # get the list of installed CUDA devices
    Ldev = dinf.dev_info(0)
    if len(Ldev)==0:
        raise IOError('No CUDA devices have been detected')
    # extract the compute capability as a single number
    cclist = [int(str(e[2])+str(e[3])) for e in Ldev]
    # get the list of supported CUDA devices (with minimum compute capability)
    spprtd = [str(cc) for cc in cclist if cc>=mincc]
    if not spprtd:
        log.error('installed devices have the compute capability of: {}'.format(spprtd))
        raise IOError('No supported CUDA devices have been found.')
    # best for the default CUDA device
    i = [int(s) for s in spprtd]
    devid = i.index(max(i))
    #-----------------------------------------------------------------------------------
    # form return list of compute capability numbers for which the software will be compiled
    ccstr = ''
    for cc in spprtd:
        ccstr += '-gencode=arch=compute_'+cc+',code=compute_'+cc+';'
    #-----------------------------------------------------------------------------------

    # remove the temporary path
    sys.path.remove(path_tmp_build)
    # delete the build once the info about the GPUs has been obtained
    os.chdir(path_current)
    shutil.rmtree(path_tmp_dinf, ignore_errors=True)

    # passing this setting to resources.py
    fpth = os.path.join(path_resources,'resources.py') #resource_filename(__name__, 'resources/resources.py')
    f = open(fpth, 'r')
    rsrc = f.read()
    f.close()
    # get the region of keeping in synch with Python
    i0 = rsrc.find('### start GPU properties ###')
    i1 = rsrc.find('### end GPU properties ###')
    # list of constants which will be kept in sych from Python
    cnt_list = ['DEV_ID', 'CC_ARCH']
    val_list = [str(devid), '\''+ccstr+'\'']
    # update the resource.py file
    strNew = '### start GPU properties ###\n'
    for i in range(len(cnt_list)):
        strNew += cnt_list[i]+' = '+val_list[i] + '\n'
    rsrcNew = rsrc[:i0] + strNew + rsrc[i1:]
    f = open(fpth, 'w')
    f.write(rsrcNew)
    f.close()

    return ccstr



#=================================================================================================
# automatically detects if the CUDA header files are in agreement with Python constants.
#=================================================================================================


def chck_vox_h(Cnt):
    '''check if voxel size in Cnt and adjust the CUDA header files accordingly.'''
    rflg = False
    fpth = resource_filename(__name__, 'niftypet/nipet/def.h')
    f = open(fpth, 'r')
    def_h = f.read()
    f.close()
    # get the region of keeping in synch with Python
    i0 = def_h.find('//## start ##//')
    i1 = def_h.find('//## end ##//')
    defh = def_h[i0:i1]
    # list of constants which will be kept in synch from Python
    cnt_list = ['SZ_IMX', 'SZ_IMY',  'SZ_IMZ',
                'TFOV2',  'SZ_VOXY', 'SZ_VOXZ', 'SZ_VOXZi']
    flg = False
    for s in cnt_list:
        m = re.search('(?<=#define ' + s + r')\s*\d*\.*\d*', defh)
        if s[3]=='V':
            #print(s, float(m.group(0)), Cnt[s])
            if Cnt[s]!=float(m.group(0)):
                flg = True
                break
        else:
            #print(s, int(m.group(0)), Cnt[s])
            if Cnt[s]!=int(m.group(0)):
                flg = True
                break
    # if flag is set then redefine the constants in the sct.h file
    if flg:
        strNew = '//## start ##// constants definitions in synch with Python.   DON''T MODIFY MANUALLY HERE!\n'+\
        '// IMAGE SIZE\n'+\
        '// SZ_I* are image sizes\n'+\
        '// SZ_V* are voxel sizes\n'
        strDef = '#define '
        for s in cnt_list:
            strNew += strDef+s+' '+str(Cnt[s])+(s[3]=='V')*'f' + '\n'

        scthNew = def_h[:i0] + strNew + def_h[i1:]
        f = open(fpth, 'w')
        f.write(scthNew)
        f.close()
        rflg = True

    return rflg


def chck_sct_h(Cnt):
    '''
    check if voxel size for scatter correction changed and adjust
    the CUDA header files accordingly.
    '''
    rflg = False
    fpth = resource_filename(__name__, 'niftypet/nipet/sct/src/sct.h')
    #pthcmpl = os.path.dirname(resource_filename(__name__, ''))
    f = open(fpth, 'r')
    sct_h = f.read()
    f.close()
    # get the region of keeping in synch with Python
    i0 = sct_h.find('//## start ##//')
    i1 = sct_h.find('//## end ##//')
    scth = sct_h[i0:i1]
    # list of constants which will be kept in sych from Python
    cnt_list = ['SS_IMX', 'SS_IMY', 'SS_IMZ',
                'SSE_IMX', 'SSE_IMY', 'SSE_IMZ', 'NCOS',
                'SS_VXY',  'SS_VXZ',  'IS_VXZ', 'SSE_VXY', 'SSE_VXZ',
                'R_RING', 'R_2', 'IR_RING',  'SRFCRS']
    flg = False
    for i,s in enumerate(cnt_list):
        m = re.search('(?<=#define ' + s + r')\s*\d*\.*\d*', scth)
        # if s[-3]=='V':
        if i<7:
            #print(s, int(m.group(0)), Cnt[s])
            if Cnt[s]!=int(m.group(0)):
                flg = True
                break
        else:
            #print(s, float(m.group(0)), Cnt[s])
            if Cnt[s]!=float(m.group(0)):
                flg = True
                break

    # if flag is set then redefine the constants in the sct.h file
    if flg:
        strNew = dedent('''\
            //## start ##// constants definitions in synch with Python.   DO NOT MODIFY!\n
            // SCATTER IMAGE SIZE AND PROPERTIES
            // SS_* are used for the mu-map in scatter calculations
            // SSE_* are used for the emission image in scatter calculations
            // R_RING, R_2, IR_RING are ring radius, squared radius and inverse of the radius, respectively.
            // NCOS is the number of samples for scatter angular sampling
            ''')

        strDef = '#define '
        for i,s in enumerate(cnt_list):
            strNew += strDef+s+' '+str(Cnt[s])+(i>6)*'f' + '\n'

        scthNew = sct_h[:i0] + strNew + sct_h[i1:]
        f = open(fpth, 'w')
        f.write(scthNew)
        f.close()
        #sys.path.append(pthcmpl)
        rflg = True

    return rflg


def check_constants():
    '''get the constants for the mMR from the resources file before
    getting the path to the local resources.py (on Linux machines it is in ~/.niftypet)'''
    path_resources = path_niftypet_local()
    # import resource
    sys.path.append(path_resources)
    try:
        import resources
    except ImportError as ie:
        log.error(dedent('''\
            --------------------------------------------------------------------------
            NiftyPET resources file <resources.py> could not be imported.
            It should be in ~/.niftypet/resources.py (Linux) or
            in //Users//USERNAME//AppData//Local//niftypet//resources.py (Windows)
            but likely it does not exists.
            --------------------------------------------------------------------------'''))
        raise ImportError('Could not find resources.py')
    #===========================
    Cnt = resources.get_mmr_constants()

    sct_compile = chck_sct_h(Cnt)
    def_compile = chck_vox_h(Cnt)
    # sct_compile = False
    # def_compile = False

    if sct_compile or def_compile:
        txt = 'NiftyPET constants were changed: needs CUDA compilation.'
    else:
        txt = '- - . - -'

    log.info(dedent('''\
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        changed sct.h: {}
        changed def.h: {}
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        {}
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~''').format(
        sct_compile, def_compile, txt))


def resources_setup():
    '''
    This function checks CUDA devices, selects some and installs resources.py
    '''
    log.info('installing file <resources.py> into home directory if it does not exist.')
    path_current = os.path.dirname( os.path.realpath(__file__) )
    # path to the install version of resources.py.
    path_install = os.path.join(path_current, 'resources')
    # get the path to the local resources.py (on Linux machines it is in ~/.niftypet)
    path_resources = path_niftypet_local()
    log.info('current path: {}'.format(path_current))

    # flag for the resources file if already installed (initially assumed not)
    flg_resources = False
    # does the local folder for niftypet exists? if not create one.
    if not os.path.exists(path_resources):
        os.makedirs(path_resources)
    # is resources.py in the folder?
    if not os.path.isfile(os.path.join(path_resources,'resources.py')):
        if os.path.isfile(os.path.join(path_install,'resources.py')):
            shutil.copyfile( os.path.join(path_install,'resources.py'), os.path.join(path_resources,'resources.py') )
        else:
            raise IOError('could not find <resources.py')
    else:
        log.info('<resources.py> should be already in the local NiftyPET folder:\n  {}'.format(path_resources))
        # set the flag that the resources file is already there
        flg_resources = True
        sys.path.append(path_resources)
        try:
            import resources
        except ImportError as ie:
            log.error(dedent('''\
                --------------------------------------------------------------------------
                NiftyPET resources file <resources.py> could not be imported.
                It should be in ~/.niftypet/resources.py (Linux) or
                in //Users//USERNAME//AppData//Local//niftypet//resources.py (Windows)
                but likely it does not exists.
                --------------------------------------------------------------------------'''))

    # check and update the constants in C headers according to resources.py
    check_constants()

    # find available GPU devices, select one or more and output the compilation flags
    gpuarch = dev_setup()

    # return gpuarch for cmake compilation
    return gpuarch
