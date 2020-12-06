#!/usr/bin/env python3
"""
Compile CUDA source code and setup Python 3 package 'nipet'
for namespace 'niftypet'.
"""
import logging
import os
import platform
import re
from setuptools import setup, find_packages
from subprocess import run, PIPE
import sys
from textwrap import dedent

from niftypet.ninst import cudasetup as cs
from niftypet.ninst import install_tools as tls
from niftypet.ninst.tools import LogHandler
__author__ = ("Pawel J. Markiewicz", "Casper O. da Costa-Luis")
__copyright__ = "Copyright 2020"
__licence__ = __license__ = "Apache 2.0"

logging.basicConfig(level=logging.INFO)
logroot = logging.getLogger('nipet')
logroot.addHandler(LogHandler())
log = logging.getLogger('nipet.setup')

tls.check_platform()
ext = tls.check_depends()  # external dependencies



#=================================================================================================
# automatically detects if the CUDA header files are in agreement with Python constants.
#=================================================================================================


def chck_vox_h(Cnt):
    '''check if voxel size in Cnt and adjust the CUDA header files accordingly.'''
    rflg = False
    path_current = os.path.dirname(os.path.realpath(__file__))
    fpth = os.path.join(path_current, "niftypet", "nipet", "def.h")
    with open(fpth, 'r') as fd:
        def_h = fd.read()
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
        with open(fpth, 'w') as fd:
            fd.write(scthNew)
        rflg = True

    return rflg


def chck_sct_h(Cnt):
    '''
    check if voxel size for scatter correction changed and adjust
    the CUDA header files accordingly.
    '''
    rflg = False
    path_current = os.path.dirname(os.path.realpath(__file__))
    fpth = os.path.join(path_current, "niftypet", "nipet", "sct", "src", "sct.h")
    #pthcmpl = os.path.dirname(resource_filename(__name__, ''))
    with open(fpth, 'r') as fd:
        sct_h = fd.read()
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
        with open(fpth, 'w') as fd:
            fd.write(scthNew)
        #sys.path.append(pthcmpl)
        rflg = True

    return rflg


def check_constants():
    '''get the constants for the mMR from the resources file before
    getting the path to the local resources.py (on Linux machines it is in ~/.niftypet)'''
    resources = cs.get_resources()
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


if ext["cuda"] and ext["cmake"]:
    cs.resources_setup(gpu=False)  # install resources.py
    # check and update the constants in C headers according to resources.py
    check_constants()
    gpuarch = cs.dev_setup()  # update resources.py with a supported GPU device
else:
    raise SystemError("Need nvcc and cmake")


log.info(
    dedent(
        """
        --------------------------------------------------------------
        Finding hardware mu-maps
        --------------------------------------------------------------"""
    )
)

# get the local path to NiftyPET resources.py
path_resources = cs.path_niftypet_local()
# if exists, import the resources and get the constants
resources = cs.get_resources()
# get the current setup, if any
Cnt = resources.get_setup()

if True:
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
        prompt = dict(title='Folder for hardware mu-maps: ',
                      initialdir=os.path.expanduser('~'))
        if not os.getenv("DISPLAY", False):
            prompt["name"] = "HMUDIR"
        Cnt['HMUDIR'] = tls.askdirectory(**prompt)
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # update the path in resources.py
    tls.update_resources(Cnt)
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

log.info('hardware mu-maps have been located')


# CUDA installation
if True:
    log.info(
        dedent(
            """
            --------------------------------------------------------------
            CUDA compilation for NIPET ...
            --------------------------------------------------------------"""
        )
    )

    path_current = os.path.dirname(os.path.realpath(__file__))
    path_build = os.path.join(path_current, "build")
    if not os.path.isdir(path_build):
        os.makedirs(path_build)
    os.chdir(path_build)

    # cmake installation commands
    cmds = [
        [
            "cmake",
            os.path.join("..", "niftypet"),
            "-DPYTHON_INCLUDE_DIRS=" + cs.pyhdr,
            "-DPYTHON_PREFIX_PATH=" + cs.prefix,
            "-DCUDA_NVCC_FLAGS=" + gpuarch,
        ],
        ["cmake", "--build", "./"]
    ]

    if platform.system() == "Windows":
        cmds[0] += ["-G", Cnt["MSVC_VRSN"]]
        cmds[1] += ["--config", "Release"]

    # error string for later reporting
    errs = []
    # the log files the cmake results are written
    cmakelogs = ["nipet_cmake_config.log", "nipet_cmake_build.log"]
    # run commands with logging
    for cmd, cmakelog in zip(cmds, cmakelogs):
        p = run(cmd, stdout=PIPE, stderr=PIPE)
        stdout = p.stdout.decode("utf-8")
        stderr = p.stderr.decode("utf-8")

        with open(cmakelog, "w") as fd:
            fd.write(stdout)

        ei = stderr.find("error")
        if ei >= 0:
            errs.append(stderr[ei : ei + 60] + "...")
        else:
            errs.append("_")

        if p.stderr:
            log.warning(
                dedent(
                    """
                    ---------- process warnings/errors ------------
                    {}
                    --------------------- end ---------------------"""
                ).format(
                    stderr
                )
            )

        log.info(
            dedent(
                """
                ---------- compilation output ------------
                {}
                ------------------- end ------------------"""
            ).format(stdout)
        )

    log.info("\n------------- error report -------------")
    for cmd, err in zip(cmds, errs):
        if err != "_":
            log.error(" found error(s) in %s >> %s", " ".join(cmd), err)
    log.info("------------------ end -----------------")

    # come back from build folder
    os.chdir(path_current)
#===============================================================



#===============================================================
# PYTHON SETUP
#===============================================================
log.info('''found those packages:\n{}'''.format(find_packages(exclude=['docs'])))

freadme = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'README.rst')
log.info('''\
    \rUsing this README file:
    {}
    '''.format(freadme))

with open(freadme) as file:
    long_description = file.read()

#---- for setup logging -----
stdout = sys.stdout
stderr = sys.stderr
log_file = open('setup_nipet.log', 'w')
sys.stdout = log_file
sys.stderr = log_file
#----------------------------

if platform.system() in ['Linux', 'Darwin'] :
    fex = '*.so'
elif platform.system() == 'Windows' :
    fex = '*.pyd'
#----------------------------
setup(
    name='nipet',
    license=__licence__,
    version='2.0.0',
    description='CUDA-accelerated Python utilities for high-throughput PET/MR image reconstruction and analysis.',
    long_description=long_description,
    author=__author__[0],
    author_email='p.markiewicz@ucl.ac.uk',
    url='https://github.com/NiftyPET/NiftyPET',
    keywords='PET image reconstruction and analysis',
    python_requires='>=3.6',
    packages=find_packages(exclude=['docs']),
    package_data={
        'niftypet': ['auxdata/*'],
        'niftypet.nipet.lm' : [fex],
        'niftypet.nipet.prj' : [fex],
        'niftypet.nipet.sct' : [fex],
        'niftypet.nipet' : [fex],
    },
    zip_safe=False,
    # namespace_packages=['niftypet'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Education',
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: C',
        'Programming Language :: C++',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
    ],
)
