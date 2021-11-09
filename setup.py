#!/usr/bin/env python3
"""
Compile CUDA source code and setup Python 3 package 'nipet'
for namespace 'niftypet'.
"""
import logging
import re
import sys
from pathlib import Path
from textwrap import dedent

from setuptools import find_packages
from setuptools_scm import get_version
from skbuild import setup

from niftypet.ninst import cudasetup as cs
from niftypet.ninst import install_tools as tls

__version__ = get_version(root=".", relative_to=__file__)

logging.basicConfig(level=logging.INFO, format=tls.LOG_FORMAT)
log = logging.getLogger("nipet.setup")
path_current = Path(__file__).resolve().parent

tls.check_platform()

# =================================================================================================
# automatically detects if the CUDA header files are in agreement with Python constants.
# =================================================================================================


def chck_vox_h(Cnt):
    """check if voxel size in Cnt and adjust the CUDA header files accordingly."""
    rflg = False
    fpth = path_current / "niftypet" / "nipet" / "include" / "def.h"
    def_h = fpth.read_text()
    # get the region of keeping in synch with Python
    i0 = def_h.find("//## start ##//")
    i1 = def_h.find("//## end ##//")
    defh = def_h[i0:i1]
    # list of constants which will be kept in synch from Python
    cnt_list = [
        "SZ_IMX", "SZ_IMY", "SZ_IMZ", "TFOV2", "SZ_VOXY", "SZ_VOXZ", "SZ_VOXZi", "RSZ_PSF_KRNL"]
    flg = False
    for s in cnt_list:
        m = re.search("(?<=#define " + s + r")\s*\d*\.*\d*", defh)
        if s[3] == "V":
            # print(s, float(m.group(0)), Cnt[s])
            if Cnt[s] != float(m.group(0)):
                flg = True
                break
        else:
            # print(s, int(m.group(0)), Cnt[s])
            if Cnt[s] != int(m.group(0)):
                flg = True
                break
    # if flag is set then redefine the constants in the sct.h file
    if flg:
        strNew = ("//## start ##// constants definitions in synch with Python.   DON"
                  "T MODIFY MANUALLY HERE!\n" + "// IMAGE SIZE\n" + "// SZ_I* are image sizes\n" +
                  "// SZ_V* are voxel sizes\n")
        strDef = "#define "
        for s in cnt_list:
            strNew += strDef + s + " " + str(Cnt[s]) + (s[3] == "V") * "f" + "\n"

        fpth.write_text(def_h[:i0] + strNew + def_h[i1:])
        rflg = True

    return rflg


def chck_sct_h(Cnt):
    """
    check if voxel size for scatter correction changed and adjust
    the CUDA header files accordingly.
    """
    rflg = False
    fpth = path_current / "niftypet" / "nipet" / "sct" / "src" / "sct.h"
    # pthcmpl = path.dirname(resource_filename(__name__, ''))
    sct_h = fpth.read_text()
    # get the region of keeping in synch with Python
    i0 = sct_h.find("//## start ##//")
    i1 = sct_h.find("//## end ##//")
    scth = sct_h[i0:i1]
    # list of constants which will be kept in sych from Python
    cnt_list = [
        "SS_IMX", "SS_IMY", "SS_IMZ", "SSE_IMX", "SSE_IMY", "SSE_IMZ", "NCOS", "SS_VXY", "SS_VXZ",
        "IS_VXZ", "SSE_VXY", "SSE_VXZ", "R_RING", "R_2", "IR_RING", "SRFCRS"]
    flg = False
    for i, s in enumerate(cnt_list):
        m = re.search("(?<=#define " + s + r")\s*\d*\.*\d*", scth)
        # if s[-3]=='V':
        if i < 7:
            # print(s, int(m.group(0)), Cnt[s])
            if Cnt[s] != int(m.group(0)):
                flg = True
                break
        else:
            # print(s, float(m.group(0)), Cnt[s])
            if Cnt[s] != float(m.group(0)):
                flg = True
                break

    # if flag is set then redefine the constants in the sct.h file
    if flg:
        strNew = dedent("""\
            //## start ##// constants definitions in synch with Python.   DO NOT MODIFY!\n
            // SCATTER IMAGE SIZE AND PROPERTIES
            // SS_* are used for the mu-map in scatter calculations
            // SSE_* are used for the emission image in scatter calculations
            // R_RING, R_2, IR_RING: ring radius, squared radius, inverse radius
            // NCOS is the number of samples for scatter angular sampling
            """)

        strDef = "#define "
        for i, s in enumerate(cnt_list):
            strNew += strDef + s + " " + str(Cnt[s]) + (i > 6) * "f" + "\n"

        fpth.write_text(sct_h[:i0] + strNew + sct_h[i1:])
        # sys.path.append(pthcmpl)
        rflg = True

    return rflg


def check_constants():
    """get the constants for the mMR from the resources file before
    getting the path to the local resources.py (on Linux machines it is in ~/.niftypet)"""
    resources = cs.get_resources()
    Cnt = resources.get_mmr_constants()

    sct_compile = chck_sct_h(Cnt)
    def_compile = chck_vox_h(Cnt)
    # sct_compile = False
    # def_compile = False

    if sct_compile or def_compile:
        txt = "NiftyPET constants were changed: needs CUDA compilation."
    else:
        txt = "- - . - -"

    log.info(
        dedent("""\
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            changed sct.h: {}
            changed def.h: {}
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            {}
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~""").format(
            sct_compile, def_compile, txt))


cs.resources_setup(gpu=False) # install resources.py

# check and update the constants in C headers according to resources.py
check_constants()
try:
    nvcc_arches = cs.dev_setup() # update resources.py with a supported GPU device
except Exception as exc:
    nvcc_arches = []
    log.error("could not set up CUDA:\n%s", exc)

build_ver = ".".join(__version__.split('.')[:3]).split(".dev")[0]
cmake_args = [f"-DNIPET_BUILD_VERSION={build_ver}", f"-DPython3_ROOT_DIR={sys.prefix}"]
try:
    if nvcc_arches:
        cmake_args.append("-DCMAKE_CUDA_ARCHITECTURES=" + ";".join(sorted(nvcc_arches)))
except Exception as exc:
    if "sdist" not in sys.argv or any(i in sys.argv for i in ["build", "bdist", "wheel"]):
        log.warning("Import or CUDA device detection error:\n%s", exc)
for i in (Path(__file__).resolve().parent / "_skbuild").rglob("CMakeCache.txt"):
    i.write_text(re.sub("^//.*$\n^[^#].*pip-build-env.*$", "", i.read_text(), flags=re.M))
setup(use_scm_version=True, packages=find_packages(exclude=["examples", "tests"]),
      package_data={"niftypet": ["nipet/auxdata/*"]}, cmake_source_dir="niftypet",
      cmake_languages=("C", "CXX", "CUDA"), cmake_minimum_required_version="3.18",
      cmake_args=cmake_args)
