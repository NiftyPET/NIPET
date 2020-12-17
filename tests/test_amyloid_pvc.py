import logging
import sys
from collections import Iterable
from os import path
from textwrap import dedent

import numpy as np
import pytest

from niftypet import nimpa
from niftypet import nipet

mMRpars = nipet.get_mmrparams()
mMRpars["Cnt"]["VERBOSE"] = True
mMRpars["Cnt"]["LOG"] = logging.INFO

pvcroi = []  # from amyroi_def import pvcroi
# expected %error for static (SUVr) and PVC reconstructions
emape_basic = 0.1
emape_algnd = {
    "pet": 3.0,
    "pos": 0.1,
    "trm": 3.0,
    "pvc": 3.0,
    "hmu": 0.01,
    "omu": 3.0,
}


def test_histogramming(folder_in, folder_ref):
    datain = nipet.classify_input(folder_in, mMRpars)
    refpaths, testext = nipet.resources.get_refimg(folder_ref)
    hst = nipet.mmrhist(datain, mMRpars)

    # prompt counts: head curve & sinogram
    assert np.sum(hst["phc"]) == np.sum(hst["psino"])
    # delayed counts: head curve & sinogram
    assert np.sum(hst["dhc"]) == np.sum(hst["dsino"])
    # prompt counts: head curve & reference
    assert np.sum(hst["phc"]) == refpaths["histo"]["p"]
    # delayed counts: head curve & reference
    assert np.sum(hst["dhc"]) == refpaths["histo"]["d"]


def test_basic_reconstruction(folder_in, folder_ref, tmp_path):
    datain = nipet.classify_input(folder_in, mMRpars)
    refpaths, testext = nipet.resources.get_refimg(folder_ref)
    opth = str(tmp_path / "basic_reconstruction")

    muhdct = nipet.hdw_mumap(datain, [1, 2, 4], mMRpars, outpath=opth, use_stored=True)
    muodct = nipet.obj_mumap(datain, mMRpars, outpath=opth, store=True)
    recon = nipet.mmrchain(
        datain,
        mMRpars,
        frames=["timings", [3000, 3600]],
        mu_h=muhdct,
        mu_o=muodct,
        itr=4,
        fwhm=0.0,
        outpath=opth,
        fcomment="_suvr",
        store_img=True,
        store_img_intrmd=True,
    )

    testout = {"pet": recon["fpet"], "hmu": muhdct["im"], "omu": muodct["im"]}
    for k in testext["basic"]:
        diff = nimpa.imdiff(refpaths["basic"][k], testout[k], verbose=True, plot=False)
        assert diff["mape"] <= emape_basic, testext["basic"][k]


@pytest.mark.parametrize("reg_tool", ["niftyreg", "spm"])
def test_aligned_reconstruction(reg_tool, folder_in, folder_ref, tmp_path):
    datain = nipet.classify_input(folder_in, mMRpars)
    refpaths, testext = nipet.resources.get_refimg(folder_ref)
    opth = str(tmp_path / "aligned_reconstruction")

    muhdct = nipet.hdw_mumap(datain, [1, 2, 4], mMRpars, outpath=opth, use_stored=True)
    muopct = nipet.align_mumap(
        datain,
        mMRpars,
        outpath=opth,
        t0=0,
        t1=3600,
        reg_tool=reg_tool,
        store=True,
        itr=2,
        petopt="ac",
        fcomment="",
        musrc="pct",
    )
    # matshow(muopct['im'][60,:,:] + muhdct['im'][60,:,:])
    recon = nipet.mmrchain(
        datain,
        mMRpars,
        frames=["timings", [80, 260], [3000, 3600]],
        mu_h=muhdct,
        mu_o=muopct,
        itr=4,
        fwhm=0.0,
        outpath=opth,
        trim=True,
        pvcroi=pvcroi,
        pvcreg_tool=reg_tool,
        store_img=True,
        store_img_intrmd=True,
    )

    testout = {
        "pet": recon["fpet"],
        "hmu": muhdct["im"],
        "omu": muopct["im"],
        "pos": muopct["fpet"],
        "trm": recon["trimmed"]["fpet"],
        "pvc": recon["trimmed"]["fpvc"],
    }
    for k in testext["aligned"]:
        diff = nimpa.imdiff(
            refpaths["aligned"][reg_tool][k], testout[k], verbose=True, plot=False
        )
        assert diff["mape"] <= emape_algnd[k], testext["aligned"][k]
