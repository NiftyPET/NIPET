import errno
import logging
from filelock import FileLock
from os import path
from textwrap import dedent

import numpy as np
import pytest

from niftypet import nimpa
from niftypet import nipet

# segmentation/parcellation for PVC, with unique regions numbered from 0 onwards
pvcroi = []
pvcroi.append([66, 67] + list(range(81, 95)))  # white matter
pvcroi.append([36])  # brain stem
pvcroi.append([35])  # pons
pvcroi.append([39, 40, 72, 73, 74])  # cerebellum GM
pvcroi.append([41, 42])  # cerebellum WM
pvcroi.append([48, 49])  # hippocampus
pvcroi.append([167, 168])  # posterior cingulate gyrus
pvcroi.append([139, 140])  # middle cingulate gyrus
pvcroi.append([101, 102])  # anterior cingulate gyrus
pvcroi.append([169, 170])  # precuneus
pvcroi.append([32, 33])  # amygdala
pvcroi.append([37, 38])  # caudate
pvcroi.append([56, 57])  # pallidum
pvcroi.append([58, 59])  # putamen
pvcroi.append([60, 61])  # thalamus
pvcroi.append([175, 176, 199, 200])  # parietal without precuneus
pvcroi.append([133, 134, 155, 156, 201, 202, 203, 204])  # temporal
pvcroi.append([4, 5, 12, 16, 43, 44, 47, 50, 51, 52, 53])  # CSF
pvcroi.append([24, 31, 62, 63, 70, 76, 77, 96, 97])  # basal ganglia + optic chiasm
pvcroi.append(
    list(range(103, 110 + 1))
    + list(range(113, 126 + 1))
    + list(range(129, 130 + 1))
    + list(range(135, 138 + 1))
    + list(range(141, 154 + 1))
    + list(range(157, 158 + 1))
    + list(range(161, 166 + 1))
    + list(range(171, 174 + 1))
    + list(range(177, 188 + 1))
    + list(range(191, 198 + 1))
    + list(range(205, 208 + 1))
)  # remaining neocortex
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


@pytest.fixture(scope="session")
def mMRpars():
    params = nipet.get_mmrparams()
    params["Cnt"]["VERBOSE"] = True
    params["Cnt"]["LOG"] = logging.INFO
    return params


@pytest.fixture(scope="session")
def datain(mMRpars, folder_in):
    return nipet.classify_input(folder_in, mMRpars)


@pytest.fixture(scope="session")
def muhdct(mMRpars, datain, tmp_path_factory, worker_id):
    tmp_path = tmp_path_factory.getbasetemp().parent
    opth = str(tmp_path / "muhdct")

    if worker_id == "master":  # not xdist, auto-reuse
        return nipet.hdw_mumap(
            datain, [1, 2, 4], mMRpars, outpath=opth, use_stored=True
        )

    flock = FileLock(opth + ".lock")
    with flock.acquire(poll_intervall=0.5):  # xdist, force auto-reuse via flock
        return nipet.hdw_mumap(
            datain, [1, 2, 4], mMRpars, outpath=opth, use_stored=True
        )


@pytest.fixture(scope="session")
def refimg(folder_ref):
    # predetermined structure of the reference folder
    basic = path.join(folder_ref, "basic")
    spm = path.join(folder_ref, "dyn_aligned", "spm")
    niftyreg = path.join(folder_ref, "dyn_aligned", "niftyreg")
    refpaths = {
        "histo": {"p": 1570707830, "d": 817785422},
        "basic": {
            "pet": path.join(basic, "17598013_t-3000-3600sec_itr-4_suvr.nii.gz"),
            "omu": path.join(basic, "mumap-from-DICOM_no-alignment.nii.gz"),
            "hmu": path.join(basic, "hardware_umap.nii.gz"),
        },
        "aligned": {
            "spm": {
                "hmu": path.join(spm, "hardware_umap.nii.gz"),
                "omu": path.join(spm, "mumap-PCT-aligned-to_t0-3600_AC.nii.gz"),
                "pos": path.join(spm, "17598013_t0-3600sec_itr2_AC-UTE.nii.gz"),
                "pet": path.join(spm, "17598013_nfrm-2_itr-4.nii.gz"),
                "trm": path.join(
                    spm, "17598013_nfrm-2_itr-4_trimmed-upsampled-scale-2.nii.gz"
                ),
                "pvc": path.join(
                    spm, "17598013_nfrm-2_itr-4_trimmed-upsampled-scale-2_PVC.nii.gz"
                ),
            },
            "niftyreg": {
                "hmu": path.join(niftyreg, "hardware_umap.nii.gz"),
                "omu": path.join(niftyreg, "mumap-PCT-aligned-to_t0-3600_AC.nii.gz"),
                "pos": path.join(niftyreg, "17598013_t0-3600sec_itr2_AC-UTE.nii.gz"),
                "pet": path.join(niftyreg, "17598013_nfrm-2_itr-4.nii.gz"),
                "trm": path.join(
                    niftyreg, "17598013_nfrm-2_itr-4_trimmed-upsampled-scale-2.nii.gz"
                ),
                "pvc": path.join(
                    niftyreg,
                    "17598013_nfrm-2_itr-4_trimmed-upsampled-scale-2_PVC.nii.gz",
                ),
            },
        },
    }

    testext = {
        "basic": {
            "pet": "static reconstruction with unaligned UTE mu-map",
            "hmu": "hardware mu-map for the static unaligned reconstruction",
            "omu": "object mu-map for the static unaligned reconstruction",
        },
        "aligned": {
            "hmu": "hardware mu-map for the 2-frame aligned reconstruction",
            "omu": "object mu-map for the 2-frame aligned reconstruction",
            "pos": "AC reconstruction for positioning (full acquisition used)",
            "pet": "2-frame scan with aligned UTE mu-map",
            "trm": "trimming post reconstruction",
            "pvc": "PVC post reconstruction",
        },
    }

    # check basic files
    frefs = refpaths["basic"]
    for k in frefs:
        if not path.isfile(frefs[k]):
            raise FileNotFoundError(errno.ENOENT, frefs[k])

    # check reg tools: niftyreg and spm
    frefs = refpaths["aligned"]
    for r in frefs:
        for k in frefs[r]:
            if not path.isfile(frefs[r][k]):
                raise FileNotFoundError(errno.ENOENT, frefs[r][k])

    return refpaths, testext


def test_histogramming(mMRpars, datain, refimg, tmp_path):
    refpaths, _ = refimg
    opth = str(tmp_path / "histogramming")
    hst = nipet.mmrhist(datain, mMRpars, outpath=opth, store=True)

    # prompt counts: head curve & sinogram
    assert np.sum(hst["phc"]) == np.sum(hst["psino"])
    # delayed counts: head curve & sinogram
    assert np.sum(hst["dhc"]) == np.sum(hst["dsino"])
    # prompt counts: head curve & reference
    assert np.sum(hst["phc"]) == refpaths["histo"]["p"]
    # delayed counts: head curve & reference
    assert np.sum(hst["dhc"]) == refpaths["histo"]["d"]


def test_basic_reconstruction(mMRpars, datain, muhdct, refimg, tmp_path):
    refpaths, testext = refimg
    opth = str(tmp_path / "basic_reconstruction")

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
def test_aligned_reconstruction(reg_tool, mMRpars, datain, muhdct, refimg, tmp_path):
    refpaths, testext = refimg
    opth = str(tmp_path / "aligned_reconstruction")

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
