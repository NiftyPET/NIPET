===========================================================
NIPET: high-throughput Neuro-Image PET reconstruction
===========================================================

|Docs| |PyPI-Status| |PyPI-Downloads|

NIPET is a Python sub-package of NiftyPET_, offering high-throughput PET image reconstruction as well as image processing and analysis (``nimpa``: https://github.com/NiftyPET/NIMPA) for PET/MR imaging with high quantitative accuracy and precision. The software is written in CUDA C and embedded in Python C extensions.

.. _NiftyPET: https://github.com/NiftyPET/NiftyPET

The scientific aspects of this software are covered in two open-access publications:

* *NiftyPET: a High-throughput Software Platform for High Quantitative Accuracy and Precision PET Imaging and Analysis* Neuroinformatics (2018) 16:95. https://doi.org/10.1007/s12021-017-9352-y

* *Rapid processing of PET list-mode data for efficient uncertainty estimation and data analysis* Physics in Medicine & Biology (2016). https://doi.org/10.1088/0031-9155/61/13/N322

Although, the two stand-alone and independent packages, ``nipet`` and ``nimpa``, are dedicated to brain imaging, they can equally well be used for whole body imaging.  Strong emphasis is put on the data, which are acquired using positron emission tomography (PET) and magnetic resonance (MR), especially the hybrid and simultaneous PET/MR scanners.

This software platform and Python name-space *NiftyPET* covers the entire processing pipeline, from the raw list-mode (LM) PET data through to the final image statistic of interest (e.g., regional SUV), including LM bootstrapping and multiple reconstructions to facilitate voxel-wise estimation of uncertainties.

In order to facilitate all the functionality, *NiftyPET* relies on third-party software for image conversion from DICOM to NIfTI (dcm2niix) and image registration (NiftyReg).  The additional software is installed automatically to a user specified location.

**Documentation with installation manual and tutorials**: https://niftypet.readthedocs.io/

Quick Install
~~~~~~~~~~~~~

Note that installation prompts for setting the path to `NiftyPET_tools` and
hardware attenuation maps. This can be avoided by setting the environment
variables `PATHTOOLS` and `HMUDIR`, respectively.

.. code:: sh

    # optional (Linux syntax) to avoid prompts
    export PATHTOOLS=$HOME/NiftyPET_tools
    export HMUDIR=$HOME/mmr_hardwareumaps
    # cross-platform install
    conda create -n niftypet -c conda-forge python=2.7 \
      ipykernel matplotlib numpy scikit-image ipywidgets
    git clone https://github.com/NiftyPET/NIMPA.git nimpa
    git clone https://github.com/NiftyPET/NIPET.git nipet
    conda activate niftypet
    pip install --no-binary :all: --verbose -e ./nimpa
    pip install --no-binary :all: --verbose -e ./nipet

Licence
~~~~~~~

|Licence|

- Author: `Pawel J. Markiewicz <https://github.com/pjmark>`__ @ University College London
- `Contributors <https://github.com/NiftyPET/NIPET/graphs/contributors>`__:

  - `Casper O. da Costa-Luis <https://github.com/casperdcl>`__ @ King's College London

Copyright 2018-19

.. |Docs| image:: https://readthedocs.org/projects/niftypet/badge/?version=latest
   :target: https://niftypet.readthedocs.io/en/latest/?badge=latest
.. |Licence| image:: https://img.shields.io/pypi/l/nipet.svg?label=licence
   :target: https://github.com/NiftyPET/NIPET/blob/master/LICENCE
.. |PyPI-Downloads| image:: https://img.shields.io/pypi/dm/nipet.svg?label=PyPI%20downloads
   :target: https://pypi.org/project/nipet
.. |PyPI-Status| image:: https://img.shields.io/pypi/v/nipet.svg?label=latest
   :target: https://pypi.org/project/nipet
