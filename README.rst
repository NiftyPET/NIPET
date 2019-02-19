===========================================================
NIPET: high-throughput Neuro-Image PET reconstruction
===========================================================

.. image:: https://readthedocs.org/projects/niftypet/badge/?version=latest
  :target: https://niftypet.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status


NIPET is a Python sub-package of NiftyPET_, offering high-throughput PET image reconstruction as well as image processing and analysis (``nimpa``: https://github.com/pjmark/NIMPA) for PET/MR imaging with high quantitative accuracy and precision. The software is written in CUDA C and embedded in Python C extensions.

.. _NiftyPET: https://github.com/pjmark/NiftyPET

The scientific aspects of this software are covered in two open-access publications:

* *NiftyPET: a High-throughput Software Platform for High Quantitative Accuracy and Precision PET Imaging and Analysis* Neuroinformatics (2018) 16:95. https://doi.org/10.1007/s12021-017-9352-y

* *Rapid processing of PET list-mode data for efficient uncertainty estimation and data analysis* Physics in Medicine & Biology (2016). https://doi.org/10.1088/0031-9155/61/13/N322

Although, the two stand-alone and independent packages, ``nipet`` and ``nimpa``, are dedicated to brain imaging, they can equally well be used for whole body imaging.  Strong emphasis is put on the data, which are acquired using positron emission tomography (PET) and magnetic resonance (MR), especially the hybrid and simultaneous PET/MR scanners.

This software platform and Python name-space *NiftyPET* covers the entire processing pipeline, from the raw list-mode (LM) PET data through to the final image statistic of interest (e.g., regional SUV), including LM bootstrapping and multiple reconstructions to facilitate voxel-wise estimation of uncertainties.

In order to facilitate all the functionality, *NiftyPET* relies on third-party software for image conversion from DICOM to NIfTI (dcm2niix) and image registration (NiftyReg).  The additional software is installed automatically to a user specified location.

**Documentation with installation manual and tutorials**: https://niftypet.readthedocs.io/

Quick install
~~~~~~~~~~~~~

.. code:: sh

    conda create -n niftypet python=2.7 \
      conda-forge::nibabel conda-forge::pydicom ipykernel matplotlib \
      conda-forge::tqdm conda-forge::ipywidgets
    git clone https://github.com/pjmark/NIMPA.git nimpa
    git clone https://github.com/pjmark/NIPET.git nipet
    cd nimpa
    pip install --no-binary :all: --verbose .
    cd ../nipet
    pip install --no-binary :all: --verbose .

Author: Pawel J. Markiewicz @ University College London

Copyright 2018
