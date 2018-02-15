===========================================================
NiftyPET: High-throughput image reconstruction and analysis
===========================================================

*NiftyPET* is a Python software platform, offering high-throughput PET image reconstruction ( Python package ``nipet`` -- a core package of NiftyPET) as well as image processing and analysis (Python package ``nimpa``: https://github.com/pjmark/NIMPA) for PET/MR imaging with high quantitative accuracy and precision. The software is written in CUDA C and embedded in Python C extensions.

The scientific aspects of this software are covered in two open-access publications:

* *NiftyPET: a High-throughput Software Platform for High Quantitative Accuracy and Precision PET Imaging and Analysis* Neuroinformatics (2018) 16:95. https://doi.org/10.1007/s12021-017-9352-y

* *Rapid processing of PET list-mode data for efficient uncertainty estimation and data analysis* Physics in Medicine & Biology (2016). https://doi.org/10.1088/0031-9155/61/13/N322

*NiftyPET* includes two stand-alone and independent Python packages: NIPET and NIMPA, which are dedicated to high-throughput image reconstruction and analysis of brain images, respectively.  Strong emphasis is put on the data, which are acquired using positron emission tomography (PET) and magnetic resonance (MR), especially the hybrid and simultaneous PET/MR scanners.  

This software platform covers the entire processing pipeline, from the raw list-mode (LM) PET data through to the final image statistic of interest (e.g., regional SUV), including LM bootstrapping and multiple reconstructions to facilitate voxel-wise estimation of uncertainties.

In order to facilitate all the functionality, *NiftyPET* relies on third-party software for image conversion from DICOM to NIfTI (dcm2niix) and image registration (NiftyReg).  The additional software is installed automatically to a user specified location.


Dependencies
------------

NIMPA relies on GPU computing using NVidia's CUDA platform.  The CUDA routines are wrapped in Python C extensions.  The provided software has to be compiled from source (done automatically) for any given Linux flavour (Linux is preferred over Windows) using Cmake.

The following software has to be installed prior to NIMPA installation:

* CUDA (currently the latest is 9.1): https://developer.nvidia.com/cuda-downloads

* Cmake (version 3.xx): https://cmake.org/download/

* Python with the recommended Anaconda distribution: https://www.anaconda.com/download


Installation
------------

To install NIMPA from source for any given CUDA version and operating system (Linux is preferred), simply type:

.. code-block:: bash

  pip install --no-binary :all: --verbose nipet 
  

Usage
-----

.. code-block:: python

  from niftypet import nipet
  from niftypet import nimpa



Author: Pawel J. Markiewicz

Copyright 2018