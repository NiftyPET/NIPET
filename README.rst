===========================================================
NIPET: high-throughput Neuro-Image PET reconstruction
===========================================================

|Docs| |Version| |Downloads| |Py-Versions| |DOI| |Licence| |Tests|

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

Note that installation prompts for setting the path to ``NiftyPET_tools`` and
hardware attenuation maps. This can be avoided by setting the environment
variables ``PATHTOOLS`` and ``HMUDIR``, respectively.
It's also recommended (but not required) to use `conda`.

.. code:: sh

    # optional (Linux syntax) to avoid prompts
    export PATHTOOLS=$HOME/NiftyPET_tools
    export HMUDIR=$HOME/mmr_hardwareumaps
    # cross-platform install
    conda install -c conda-forge python=3 \
      ipykernel numpy scipy scikit-image matplotlib ipywidgets dipy nibabel pydicom
    pip install "nipet>=2"

External CMake Projects
~~~~~~~~~~~~~~~~~~~~~~~

The raw C/CUDA libraries may be included in external projects using ``cmake``.
Simply build the project and use ``find_package(NiftyPETnipet)``.

.. code:: sh

    # print installation directory (after `pip install nipet`)...
    python -c "from niftypet.nipet import cmake_prefix; print(cmake_prefix)"

    # ... or build & install directly with cmake
    mkdir build && cd build
    cmake ../niftypet && cmake --build . && cmake --install . --prefix /my/install/dir

At this point any external project may include NIPET as follows
(Once setting ``-DCMAKE_PREFIX_DIR=<installation prefix from above>``):

.. code:: cmake

    cmake_minimum_required(VERSION 3.3 FATAL_ERROR)
    project(myproj)
    find_package(NiftyPETnipet COMPONENTS mmr_auxe mmr_lmproc petprj nifty_scatter REQUIRED)
    add_executable(myexe ...)
    target_link_libraries(myexe PRIVATE
      NiftyPET::mmr_auxe NiftyPET::mmr_lmproc NiftyPET::petprj NiftyPET::nifty_scatter)

Licence
~~~~~~~

|Licence| |DOI|

Copyright 2018-21

- `Pawel J. Markiewicz <https://github.com/pjmark>`__ @ University College London
- `Casper O. da Costa-Luis <https://github.com/casperdcl>`__ @ King's College London
- `Contributors <https://github.com/NiftyPET/NIPET/graphs/contributors>`__

.. |Docs| image:: https://readthedocs.org/projects/niftypet/badge/?version=latest
   :target: https://niftypet.readthedocs.io/en/latest/?badge=latest
.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4417679.svg
   :target: https://doi.org/10.5281/zenodo.4417679
.. |Licence| image:: https://img.shields.io/pypi/l/nipet.svg?label=licence
   :target: https://github.com/NiftyPET/NIPET/blob/master/LICENCE
.. |Tests| image:: https://img.shields.io/github/workflow/status/NiftyPET/NIPET/Test?logo=GitHub
   :target: https://github.com/NiftyPET/NIPET/actions
.. |Downloads| image:: https://img.shields.io/pypi/dm/nipet.svg?logo=pypi&logoColor=white&label=PyPI%20downloads
   :target: https://pypi.org/project/nipet
.. |Version| image:: https://img.shields.io/pypi/v/nipet.svg?logo=python&logoColor=white
   :target: https://github.com/NiftyPET/NIPET/releases
.. |Py-Versions| image:: https://img.shields.io/pypi/pyversions/nipet.svg?logo=python&logoColor=white
   :target: https://pypi.org/project/nipet
