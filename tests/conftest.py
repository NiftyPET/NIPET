from os import getenv, path

import pytest

HOME = getenv("DATA_ROOT", path.expanduser("~"))


@pytest.fixture(scope="session")
def folder_in():
    Ab_PET_mMR_test = path.join(HOME, "Ab_PET_mMR_test")
    if not path.isdir(Ab_PET_mMR_test):
        pytest.skip(f"""Cannot find Ab_PET_mMR_test in ${{DATA_ROOT:-~}} ({HOME}).
Get it from https://zenodo.org/record/3877529
""")
    return Ab_PET_mMR_test


@pytest.fixture(scope="session")
def folder_ref(folder_in):
    Ab_PET_mMR_ref = path.join(folder_in, "testing_reference", "Ab_PET_mMR_ref")
    if not path.isdir(Ab_PET_mMR_ref):
        pytest.skip(f"""Cannot find Ab_PET_mMR_ref in
${{DATA_ROOT:-~}}/testing_reference ({HOME}/testing_reference).
Get it from https://zenodo.org/record/3877529
""")
    return Ab_PET_mMR_ref
