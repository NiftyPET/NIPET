from os import getenv
from pathlib import Path

from pytest import fixture, skip

HOME = Path(getenv("DATA_ROOT", "~")).expanduser()


@fixture(scope="session")
def folder_in():
    Ab_PET_mMR_test = HOME / "Ab_PET_mMR_test"
    if not Ab_PET_mMR_test.is_dir():
        skip(f"""Cannot find Ab_PET_mMR_test in ${{DATA_ROOT:-~}} ({HOME}).
Try running `python -m tests` to download it.
""")
    return Ab_PET_mMR_test


@fixture(scope="session")
def folder_ref(folder_in):
    Ab_PET_mMR_ref = folder_in / "testing_reference" / "Ab_PET_mMR_ref"
    if not Ab_PET_mMR_ref.is_dir():
        skip(f"""Cannot find Ab_PET_mMR_ref in
${{DATA_ROOT:-~}}/testing_reference ({HOME}/testing_reference).
Try running `python -m tests` to download it.
""")
    return Ab_PET_mMR_ref
