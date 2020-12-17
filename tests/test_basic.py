from os import path

from niftypet import nipet


def test_dev_info(capsys):
    devs = nipet.dev_info()
    out, err = capsys.readouterr()
    assert not any((out, err))
    nipet.dev_info(showprop=True)
    out, err = capsys.readouterr()
    assert not err
    assert not devs or out


def test_resources():
    assert path.exists(nipet.path_resources)
    assert nipet.resources.DIRTOOLS
