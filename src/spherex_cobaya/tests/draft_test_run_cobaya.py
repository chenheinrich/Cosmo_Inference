import pytest
from cobaya.yaml import yaml_load_file
from cobaya.run import run

from spherelikes.theory.PowerSpectrum3D import PowerSpectrum3D


def test_cobaya_run():
    info = yaml_load_file("./inputs/sample.yaml")
    info["force"] = True
    info["debug"] = True
    updated_info, sampler = run(info)
