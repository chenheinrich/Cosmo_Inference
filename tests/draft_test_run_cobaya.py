import pytest
from cobaya.yaml import yaml_load_file
from cobaya.run import run

from spherelikes.theory.PowerSpectrum3D import PowerSpectrum3D


def test_cobaya_run():
    info = yaml_load_file("./inputs/sample.yaml")
    info["force"] = True
    info["debug"] = True
    updated_info, sampler = run(info)


def test_ps_base_theory():  # TODO to be written
    ps_base = PowerSpectrumBase()
    ps_base.nk == 21
    ps_base.nmu == 5
    ps_base.get_requirements == {}
    var = ps_base._get_var_fid('Hubble')
    assert var.size == ps_base.nz
    var = ps_base._get_var_fid('angular_diameter_distance')
    assert var.size == ps_base.nz
