import pytest
import numpy as np

from spherex_cobaya.params import CobayaPar

@pytest.fixture
def cobaya_par():
    cobaya_par_file = './tests/inputs/cobaya_pars/ps_base.yaml'
    return CobayaPar(cobaya_par_file)

@pytest.mark.short
def test_get_spherex_theory_list(cobaya_par):
    spherex_theories = cobaya_par.get_spherex_theory_list()
    expected = ['spherex_cobaya.theory.PowerSpectrum3D']
    assert spherex_theories == expected
