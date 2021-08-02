import pytest
import os
import copy
import numpy as np
import yaml 


@pytest.mark.debug
@pytest.mark.parametrize("yaml_file, cosmo_par_file, fn_expected", \
    [
        ("./lss_theory/tests/data/get_ps.yaml", './inputs/cosmo_pars/planck2018_fiducial.yaml', './lss_theory/tests/data/ps_fnl_0.npy'), \
        ("./lss_theory/tests/data/get_ps.yaml", './inputs/cosmo_pars/planck2018_fnl_1p0.yaml', './lss_theory/tests/data/ps_fnl_1.npy'), \
    ]
)
def test_PowerSpectrum3D(yaml_file, cosmo_par_file, fn_expected):
    
    from lss_theory.scripts.get_ps import get_data_vec_p3d

    with open(yaml_file, 'r') as f:
        info = yaml.safe_load(f)
        
    info['cosmo_par_file'] = cosmo_par_file
    data_vec = get_data_vec_p3d(info)

    exp = np.load(fn_expected)
    galaxy_ps = data_vec.get('galaxy_ps')

    assert np.allclose(exp, galaxy_ps)

