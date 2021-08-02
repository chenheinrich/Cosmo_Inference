import pytest
import os
import copy
import numpy as np
import yaml 

@pytest.mark.debug
@pytest.mark.parametrize("yaml_file, cosmo_par_file, fn_expected", \
    [
        ("./galaxy_3d/tests/data/get_bis_rsd.yaml", './inputs/cosmo_pars/planck2018_fiducial.yaml', './galaxy_3d/tests/data/bis_rsd_fnl_0.npy'), \
        ("./galaxy_3d/tests/data/get_bis_rsd.yaml", './inputs/cosmo_pars/planck2018_fnl_1p0.yaml', './galaxy_3d/tests/data/bis_rsd_fnl_1.npy'), \
    ]
)
def test_Bispectrum3DRSD(yaml_file, cosmo_par_file, fn_expected):

    from lss_theory.scripts.get_bis_rsd import get_b3d_rsd

    with open(yaml_file, 'r') as f:
        info_rsd = yaml.safe_load(f)
        
    info_rsd['cosmo_par_file'] = cosmo_par_file
    data_vec_b3d_rsd = get_b3d_rsd(info_rsd)

    exp = np.load(fn_expected)
    galaxy_bis = data_vec_b3d_rsd.get('galaxy_bis')

    assert np.allclose(exp, galaxy_bis)

