import pytest
import os
import numpy as np
import yaml 

test_dir = os.path.dirname(os.path.realpath(__file__))
input_dir = os.path.join(test_dir, "inputs/")
data_dir = os.path.join(test_dir, "data/")

@pytest.mark.debug
@pytest.mark.parametrize("yaml_file, cosmo_par_file, fn_expected", \
    [
        (
            os.path.join(input_dir, "get_b3d_rsd.yaml"), \
            os.path.join(input_dir, "cosmo_pars/planck2018_fiducial.yaml"), \
            os.path.join(data_dir, "b3d_rsd_fnl_0.npy") \
        ), \
        (
            os.path.join(input_dir, "get_b3d_rsd.yaml"), \
            os.path.join(input_dir, "cosmo_pars/planck2018_fnl_1p0.yaml"), \
            os.path.join(data_dir, "b3d_rsd_fnl_1.npy") \
        ), \
    ]
)
def test_Bispectrum3DRSD(yaml_file, cosmo_par_file, fn_expected):

    from lss_theory.scripts.get_b3d_rsd import get_b3d_rsd, get_fn

    with open(yaml_file, 'r') as f:
        info = yaml.safe_load(f)

    info['cosmo_par_file'] = cosmo_par_file
    info['cosmo_par_fid_file'] = os.path.join(test_dir, info['cosmo_par_fid_file'])
    info['survey_par_file'] = os.path.join(test_dir, info['survey_par_file'])
    
    info['output_dir'] = os.path.join(test_dir, info['output_dir'])
    info['plot_dir'] = os.path.join(test_dir, info['plot_dir'])
    
    data_vec_b3d_rsd = get_b3d_rsd(info)

    exp = np.load(fn_expected)
    galaxy_bis = data_vec_b3d_rsd.get('galaxy_bis')

    fn = get_fn(info)
    np.save(fn, galaxy_bis)

    #TODO test not passing anymore, need to figure out why.
    assert np.allclose(exp, galaxy_bis)

