import pytest
import os
import copy
import numpy as np
import yaml 

test_dir = os.path.dirname(os.path.realpath(__file__))
input_dir = os.path.join(test_dir, "inputs/")
data_dir = os.path.join(test_dir, "data/")

@pytest.mark.debug
@pytest.mark.parametrize("yaml_file, cosmo_par_file, fn_expected", \
    [
        (
            os.path.join(input_dir, "get_ps.yaml"), \
            os.path.join(input_dir, "cosmo_pars/planck2018_fiducial.yaml"), \
            os.path.join(data_dir, "ps_fnl_0.npy") \
        ), \
        (
            os.path.join(input_dir, "get_ps.yaml"), \
            os.path.join(input_dir, "cosmo_pars/planck2018_fnl_1p0.yaml"), \
            os.path.join(data_dir, "ps_fnl_1.npy") \
        ), \
    ]
)
def test_PowerSpectrum3D(yaml_file, cosmo_par_file, fn_expected):
    
    from lss_theory.scripts.get_ps import get_data_vec_p3d, get_fn

    with open(yaml_file, 'r') as f:
        info = yaml.safe_load(f)
        
    info['cosmo_par_file'] = cosmo_par_file
    info['cosmo_par_fid_file'] = os.path.join(test_dir, info['cosmo_par_fid_file'])
    info['survey_par_file'] = os.path.join(test_dir, info['survey_par_file'])
    info['plot_dir'] = os.path.join(test_dir, info['plot_dir'])
    
    data_vec = get_data_vec_p3d(info)

    exp = np.load(fn_expected)
    galaxy_ps = data_vec.get('galaxy_ps')

    fn = get_fn(info)
    np.save(fn, galaxy_ps)

    #TODO need to investigate why not better
    assert np.allclose(exp, galaxy_ps, atol=1e-3, rtol=1e-3)

