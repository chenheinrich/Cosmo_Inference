import pytest
import os
import copy
import numpy as np

CWD = os.getcwd()
info = {}
info['run_name'] = 'test_b3D'
info['cosmo_par_file'] =  CWD + '/inputs/cosmo_pars/planck2018_fiducial.yaml'
info['cosmo_par_fid_file'] = CWD + '/inputs/cosmo_pars/planck2018_fiducial.yaml'
info['survey_par_file'] = CWD + '/inputs/survey_pars/survey_pars_v27_base_cbe_Mar21.yaml'
info['Bispectrum3D'] = {
  'nk': 21, # number of k points (to be changed into bins)
  'nmu': 5, # number of mu bins
  'kmin': 0.0007, # equivalent to 0.001 h/Mpc
  'kmax': 0.1, # equivalent to 0.2 h/Mpc
}
info['plot_dir'] = './plots/theory/bispectrum/'

from theory.get_bis import get_data_vec_bis, get_data_spec
data_vec = get_data_vec_bis(info)
data_spec = get_data_spec(info)

@pytest.mark.parametrize("data_vec, data_spec, isample, iz, imu", \
    [
    (data_vec, data_spec, 0, 0, 0), \
    (data_vec, data_spec, 0, 5, 0), \
    ]
)
def test_Bggg_b10_equilateral_triangles_single_tracer(data_vec, data_spec, isample, iz, imu):
    expected = data_vec.get_expected_Bggg_b10_equilateral_triangles_single_tracer(\
        isample=isample, iz=iz, imu=imu)
    answer = get_Bggg_b10_equilateral_triangles_single_tracer(\
        data_vec, data_spec, isample, iz)
    assert np.allclose(answer, expected)

def get_Bggg_b10_equilateral_triangles_single_tracer(data_vec, data_spec, isample, iz):
    
    ibis = data_spec.dict_isamples_to_ib['%i_%i_%i'%(isample, isample, isample)]
    indices_equilateral = data_spec.triangle_spec.indices_equilateral
    
    Bggg_b10_all = data_vec.get('Bggg_b10')
    ans = Bggg_b10_all[ibis, iz, indices_equilateral]
    
    return ans

@pytest.mark.parametrize("data_vec, data_spec, isample1, isample2, isample3, iz, imu, itri", \
    [
    (data_vec, data_spec, 0, 1, 2, 0, 0, 0), \
    (data_vec, data_spec, 0, 4, 2, 5, 0, 11), \
    (data_vec, data_spec, 0, 4, 2, 5, 0, None), \
    ]
)
def test_Bggg_b10_general_triangles_multi_tracer(data_vec, data_spec, isample1, isample2, isample3, iz, itri, imu):
    """itri = None tests all triangles at the same time."""
    expected = data_vec.get_expected_Bggg_b10_general_triangles_multi_tracer(\
        isample1, isample2, isample3, iz, itri=itri, imu=imu)
    
    answer = get_Bggg_general_triangles_multi_tracer(data_vec, data_spec, \
        isample1, isample2, isample3, iz, itri=itri, imu=imu, name='Bggg_b10')
    
    assert np.allclose(answer, expected)

@pytest.mark.parametrize("data_vec, data_spec, isample1, isample2, isample3, iz, imu, itri", \
    [
    (data_vec, data_spec, 0, 4, 2, 5, 0, None), \
    ]
)

def test_Bggg_b20_general_triangles_multi_tracer(data_vec, data_spec, isample1, isample2, isample3, iz, itri, imu):
    """itri = None tests all triangles at the same time."""
    expected = data_vec.get_expected_Bggg_b20_general_triangles_multi_tracer(\
        isample1, isample2, isample3, iz, itri=itri, imu=imu)
    
    answer = get_Bggg_general_triangles_multi_tracer(data_vec, data_spec, \
        isample1, isample2, isample3, iz, itri=itri, imu=imu, name='Bggg_b20')
    
    assert np.allclose(answer, expected)

@pytest.mark.parametrize("data_vec, data_spec, isample1, isample2, isample3, iz, imu, itri, expected", \
    [
    (data_vec, data_spec, 0, 4, 2, 5, 0, 3, 772698575.5929023), \
    ]
)
def test_Bggg_general_triangles_multi_tracer(data_vec, data_spec, \
    isample1, isample2, isample3, iz, itri, imu, expected):
    """itri = None tests all triangles at the same time."""
    
    answer = get_Bggg_general_triangles_multi_tracer(data_vec, data_spec, \
        isample1, isample2, isample3, iz, itri=itri, imu=imu, name='galaxy_bis')
    
    assert np.allclose(answer, expected), (answer, expected)

def get_Bggg_general_triangles_multi_tracer(data_vec, data_spec,\
    isample1, isample2, isample3, iz, itri=None, imu=0, name='galaxy_bis'):
    """name can be 'Bggg',  'Bggg_b10', or 'Bggg_b20'"""
    
    ibis = data_spec.dict_isamples_to_ib['%i_%i_%i'%(isample1, isample2, isample3)]
    Bggg_all = data_vec.get(name)
    if itri is None:
        itri = np.arange(Bggg_all.shape[2]) 
    ans = Bggg_all[ibis, iz, itri]
    
    return ans