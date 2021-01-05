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

@pytest.mark.long
@pytest.mark.parametrize("data_vec, data_spec, isample, iz, imu", \
    [
    (data_vec, data_spec, 0, 0, 0), \
    (data_vec, data_spec, 0, 5, 0), \
    ]
)

def test_B3D_Bggg_b10_equilateral_triangles_single_tracer(data_vec, data_spec, isample, iz, imu):
    expected = data_vec.get_expected_Bggg_b10_equilateral_triangles_single_tracer(\
        isample=isample, iz=iz, imu=imu)
    answer = get_Bggg_b10_equilateral_triangles_single_tracer(\
        data_vec, data_spec, isample, iz)
    assert np.allclose(answer, expected)

def get_Bggg_b10_equilateral_triangles_single_tracer(data_vec, data_spec, isample, iz):
    
    ibis = data_spec.dict_isamples_to_ib['%i_%i_%i'%(isample, isample, isample)]
    indices_equilateral = data_spec.triangle_specs.indices_equilateral
    
    Bggg_b10_all = data_vec.get('Bggg_b10')
    ans = Bggg_b10_all[ibis, iz, indices_equilateral]
    
    return ans
