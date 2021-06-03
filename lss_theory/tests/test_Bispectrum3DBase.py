import pytest
import os
import copy
import numpy as np
import yaml 


@pytest.mark.debug
@pytest.mark.parametrize("yaml_file, cosmo_par_file, fn_expected", \
    [
        ("./galaxy_3d/tests/data/get_bis_base.yaml", './inputs/cosmo_pars/planck2018_fiducial.yaml', './galaxy_3d/tests/data/bis_base_fnl_0.npy'), \
        ("./galaxy_3d/tests/data/get_bis_base.yaml", './inputs/cosmo_pars/planck2018_fnl_1p0.yaml', './galaxy_3d/tests/data/bis_base_fnl_1.npy'), \
    ]
)
def test_Bispectrum3DBase_mu_set_to_zero(yaml_file, cosmo_par_file, fn_expected):
#TODO probably want to change to testing versions without setting mu to zero (nmu =-1)
    from lss_theory.scripts.get_bis_base import get_b3d_base

    with open(yaml_file, 'r') as f:
        info = yaml.safe_load(f)
        
    info['cosmo_par_file'] = cosmo_par_file
    data_vec_b3d_base = get_b3d_base(info)

    exp = np.load(fn_expected)
    galaxy_bis = data_vec_b3d_base.get('galaxy_bis')

    assert np.allclose(exp, galaxy_bis)


CWD = os.getcwd()
info = {}
info['run_name'] = 'test_Bispectrum3DBase'
info['cosmo_par_file'] =  CWD + '/inputs/cosmo_pars/planck2018_fiducial.yaml'
info['cosmo_par_fid_file'] = CWD + '/inputs/cosmo_pars/planck2018_fiducial.yaml'
info['survey_par_file'] = CWD + '/inputs/survey_pars/survey_pars_v27_base_cbe_Mar21.yaml'
info['Bispectrum3DBase'] = {
  'nk': 21, # number of k points (to be changed into bins)
  'nmu': 5, # number of mu bins
  'kmin': 0.0007, # equivalent to 0.001 h/Mpc
  'kmax': 0.1, # equivalent to 0.2 h/Mpc
}
info['plot_dir'] = './plots/theory/bispectrum/'

from lss_theory.scripts.get_bis_base import get_b3d_base, get_data_spec
data_vec = get_b3d_base(info)
data_spec = get_data_spec(info)

@pytest.mark.parametrize("data_vec, data_spec, isample, iz, imu", \
    [
    (data_vec, data_spec, 0, 0, 0), \
    (data_vec, data_spec, 0, 5, 0), \
    ]
)
def test_Bggg_b10_equilateral_triangles_single_tracer(data_vec, data_spec, isample, iz, imu):
    expected = get_expected_Bggg_b10_equilateral_triangles_single_tracer(data_vec,
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
    expected = get_expected_Bggg_b10_general(data_vec, \
        isample1, isample2, isample3, iz, itri=itri, imu=imu)
    
    answer = get_Bggg_general(data_vec, data_spec, \
        isample1, isample2, isample3, iz, itri=itri, imu=imu, name='Bggg_b10')
    
    assert np.allclose(answer, expected)

def get_expected_Bggg_b10_general(self, \
    isample1, isample2, isample3, iz, itri=None, imu=0): 

    if itri is None:
        ik1 = self._ik1
        ik2 = self._ik2
        ik3 = self._ik3
    else:
        iks = self._triangle_spec.tri_index_array[itri]
        ik1 = iks[0]
        ik2 = iks[1]
        ik3 = iks[2]

    bias = self._grs_ingredients.get('galaxy_bias_without_AP') 
    b_g1 = bias[isample1, iz, ik1] 
    b_g2 = bias[isample2, iz, ik2] 
    b_g3 = bias[isample3, iz, ik3]

    matter_power = self._grs_ingredients.get('matter_power_without_AP')
    Pm = matter_power[iz, :]
    pk12 = Pm[ik1] * Pm[ik2] 
    pk23 = Pm[ik2] * Pm[ik3]
    pk13 = Pm[ik1] * Pm[ik3]

    alpha = self._grs_ingredients.get('alpha_without_AP') # shape = (nz, nk, nmu)
    alpha1 = alpha[iz, ik1]
    alpha2 = alpha[iz, ik2]
    alpha3 = alpha[iz, ik3]

    k1_array = self._data_spec.k[ik1]
    k2_array = self._data_spec.k[ik2]
    k3_array = self._data_spec.k[ik3]

    fnl = self._grs_ingredients.get('fnl')
    t1 = 2.0 * fnl * alpha3 / (alpha1 * alpha2) + \
            2.0 * self._get_F2(k1_array, k2_array, k3_array)
    t2 = 2.0 * fnl * alpha2 / (alpha1 * alpha3) + \
            2.0 * self._get_F2(k1_array, k3_array, k2_array)
    t3 = 2.0 * fnl * alpha1 / (alpha2 * alpha3) + \
            2.0 * self._get_F2(k2_array, k3_array, k1_array)
    Bmmm = t1 * pk12 + t2 * pk13 + t3 * pk23

    Bggg_b10 = Bmmm * b_g1 * b_g2 * b_g3

    return Bggg_b10




@pytest.mark.parametrize("data_vec, data_spec, isample1, isample2, isample3, iz, imu, itri", \
    [
        (data_vec, data_spec, 0, 4, 2, 5, 0, None), \
    ]
)
def test_Bggg_b20_general_triangles_multi_tracer(data_vec, data_spec, isample1, isample2, isample3, iz, itri, imu):
    """itri = None tests all triangles at the same time."""
    expected = get_expected_Bggg_b20_general(data_vec, \
        isample1, isample2, isample3, iz, itri=itri, imu=imu)
    
    answer = get_Bggg_general(data_vec, data_spec, \
        isample1, isample2, isample3, iz, itri=itri, imu=imu, name='Bggg_b20')
    
    assert np.allclose(answer, expected)


def get_expected_Bggg_b10_equilateral_triangles_single_tracer(self, isample=0, iz=0, imu=0):
    """Returns a 1D numpy array for expected value of Bggg b10 terms 
    for equilateral triangles in single tracer specified by isample."""

    matter_power = self._grs_ingredients.get('matter_power_without_AP')
    Pm = matter_power[iz, :]
    
    bias = self._grs_ingredients.get('galaxy_bias_without_AP') 
    b = bias[isample, iz, :]

    alpha = self._grs_ingredients.get('alpha_without_AP') 
    alpha1 = alpha[iz, np.arange(self._data_spec.nk)]

    fnl = self._grs_ingredients.get('fnl')
    
    F2_equilateral = 0.2857142857142857
    Bmmm_equilateral = 3.0 * (2.0 * F2_equilateral * Pm ** 2)
    Bmmm_equilateral += 3.0 * (2.0 * fnl / alpha1 * Pm ** 2)
    
    Bggg_b10_equilateral_triangles_single_tracer = b ** 3 * Bmmm_equilateral 

    return Bggg_b10_equilateral_triangles_single_tracer


def get_expected_Bggg_b20_general(self, \
    isample1, isample2, isample3, iz, itri=None, imu=0): 

    if itri is None:
        ik1 = self._ik1
        ik2 = self._ik2
        ik3 = self._ik3
    else:
        iks = self._triangle_spec.tri_index_array[itri]
        ik1 = iks[0]
        ik2 = iks[1]
        ik3 = iks[2]

    bias = self._grs_ingredients.get('galaxy_bias_without_AP') 
    bias_20 = self._grs_ingredients.get('galaxy_bias_20') 

    matter_power = self._grs_ingredients.get('matter_power_without_AP')
    Pm = matter_power[iz, :]
    pk12 = Pm[ik1] * Pm[ik2] 
    pk23 = Pm[ik2] * Pm[ik3]
    pk13 = Pm[ik1] * Pm[ik3]
    
    Bggg_b20 = bias[isample1, iz, ik1] \
                    * bias[isample2, iz, ik2] \
                    * bias_20[isample3, iz] \
                    * pk12 \
            + bias[isample1, iz, ik1] \
                    * bias_20[isample2, iz] \
                    * bias[isample3, iz, ik3] \
                    * pk13 \
            + bias_20[isample1, iz] \
                    * bias[isample2, iz, ik2] \
                    * bias[isample3, iz, ik3] \
                    * pk23 

    return Bggg_b20




@pytest.mark.parametrize("data_vec, data_spec, isample1, isample2, isample3, iz, imu, itri, expected", \
    [
    (data_vec, data_spec, 0, 4, 2, 5, 0, 3, 772698575.5929023), \
    ]
)
def test_Bggg_general(data_vec, data_spec, \
    isample1, isample2, isample3, iz, itri, imu, expected):
    """itri = None tests all triangles at the same time."""
    
    answer = get_Bggg_general(data_vec, data_spec, \
        isample1, isample2, isample3, iz, itri=itri, imu=imu, name='galaxy_bis')
    
    assert np.allclose(answer, expected), (answer, expected)

def get_Bggg_general(data_vec, data_spec,\
    isample1, isample2, isample3, iz, itri=None, imu=0, name='galaxy_bis'):
    """name can be 'Bggg',  'Bggg_b10', or 'Bggg_b20'"""
    
    ibis = data_spec.dict_isamples_to_ib['%i_%i_%i'%(isample1, isample2, isample3)]
    Bggg_all = data_vec.get(name)
    if itri is None:
        itri = np.arange(Bggg_all.shape[2]) 
    ans = Bggg_all[ibis, iz, itri]
    
    return ans

