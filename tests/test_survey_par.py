import pytest
import numpy as np

from spherelikes.params import SurveyPar

@pytest.fixture
def survey_par():
    survey_par_file = './tests/inputs/survey_pars/survey_pars_v28_base_cbe.yaml'
    return SurveyPar(survey_par_file)

@pytest.fixture
def nz():
    return 11

@pytest.fixture
def nsample():
    return 5

@pytest.fixture
def expected_shape(nsample, nz):
    return (nsample, nz)

@pytest.mark.short
def test_survey_par_get_sigz_array_over_one_plus_z(survey_par):
    data = survey_par.get_sigz_array_over_one_plus_z()
    expected = np.array([0.003, 0.01, 0.03, 0.1, 0.2])
    assert np.allclose(data, expected)

@pytest.mark.short
def test_survey_par_get_galaxy_bias_array(survey_par, expected_shape):
    data = survey_par.get_galaxy_bias_array()
    msg = 'galaxy bias shape = {}, expected shape = {}'.format(data.shape, expected_shape)
    assert data.shape == expected_shape, (msg)
    
    expected = np.array([1.3, 1.5, 1.8, 2.3, 2.1, 2.7, 3.6, 2.3, 3.2, 2.7, 3.8])
    msg = 'galaxy bias for first sample loaded = {}, expected {}'.format(data[0,:], expected)
    assert np.allclose(data[0,:], expected), (msg)

def test_survey_par_get_number_density_array(survey_par, expected_shape):
    data = survey_par.get_number_density_array()
    msg = 'number_density shape = {}, expected shape = {}'.format(data.shape, expected_shape)
    assert data.shape == expected_shape, (msg)
    
    expected = np.array([0.00997, 0.00411, 0.000501])
    msg = 'number density for first sample loaded = {}, expected {}'.format(data[0,:], expected)
    assert np.allclose(data[0,:3], expected), (msg)
    
@pytest.mark.short
def test_survey_par_get_nz(survey_par, nz):
    data = survey_par.get_nz()
    assert data == nz

@pytest.mark.short
def test_survey_par_get_nsample(survey_par, nsample):
    data = survey_par.get_nsample()
    assert data == nsample

@pytest.mark.short
def test_survey_par_get_zlo_array(survey_par, nz):
    data = survey_par.get_zlo_array()
    expected = np.array([0.0, 0.2, 0.4])
    assert data.size == nz
    assert np.allclose(data[:3], expected)

@pytest.mark.short
def test_survey_par_get_zhi_array(survey_par, nz):
    data = survey_par.get_zhi_array()
    expected = np.array([0.2, 0.4, 0.6])
    assert data.size == nz
    assert np.allclose(data[:3], expected)

@pytest.mark.short
def test_survey_par_get_zmid_array(survey_par, nz):
    data = survey_par.get_zmid_array()
    expected = 0.5 * (np.array([0.2, 0.4, 0.6]) + np.array([0.0, 0.2, 0.4]))
    assert data.size == nz
    assert np.allclose(data[:3], expected)

@pytest.mark.short
def test_survey_par_get_sigz_array(survey_par, expected_shape):
    data = survey_par.get_sigz_array()
    zmid_first_bin = 0.5 * (0.0+0.2)
    expected = np.array([0.003, 0.01, 0.03, 0.1, 0.2]) * (1.0+zmid_first_bin)
    assert data.shape == expected_shape
    msg = 'data first column is {}, expected = {}'.format(data[:,0], expected)
    assert np.allclose(data[:,0], expected), msg