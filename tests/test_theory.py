import pytest
import os
import copy
import numpy as np

CWD = os.getcwd()
cobaya_par_file = CWD + '/tests/inputs/cobaya_pars/ps_base.yaml'
cosmo_par_file_sim = CWD + '/tests/inputs/cosmo_pars/planck2018_fnl_1p0.yaml'
cosmo_par_file_ref = CWD + '/tests/inputs/cosmo_pars/planck2018_fiducial.yaml'

@pytest.mark.long
@pytest.mark.parametrize("cobaya_par_file, cosmo_par_file, expected", \
    [
    (cobaya_par_file, cosmo_par_file_sim, 0), \
    (cobaya_par_file, cosmo_par_file_ref, 0.44438055)
    ]
)
def test_cobaya(cobaya_par_file, cosmo_par_file, expected):

    from cobaya.model import get_model
    from cobaya.yaml import yaml_load_file
    from spherelikes.theory.PowerSpectrum3D import make_dictionary_for_bias_params
    from spherelikes.params import CobayaPar, SurveyPar

    cobaya_par = CobayaPar(cobaya_par_file)
    survey_par = cobaya_par.get_survey_par()
    bias_params = make_dictionary_for_bias_params(survey_par, \
        fix_to_default=True, include_latex=False)

    camb_params = yaml_load_file(cosmo_par_file)
    camb_params = convert_camb_params(camb_params)
    params = dict(camb_params, **bias_params) 

    info = yaml_load_file(cobaya_par_file)
    info['params'] = params
    info['debug'] = True

    model = get_model(info)
    chi2 = -2.0 * model.loglikes({'my_foreground_amp': 1.0})[0] 

    assert np.isclose(chi2[0], expected)

def convert_camb_params(camb_params):
    camb_params['As'] = 1e-10*np.exp(camb_params['logA'])
    del camb_params['logA']
    camb_params['cosmomc_theta'] = camb_params['theta_MC_100']/100
    del camb_params['theta_MC_100']
    return camb_params