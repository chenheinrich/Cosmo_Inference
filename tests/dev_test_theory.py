import os
import numpy as np
import copy 
import pickle 

from cobaya.model import get_model
from cobaya.yaml import yaml_load_file
from spherelikes.theory.PowerSpectrum3D import make_dictionary_for_bias_params
from spherelikes.params import CobayaPar, SurveyPar
#TODO need importing from the right place if SurveyPar is refactored

def test_cobaya(cobaya_par_file, cosmo_par_file, chi2_expected, fn_expected):

    cobaya_par = CobayaPar(cobaya_par_file)
    survey_par = cobaya_par.get_survey_par()
    bias_params = make_dictionary_for_bias_params(survey_par, \
        fix_to_default=True, include_latex=False)

    camb_params = yaml_load_file(cosmo_par_file)
    camb_params = convert_camb_params(camb_params)
    params = dict(camb_params, **bias_params) 
    print('param = ', params)

    info = yaml_load_file(cobaya_par_file)
    info['params'] = params
    info['debug'] = True

    model = get_model(info)
    ps = model.provider.get_galaxy_ps()
    ps_expected = np.load(fn_expected)
    #print('frac diff ps:', (ps-ps_expected)/ps_expected)
    print('ps agrees? ', np.allclose(ps, ps_expected))
    assert np.allclose(ps, ps_expected)

    chi2 = -2.0 * model.loglikes({'my_foreground_amp': 1.0})[0]
    print('chi2 = {}'.format(chi2))
    print('chi2 agrees?', np.isclose(chi2[0], chi2_expected))
    assert np.isclose(chi2[0], chi2_expected), (chi2[0], chi2_expected)
    
    #TODO turn this into a test
    # save new sim_data and invcov
    # decide which format, .pickle or .npy
    # clean up prep_chain etc
    # fix mistakes in invcov calculations for PS
    # set chains going for bispectrum.

def convert_camb_params(camb_params):
    camb_params['As'] = 1e-10*np.exp(camb_params['logA'])
    del camb_params['logA']
    camb_params['cosmomc_theta'] = camb_params['theta_MC_100']/100
    del camb_params['theta_MC_100']
    return camb_params

CWD = os.getcwd()
cobaya_par_file = CWD + '/tests/inputs/cobaya_pars/ps_base.yaml'
#cobaya_par_file = CWD + '/tests/inputs/cobaya_pars/ps_base_old.yaml'
cosmo_par_file_sim = CWD + '/tests/inputs/cosmo_pars/planck2018_fnl_1p0.yaml'
cosmo_par_file_ref = CWD + '/tests/inputs/cosmo_pars/planck2018_fiducial.yaml'

fn_sim = './plots/theory/PowerSpectrum3D/nk_21_nmu_5_v28/fnl_1/ps.npy'
fn_ref = './plots/theory/PowerSpectrum3D/nk_21_nmu_5_v28/fnl_0/ps.npy'

test_cobaya(cobaya_par_file, cosmo_par_file_sim, 0.0, fn_sim)
test_cobaya(cobaya_par_file, cosmo_par_file_ref, 0.44438055, fn_ref)