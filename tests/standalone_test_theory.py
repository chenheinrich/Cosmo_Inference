import pytest
import os
import copy
import numpy as np

def input_args():
    CWD = os.getcwd()
    args = {
        #'cosmo_par_file': CWD + '/tests/inputs/cosmo_pars/planck2018_fnl_1p0.yaml',
        'cosmo_par_file': CWD + '/tests/inputs/cosmo_pars/planck2018_fiducial.yaml',
        'cobaya_par_file': CWD + '/tests/inputs/cobaya_pars/ps_base.yaml',
        'output_dir': CWD + '/tests/data/ps_base/',
    }
    return args

def test_cobaya(input_args):

    from cobaya.model import get_model
    from cobaya.yaml import yaml_load_file
    from spherelikes.theory.PowerSpectrum3D import make_dictionary_for_bias_params
    from spherelikes.params import CobayaPar

    cobaya_par = CobayaPar(input_args['cobaya_par_file'])
    survey_par_file = cobaya_par.get_survey_par_file()
    bias_params = make_dictionary_for_bias_params(survey_par_file, \
        fix_to_default=True, include_latex=False)

    camb_params = yaml_load_file(input_args['cosmo_par_file'])
    camb_params = convert_camb_params(camb_params)
    params = dict(camb_params, **bias_params) 

    info = yaml_load_file(input_args['cobaya_par_file'])
    info['params'] = params
    info['debug'] = True

    model = get_model(info)
    chi2 = -2.0 * model.loglikes({'my_foreground_amp': 1.0})[0] 
    print('chi2 = {}'.format(chi2))

    assert np.isclose(chi2[0], 0)

def convert_camb_params(camb_params):
    camb_params['As'] = 1e-10*np.exp(camb_params['logA'])
    del camb_params['logA']
    camb_params['cosmomc_theta'] = camb_params['theta_MC_100']/100
    del camb_params['theta_MC_100']
    return camb_params

if __name__ == '__main__':
    input_args = input_args()
    test_cobaya(input_args)