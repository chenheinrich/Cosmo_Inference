import pytest
import os
import copy
import numpy as np
from spherelikes.utils.log import file_logger

from cobaya.yaml import yaml_load_file

logger = file_logger(__file__)

camb_params = {
    "ombh2": 0.022274,
    "omch2": 0.11913,
    "cosmomc_theta": 0.01040867,
    "As": 0.2132755716e-8,
    "ns": 0.96597,
    "tau": 0.0639, 
    "fnl": 0.0,
    "nrun": 0.0,
    "omegak": 0.0, 
    "mnu": 0.06
    }

#TODO this whole needs more polishing

@pytest.fixture
def input_args():
    CWD = os.getcwd()
    args = {
        'model_name': None,
        #'cosmo_par_file': CWD + '/tests/inputs/cosmo_pars/planck2018_fiducial.yaml',
        'cosmo_par_file': CWD + '/tests/inputs/cosmo_pars/planck2018_fnl_1p0.yaml',
        'cobaya_par_file': CWD + '/tests/inputs/cobaya_pars/unit_test_ps_base.yaml',
        'survey_par_file': CWD + '/tests/inputs/survey_pars/survey_pars_v28_base_cbe.yaml',
        'output_dir': CWD + '/tests/data/ps_base/',
        'theory_name': "theories.base_classes.ps_base.ps_base.PowerSpectrumBase",
        'fix_default_bias': True,
    }
    return args

# TODO To be written
def test_indep(input_args):
    from plancklensing import PlanckLensingMarged
    import camb
    lmax = 2500
    opts = camb_params.copy()
    opts['lens_potential_accuracy'] = 1
    opts['lmax'] = lmax
    pars = camb.set_params(**opts)
    results = camb.get_results(pars)
    cls = results.get_total_cls(lmax, CMB_unit='muK')
    cl_dict = {p: cls[:, i] for i, p in enumerate(['tt', 'ee', 'bb', 'te'])}
    cl_dict['pp'] = results.get_lens_potential_cls(lmax)[:, 0]
    like = PlanckLensingMarged()
    self.assertAlmostEqual(-2 * like.log_likelihood(cl_dict), 8.76, 1)

    from plancklensing import PlanckLensing
    like = PlanckLensing()
    self.assertAlmostEqual(-2 * like.log_likelihood(cl_dict, A_planck=1.0), 8.734, 1)

    # aggressive likelihood
    like = PlanckLensingMarged(
        {'dataset_file': 'data_2018/smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_agr2_CMBmarged.dataset'})
    self.assertAlmostEqual(-2 * like.log_likelihood(cl_dict), 13.5, 1)


@pytest.mark.likelihood
def test_cobaya(input_args):

    from cobaya.model import get_model
    from spherelikes.params import get_bias_params_for_survey_file

    info = yaml_load_file(input_args['cobaya_par_file'])
    cosmo_params = yaml_load_file(input_args['cosmo_par_file'])

    bias_params = get_bias_params_for_survey_file(
        input_args['survey_par_file'], 
        fix_to_default=True, include_latex=False,
        prior_name=None)

    info['params'] = dict(camb_params, **bias_params) #cosmo_params
    info['debug'] = True

    model = get_model(info)
    chi2 = -2.0 * model.loglikes({'my_foreground_amp': 1.0})[0] 
    assert np.isclose(chi2[0], 19.307424996210607)
