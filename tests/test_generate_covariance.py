import pytest
import os
import copy

from spherelikes.model import ModelCalculator

from scripts.generate_covariance import generate_covariance, CovCalculator


@pytest.fixture
def input_args():
    CWD = os.getcwd()
    args = {
        'model_name': None,
        'cosmo_par_file': CWD + '/tests/inputs/cosmo_pars/planck2018_fiducial.yaml',
        'cobaya_par_file': CWD + '/tests/inputs/cobaya_pars/ps_base.yaml',
        'survey_par_file': CWD + '/tests/inputs/survey_pars/survey_pars_v28_base_cbe.yaml',
        'output_dir': CWD + '/tests/data/ps_base/',
        'theory_name': "theories.base_classes.ps_base.ps_base.PowerSpectrumBase",
        'overwrite_covariance': False,
        'fix_default_bias': True,
    }
    return args


@pytest.fixture
def cov_calc(input_args):

    args = copy.deepcopy(input_args)

    if args['model_name'] is None:
        args['model_name'] = 'testing_covariance'

    args['is_reference_model'] = True
    args['is_reference_likelihood'] = True

    model_calc = ModelCalculator(args)
    results = model_calc.get_results()
    cov_calc = CovCalculator(results, args)

    return cov_calc


@pytest.mark.short
def test_dictionaries(cov_calc):
    cov_calc.test_dictionaries_are_constructed_correctly()


@pytest.mark.short
def test_cov_is_symmetric_exchanging_ips1_and_ips2(cov_calc):
    cov_calc.get_cov()
    cov_calc.test_cov_is_symmetric_exchanging_ips1_and_ips2()


@pytest.mark.short
def test_get_noise(cov_calc):
    cov_calc.test_get_noise()


def test_inversion(cov_calc):
    cov_calc.get_and_save_invcov()
