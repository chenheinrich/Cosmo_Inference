from spherelikes.model import ModelCalculator
import os
import copy


def generate_ref(args_in):
    """Computes and saves reference results used for AP effects. This could be 
    different than the simulated data vector cosmology, but must be the same as 
    the covariance matrix.

    Note: we set is_reference_model = True automatically in this script to disable
    loading reference results since we are calculating them.

    Note: You can also disable the likelihood calculation to not load elements 
    yet to be calculated (e.g. inverse covariance and simulated data vectors) 
    by setting is_reference_likelihood = True.
    """

    args = copy.deepcopy(args_in)

    if args['model_name'] is None:
        args['model_name'] = 'ref'

    args['is_reference_model'] = True
    args['is_reference_likelihood'] = True

    calc = ModelCalculator(args)
    results = calc.get_and_save_results()

    return results


if __name__ == '__main__':
    CWD = os.getcwd()
    args = {
        'model_name': 'ref',
        'model_yaml_file': CWD + '/inputs/cosmo_pars/planck2018_fiducial.yaml',
        'cobaya_yaml_file': CWD + '/inputs/cobaya_pars/ps_base_minimal.yaml',
        'input_survey_pars': CWD + '/inputs/survey_pars/survey_pars_v28_base_cbe.yaml',
        'output_dir': CWD + '/data/ps_base_minimal/',
        'theory_name': "theories.base_classes.ps_base.ps_base.PowerSpectrumSingleTracer"
    }
    generate_ref(args)
