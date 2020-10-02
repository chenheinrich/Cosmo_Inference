from spherelikes.model import ModelCalculator
import os


def main():
    """Computes and saves reference results used for AP effects. This could be 
    different than the simulated data vector cosmology, but must be the same as 
    the covariance matrix.

    Note: we set is_reference_model = True automatically in this script to disable
    loading reference results since we are calculating them.

    Note: You can also disable the likelihood calculation to not load elements 
    yet to be calculated (e.g. inverse covariance and simulated data vectors) 
    by setting is_reference_likelihood = True.
    """

    CWD = os.getcwd()
    args = {
        'model_name': 'ref',
        'model_yaml_file': CWD + '/inputs/cosmo_pars/planck2018_fiducial.yaml',
        'cobaya_yaml_file': CWD + '/inputs/cobaya_pars/ps_base.yaml',
        'output_dir': CWD + '/data/ps_base/',
    }

    args['is_reference_model'] = True
    args['is_reference_likelihood'] = True

    calc = ModelCalculator(args)
    results = calc.get_and_save_results()


if __name__ == '__main__':
    main()
