from spherelikes.model import ModelCalculator
import os


def main():
    """Computes and saves simulated data vector, which could have a different 
    cosmology than the fiducial cosmology used in the covariance matrix or the
    reference cosmology for AP effects. 

    Note: we set is_reference_model = False automatically in this script since
    we are interested in getting galaxy_ps data vector with AP effects in it.

    Note: You can also disable the likelihood calculation to not load elements 
    yet to be calculated (e.g. inverse covariance and simulated data vectors) 
    by setting is_reference_likelihood = True.
    """

    CWD = os.getcwd()
    args = {
        'model_name': 'sim_data',
        'model_yaml_file': CWD + '/inputs/cosmo_pars/planck2018_fiducial.yaml',
        'cobaya_yaml_file': CWD + '/inputs/cobaya_pars/ps_base.yaml',
        'output_dir': CWD + '/data/ps_base/',
    }

    args['is_reference_model'] = False
    args['is_reference_likelihood'] = True

    calc = ModelCalculator(args)
    results = calc.get_and_save_results()
    # TODO add nuisance model in the future, going through the likelihood


if __name__ == '__main__':
    main()
