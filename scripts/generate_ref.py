from spherelikes.model import ModelCalculator
import os


def main():
    """Computes and saves reference results used for AP effects. This could be 
    different than the simulated data vector cosmology, but usually the same as 
    the covariance matrix, unless some explicit testing is being done. 

    Note: we set is_reference_model = True automatically in this script to disable
    loading reference results since we are calculating them."""

    CWD = os.getcwd()
    args = {
        'model_name': 'ref',
        'model_yaml_file': CWD + '/inputs/cosmo_pars/planck2018_fiducial.yaml',
        'cobaya_yaml_file': CWD + '/inputs/cobaya_pars/ps_base.yaml',
        'output_dir': CWD + '/data/ps_base/',
        'is_reference_model': True
    }

    calc = ModelCalculator(args)
    results = calc.get_and_save_results()


if __name__ == '__main__':
    main()
