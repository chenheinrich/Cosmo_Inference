import os
import copy

from spherelikes.model import ModelCalculator


def generate_data(args_in):
    """Computes and saves simulated data vector, which could have a different 
    cosmology than the fiducial cosmology used in the covariance matrix or the
    reference cosmology for AP effects. 

    Note: We set is_reference_model = False automatically in this script since
    we are interested in getting galaxy_ps data vector with AP effects in it.

    Note: We also disable the likelihood calculation to not load elements 
    yet to be calculated (e.g. inverse covariance and simulated data vectors) 
    by setting is_reference_likelihood = True here.
    """

    args = copy.deepcopy(args_in)

    if args['model_name'] is None:
        args['model_name'] = 'sim_data'

    args['is_reference_model'] = False
    args['is_reference_likelihood'] = True

    calc = ModelCalculator(args)
    results = calc.get_and_save_results()
    # TODO add nuisance model in the future, going through the likelihood

    return results

