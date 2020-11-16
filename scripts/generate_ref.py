from spherelikes.model import ModelCalculator
import os
import copy


def generate_ref(args_in):
    """Computes and saves reference results used for AP effects. This could be 
    different than the simulated data vector cosmology, but must be the same as 
    the covariance matrix.

    Note: we set is_reference_model = True automatically in this script to disable
    loading reference results since we are calculating them.

    Note: We also disable the likelihood calculation to not load elements 
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
