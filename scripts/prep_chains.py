import sys
import os
import copy
import argparse

from cobaya.run import run
from cobaya.yaml import yaml_load_file

from scripts.generate_ref import generate_ref
from scripts.generate_covariance import generate_covariance
from scripts.generate_data import generate_data

from spherelikes.params import CobayaPar

class ChainPreparation():

    def __init__(self, config_file):

        self._common_args = self._get_common_args(config_file)
        
    def _get_common_args(self, config_file):

        common_args = yaml_load_file(config_file)
        cobaya_par = CobayaPar(common_args['cobaya_par_file'])
        survey_par = cobaya_par.get_survey_par()
        common_args['survey_par_file'] = survey_par.filename
        common_args['fix_default_bias'] = True

        if 'overwrite_covariance' not in common_args.keys():
            common_args['overwrite_covariance'] = False

        return common_args

    def prepare_chains(self):
        self._get_reference_model()
        self._get_covariance()
        self._get_simulated_data()

    def _get_reference_model(self):
        """Generate reference cosmology results for AP"""
        args_in = self._get_args_for_model_type('ref')
        generate_ref(args_in)
        print('Got reference model successfully!')

    def _get_covariance(self):
        """generate inverse covariance if it doesn't already exist"""
        invcov_path = os.path.join(self._common_args['output_dir'], 'invcov.npy')

        skip = (os.path.exists(invcov_path) and self._common_args['overwrite_covariance'] is not True)
        if skip is True:
            print("Skip making inverse covariance.")
            print("Found invcov file at: {}".format(invcov_path))
        else:
            args_in = self._get_args_for_model_type('ref')
            generate_covariance(args_in)
        print('Got inverse covariance successfully!')

    def _get_simulated_data(self):
        """Generate simulated data"""
        args_in = self._get_args_for_model_type('data')
        generate_data(args_in)
        print('Got simulated data successfully!')

    def _get_args_for_model_type(self, model_type):
        args_in = copy.deepcopy(self._common_args)
        if model_type == 'ref':
            args_in['cosmo_par_file'] = self._common_args['ref_cosmo_par_file']
        elif model_type == 'data':
            args_in['cosmo_par_file'] = self._common_args['data_cosmo_par_file']
        return args_in

if __name__ == '__main__':
    """
    Example usage:
        python3 scripts/prep_chains.py ./inputs/chains_pars/ps_base.yaml 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file", type=str, default=None,
        help="path to config file."
    )

    command_line_args = parser.parse_args()
    prep = ChainPreparation(command_line_args.config_file)
    prep.prepare_chains()
