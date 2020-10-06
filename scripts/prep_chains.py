import sys
import os
import copy
import argparse

from cobaya.run import run
from cobaya.yaml import yaml_load_file

from scripts.generate_ref import generate_ref
from scripts.generate_covariance import generate_covariance
from scripts.generate_data import generate_data


class SimulatedChains():

    def __init__(self, args):

        self.command_line_args = args
        self.args = yaml_load_file(self.command_line_args.config_file)

        self.info = yaml_load_file(self.args['cobaya_yaml_file'])
        theory_name = self.args['theory_name']
        self.args['input_survey_pars'] = self.info['theory'][theory_name]['survey_pars_file_name']

    def prepare_chains(self):
        self.get_reference_model()
        self.get_covariance()
        self.get_simulated_data()

    def get_reference_model(self):
        """Generate reference cosmology results for AP"""
        args_in = copy.deepcopy(self.args)
        args_in['model_yaml_file'] = self.args['ref_model_yaml_file']
        generate_ref(args_in)
        print('Got reference model successfully!')

    def get_covariance(self):
        """generate inverse covariance if it doesn't already exist"""
        invcov_path = os.path.join(self.args['output_dir'], 'invcov.npy')
        if not os.path.exists(invcov_path):
            #  pulls same survey par file as used during sampling to add shot noise to covariance
            args_in = copy.deepcopy(self.args)
            args_in['model_yaml_file'] = self.args['ref_model_yaml_file']
            generate_covariance(args_in)
            # TODO additional checks that some criterias are satisfied?
        else:
            print('Skip making inverse covariance. Found invcov file at\n    {}'.format(
                invcov_path))
        print('Got inverse covariance successfully!')

    def get_simulated_data(self):
        """Generate simulated data"""
        args_in = copy.deepcopy(self.args)
        args_in['model_yaml_file'] = self.args['data_model_yaml_file']
        generate_data(args_in)
        print('Got simulated data successfully!')

    def run_chains(self):
        for k, v in {"-f": "force", "-r": "resume", "-d": "debug"}.items():
            if k in sys.argv:
                self.info[v] = True
        print('Start sampling ...')
        updated_info, sampler = run(self.info)


if __name__ == '__main__':
    """
    Example usage:
        python3 scripts/prep_chains.py ./inputs/simulated_chains_pars/sim.yaml 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file", type=str, default=None,
        help="path to config file."
    )

    command_line_args = parser.parse_args()
    sim = SimulatedChains(command_line_args)
    sim.prepare_chains()