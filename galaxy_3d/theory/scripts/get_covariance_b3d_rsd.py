import numpy as np
import argparse
import yaml
import os

from theory.covariance import Bispectrum3DRSDCovarianceCalculator
from theory.utils import file_tools


if __name__ == '__main__':
    """
    Example usage:
        python3 -m theory.scripts.get_covariance_b3d_rsd ./inputs_theory/get_covariance_b3d_rsd.yaml
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file", type=str, default=None,
        help="path to config file."
    )

    command_line_args = parser.parse_args()

    with open(command_line_args.config_file) as file:
        info = yaml.load(file, Loader=yaml.FullLoader)
    print('info = {}'.format(info))

    cov_calculator = Bispectrum3DRSDCovarianceCalculator(info)
    
    fn_cov = os.path.join(info['plot_dir'], 'cov.npy')
    fn_invcov = os.path.join(info['plot_dir'], 'invcov.npy')
    
    cov_calculator.get_and_save_cov(fn_cov)
    cov_calculator.get_and_save_invcov(fn_invcov)

    #TODO add test to make inverse went well
    
