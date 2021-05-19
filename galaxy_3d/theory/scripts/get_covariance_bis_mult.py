import numpy as np
import argparse
import yaml
import os

from theory.covariance import BispectrumMultipoleCovariance
from theory.utils import file_tools

def check_matrix_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

if __name__ == '__main__':
    """
    Example usage:
        python3 -m galaxy_3d.theory.scripts.get_covariance_bis_mult ./galaxy_3d/inputs_theory/get_covariance_bis_mult.yaml
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

    cov_calculator = BispectrumMultipoleCovariance(info)
    
    cov_type = 'full'

    fn_cov = os.path.join(info['data_dir'], 'cov_%s.npy'%cov_type)
    fn_invcov = os.path.join(info['data_dir'], 'invcov_%s.npy'%cov_type)

    cov_calculator.get_and_save_cov(fn_cov, cov_type=cov_type, do_invcov=True)
    