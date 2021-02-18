import numpy as np
import argparse
import yaml
import os
import pickle
import matplotlib.pyplot as plt

from theory.params.cosmo_par import CosmoPar
from theory.params.survey_par import SurveyPar
from theory.data_vector.data_spec import PowerSpectrum3DSpec
from theory.data_vector.data_vector import PowerSpectrum3D
from theory.utils import file_tools

from theory.utils.profiler import profiler

def get_data_vec_p3d(info):
    cosmo_par_file = info['cosmo_par_file']
    cosmo_par_fid_file = info['cosmo_par_fid_file']
    survey_par_file = info['survey_par_file']
    data_spec_dict = info['PowerSpectrum3D'] 

    cosmo_par = CosmoPar(cosmo_par_file)
    cosmo_par_fid = CosmoPar(cosmo_par_fid_file)

    survey_par = SurveyPar(survey_par_file)
    data_spec = PowerSpectrum3DSpec(survey_par, data_spec_dict)

    data_vec = PowerSpectrum3D(cosmo_par, cosmo_par_fid, survey_par, data_spec)
    
    return data_vec

def get_fn(info):
    file_tools.mkdir_p(info['plot_dir'])
    return os.path.join(info['plot_dir'], info['run_name'] + '.npy')

@profiler
def save_galaxy_ps(info):
    data_vec = get_data_vec_p3d(info)
    fn = get_fn(info)
    file_tools.save_file_npy(fn, data_vec.get('galaxy_ps'))
    return data_vec

if __name__ == '__main__':
    """
    Example usage:
        python3 -m theory.scripts.get_ps ./inputs_theory/get_ps.yaml
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file", type=str, default=None,
        help="path to config file."
    )

    command_line_args = parser.parse_args()

    with open(command_line_args.config_file) as file:
        info = yaml.load(file, Loader=yaml.FullLoader)
    print('info', info)

    save_galaxy_ps(info)
