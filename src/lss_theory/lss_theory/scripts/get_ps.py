import numpy as np
import argparse
import yaml
import os
import pickle
import matplotlib.pyplot as plt

from lss_theory.params.cosmo_par import CosmoPar
from lss_theory.params.survey_par import SurveyPar
from lss_theory.data_vector import PowerSpectrum3DSpec
from lss_theory.data_vector import PowerSpectrum3D
from lss_theory.data_vector import GRSIngredientsCreator
from lss_theory.utils import file_tools

from lss_theory.utils.profiler import profiler

def get_data_vec_p3d(info):

    nonlinear = False # TODO hook to input file?

    cosmo_par_file = info['cosmo_par_file']
    cosmo_par_fid_file = info['cosmo_par_fid_file']
    survey_par_file = info['survey_par_file']
    print('info', info)
    data_spec_dict = info['PowerSpectrum3D'] 

    cosmo_par = CosmoPar(cosmo_par_file)
    cosmo_par_fid = CosmoPar(cosmo_par_fid_file)

    survey_par = SurveyPar(survey_par_file)
    data_spec = PowerSpectrum3DSpec(survey_par, data_spec_dict)

    creator = GRSIngredientsCreator()
    option = 'Camb'
    grs_ingredients = creator.create(option, survey_par, data_spec, nonlinear,\
        cosmo_par=cosmo_par, cosmo_par_fid=cosmo_par_fid)

    print('gaussian_bias = {}'.format(grs_ingredients.get('gaussian_bias')))

    data_vec = PowerSpectrum3D(grs_ingredients, survey_par, data_spec)
    
    return data_vec

def get_fn(info):
    file_tools.mkdir_p(info['output_dir'])
    return os.path.join(info['output_dir'], info['run_name'] + '.npy')

#@profiler
def get_galaxy_ps(info):
    data_vec = get_data_vec_p3d(info)
    return data_vec.get('galaxy_ps')

def save_galaxy_ps(info):
    galaxy_ps = get_galaxy_ps(info)
    fn = get_fn(info)
    file_tools.save_file_npy(fn, galaxy_ps)
    print('Saved galaxy_ps to file: {}'.format(fn))

if __name__ == '__main__':
    """
    Example usage:
        python3 -m lss_theory.scripts.get_ps ./lss_theory/sample_inputs/get_ps.yaml
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

    save_galaxy_ps(info)
