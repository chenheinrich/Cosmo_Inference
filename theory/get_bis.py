import numpy as np
import argparse
import yaml
import os
import pickle

from theory.params.cosmo_par import CosmoPar
from theory.params.survey_par import SurveyPar
from theory.data_vector.data_spec import DataSpecBispectrum
from theory.data_vector.data_vector import DataVector, B3D
from theory.utils import file_tools
from theory.plotting.bis_plotter import BisPlotter

def get_data_spec(info):
    survey_par_file = info['survey_par_file']
    data_spec_dict = info['Bispectrum3D'] 
    survey_par = SurveyPar(survey_par_file)
    data_spec = DataSpecBispectrum(survey_par, data_spec_dict)
    return data_spec

def get_data_vec_bis(info):
    
    cosmo_par_file = info['cosmo_par_file']
    cosmo_par_fid_file = info['cosmo_par_fid_file']
    survey_par_file = info['survey_par_file']
    data_spec_dict = info['Bispectrum3D'] 

    cosmo_par = CosmoPar(cosmo_par_file)
    cosmo_par_fid = CosmoPar(cosmo_par_fid_file)

    survey_par = SurveyPar(survey_par_file)
    data_spec = DataSpecBispectrum(survey_par, data_spec_dict)

    data_vec = B3D(cosmo_par, cosmo_par_fid, survey_par, data_spec)
    
    return data_vec

def get_fn(info):
    file_tools.mkdir_p(info['plot_dir'])
    return os.path.join(info['plot_dir'], info['run_name'] + '.npy')

if __name__ == '__main__':
    """
    Example usage:
        python3 -m theory.get_bis ./inputs_theory/single_bis.yaml 
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

    data_vec = get_data_vec_bis(info)
    fn = get_fn(info)
    file_tools.save_file_npy(fn, data_vec.get('galaxy_bis'))

    data_spec = get_data_spec(info)
    bis_plotter = BisPlotter(data_vec, data_spec)
    bis_plotter.make_plots()
