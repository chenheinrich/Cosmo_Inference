import numpy as np
import argparse
import yaml
import os
import pickle
import matplotlib.pyplot as plt

from theory.params.cosmo_par import CosmoPar
from theory.params.survey_par import SurveyPar

from theory.data_vector.data_spec import DataSpecPowerSpectrum, DataSpecBispectrumOriented
from theory.data_vector.data_vector import DataVector, P3D, B3D, B3D_RSD

from theory.covariance.bis_var import Bispectrum3DVariance
from theory.plotting.bis_sn_plotter import BisSNPlotter
from theory.utils import file_tools


def get_data_vec_ps(info):
    cosmo_par_file = info['cosmo_par_file']
    cosmo_par_fid_file = info['cosmo_par_fid_file']
    survey_par_file = info['survey_par_file']
    data_spec_dict = info['PowerSpectrum3D'] 

    cosmo_par = CosmoPar(cosmo_par_file)
    cosmo_par_fid = CosmoPar(cosmo_par_fid_file)

    survey_par = SurveyPar(survey_par_file)
    data_spec = DataSpecPowerSpectrum(survey_par, data_spec_dict)

    data_vec = P3D(cosmo_par, cosmo_par_fid, survey_par, data_spec)
    
    return data_vec

def get_ps(info):
    data_vec = get_data_vec_ps(info)
    galaxy_ps = data_vec.get('galaxy_ps')
    return galaxy_ps

def get_fn(info):
    file_tools.mkdir_p(info['plot_dir'])
    return os.path.join(info['plot_dir'], info['run_name'] + '.npy')

def get_data_vec_bis(info):
    
    cosmo_par_file = info['cosmo_par_file']
    cosmo_par_fid_file = info['cosmo_par_fid_file']
    survey_par_file = info['survey_par_file']
    data_spec_dict = info['Bispectrum3D'] 

    cosmo_par = CosmoPar(cosmo_par_file)
    cosmo_par_fid = CosmoPar(cosmo_par_fid_file)

    survey_par = SurveyPar(survey_par_file)
    data_spec = DataSpecBispectrumOriented(survey_par, data_spec_dict)

    data_vec = B3D_RSD(cosmo_par, cosmo_par_fid, survey_par, data_spec)
    
    return data_vec

def get_bis_plotter_fnl(info):
    #galaxy_ps = get_ps(info)
    #bis_var = Bispectrum3DVariance(galaxy_ps)
    p3d = get_data_vec_ps(info)
    bis_var = Bispectrum3DVariance(p3d)
    return bis_var

def get_data_spec_bis(info):
    survey_par_file = info['survey_par_file']
    data_spec_dict = info['Bispectrum3D'] 
    survey_par = SurveyPar(survey_par_file)
    data_spec = DataSpecBispectrumOriented(survey_par, data_spec_dict)
    return data_spec

if __name__ == '__main__':
    """
    Example usage:
        python3 -m theory.scripts.plot_bis_sn ./inputs_theory/plot_bis_sn.yaml
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

    data_spec_bis = get_data_spec_bis(info)
    data_vec = get_data_vec_bis(info)
   
    info['cosmo_par_file'] = info['cosmo_par_fid_file']
    data_vec2 = get_data_vec_bis(info)
    
    if info['plot_with_error'] is True:
        p3d = get_data_vec_ps(info)
        bis_var = Bispectrum3DVariance(p3d, data_spec_bis)
    else:
        bis_var = None
    
    bis_plotter = BisSNPlotter(data_vec, data_spec_bis, data_vec2=data_vec2, \
        bis_var=bis_var, plot_dir=info['plot_dir'], do_run_checks=False)
    bis_plotter.make_plots()
