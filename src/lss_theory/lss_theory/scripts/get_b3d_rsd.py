import numpy as np
import argparse
import yaml
import os
import pickle

from lss_theory.params.cosmo_par import CosmoPar
from lss_theory.params.survey_par import SurveyPar
from lss_theory.data_vector.data_spec import Bispectrum3DRSDSpec_Theta1Phi12
from lss_theory.data_vector import Bispectrum3DRSD
from lss_theory.data_vector import GRSIngredientsCreator
from lss_theory.utils import file_tools
from lss_theory.plotting.bis_plotter import Bispectrum3DRSDPlotter
from lss_theory.plotting.triangle_spec_plotter import TriangleSpecTheta1Phi12Plotter

from lss_theory.utils.profiler import profiler

def get_data_spec(info):
    survey_par_file = info['survey_par_file']
    data_spec_dict = info['Bispectrum3DRSD'] 
    survey_par = SurveyPar(survey_par_file)
    data_spec = Bispectrum3DRSDSpec_Theta1Phi12(survey_par, data_spec_dict)
    return data_spec

def get_b3d_rsd(info):
    
    cosmo_par_file = info['cosmo_par_file']
    cosmo_par_fid_file = info['cosmo_par_fid_file']
    survey_par_file = info['survey_par_file']
    data_spec_dict = info['Bispectrum3DRSD'] 

    overwrite_cosmo_par_dict = info.get('overwrite_cosmo_par', {}) 
    overwrite_other_par_dict = info.get('overwrite_other_par', {}) 

    print('overwrite_cosmo_par_dict', overwrite_cosmo_par_dict)
    print('overwrite_other_par_dict', overwrite_other_par_dict)

    cosmo_par = CosmoPar(cosmo_par_file, overwrite_dict=overwrite_cosmo_par_dict)
    cosmo_par_fid = CosmoPar(cosmo_par_fid_file)

    survey_par = SurveyPar(survey_par_file)
    data_spec = Bispectrum3DRSDSpec_Theta1Phi12(survey_par, data_spec_dict)

    creator = GRSIngredientsCreator()
    option = 'Camb'
    nonlinear = False
    grs_ingredients = creator.create(option, survey_par, data_spec,\
        nonlinear, cosmo_par=cosmo_par, cosmo_par_fid=cosmo_par_fid, \
        **overwrite_other_par_dict) 

    data_vec = Bispectrum3DRSD(grs_ingredients, survey_par, data_spec)
    
    return data_vec

def get_fn(info):
    file_tools.mkdir_p(info['output_dir'])
    return os.path.join(info['output_dir'], info['run_name'] + '.npy')

#@profiler
def get_galaxy_bis(info):
    data_vec = get_b3d_rsd(info)
    return data_vec.get('galaxy_bis')

def save_galaxy_bis_rsd(info):
    galaxy_bis = get_galaxy_bis(info)
    fn = get_fn(info)
    file_tools.save_file_npy(fn, galaxy_bis)
    print('Saved galaxy_bis to file: {}'.format(fn))

if __name__ == '__main__':
    """
    Example usage:
        python3 -m lss_theory.scripts.get_b3d_rsd ./lss_theory/sample_inputs/get_b3d_rsd.yaml 
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

    data_spec = get_data_spec(info)
    triangle_plotter = TriangleSpecTheta1Phi12Plotter(data_spec._triangle_spec, plot_dir=info['plot_dir'])
    triangle_plotter.make_plots()

    save_galaxy_bis_rsd(info)
    
    if info['do_plot'] == True:
        data_vec = get_b3d_rsd(info)
        bis_plotter = Bispectrum3DRSDPlotter(data_vec, data_spec, plot_dir=info['plot_dir'], do_run_checks=False)
        bis_plotter.make_plots()

