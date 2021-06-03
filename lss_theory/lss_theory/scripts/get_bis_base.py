import numpy as np
import argparse
import yaml
import os
import pickle

from lss_theory.params.cosmo_par import CosmoPar
from lss_theory.params.survey_par import SurveyPar
from lss_theory.data_vector import Bispectrum3DBaseSpec
from lss_theory.data_vector import Bispectrum3DBase
from lss_theory.data_vector import GRSIngredientsCreator
from lss_theory.utils import file_tools
from lss_theory.plotting.bis_plotter import Bispectrum3DBasePlotter
from lss_theory.plotting.triangle_spec_plotter import TriangleSpecPlotter

from lss_theory.utils.profiler import profiler

def get_data_spec(info):
    survey_par_file = info['survey_par_file']
    data_spec_dict = info['Bispectrum3DBase'] 
    survey_par = SurveyPar(survey_par_file)
    data_spec = Bispectrum3DBaseSpec(survey_par, data_spec_dict)
    return data_spec

def get_b3d_base(info):
    
    cosmo_par_file = info['cosmo_par_file']
    cosmo_par_fid_file = info['cosmo_par_fid_file']
    survey_par_file = info['survey_par_file']
    data_spec_dict = info['Bispectrum3DBase'] 

    cosmo_par = CosmoPar(cosmo_par_file)
    cosmo_par_fid = CosmoPar(cosmo_par_fid_file)

    survey_par = SurveyPar(survey_par_file)
    data_spec = Bispectrum3DBaseSpec(survey_par, data_spec_dict)

    creator = GRSIngredientsCreator()
    option = 'Camb'
    nonlinear = False
    grs_ingredients = creator.create(option, survey_par, data_spec,\
        nonlinear, cosmo_par=cosmo_par, cosmo_par_fid=cosmo_par_fid)

    data_vec = Bispectrum3DBase(grs_ingredients, survey_par, data_spec)
    
    return data_vec

def get_fn(info):
    file_tools.mkdir_p(info['result_dir'])
    return os.path.join(info['result_dir'], info['run_name'] + '.npy')

@profiler
def save_galaxy_bis(info):
    data_vec = get_b3d_base(info)
    fn = get_fn(info)
    file_tools.save_file_npy(fn, data_vec.get('galaxy_bis'))
    return data_vec

if __name__ == '__main__':
    """
    Example usage:
        python3 -m lss_theory.scripts.get_bis_base ./lss_theory/inputs/get_bis_base.yaml 
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
    triangle_plotter = TriangleSpecPlotter(data_spec._triangle_spec, plot_dir=info['plot_dir'])
    triangle_plotter.make_plots()

    data_vec = save_galaxy_bis(info)

    bis_plotter = Bispectrum3DBasePlotter(data_vec, data_spec, plot_dir=info['plot_dir'], do_run_checks=False)
    bis_plotter.make_plots()

