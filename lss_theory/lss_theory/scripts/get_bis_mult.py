import numpy as np
import argparse
import yaml
import os

from lss_theory.params.cosmo_par import CosmoPar
from lss_theory.params.survey_par import SurveyPar
from lss_theory.data_vector import BispectrumMultipoleSpec
from lss_theory.data_vector import BispectrumMultipole
from lss_theory.data_vector import GRSIngredientsCreator
from lss_theory.utils import file_tools
from lss_theory.plotting.bis_mult_plotter import BisMultPlotter

from lss_theory.utils.profiler import profiler

def get_data_spec(info):
    survey_par_file = info['survey_par_file']
    data_spec_dict = info['BispectrumMultipole'] 
    survey_par = SurveyPar(survey_par_file)
    data_spec = BispectrumMultipoleSpec(survey_par, data_spec_dict)
    return data_spec

def get_data_vector(info):
    
    cosmo_par_file = info['cosmo_par_file']
    cosmo_par_fid_file = info['cosmo_par_fid_file']
    survey_par_file = info['survey_par_file']
    data_spec_dict = info['BispectrumMultipole'] 

    cosmo_par = CosmoPar(cosmo_par_file)
    cosmo_par_fid = CosmoPar(cosmo_par_fid_file)

    survey_par = SurveyPar(survey_par_file)
    data_spec = BispectrumMultipoleSpec(survey_par, data_spec_dict)

    creator = GRSIngredientsCreator()
    option = 'Camb'
    nonlinear = False
    grs_ingredients = creator.create(option, survey_par, data_spec,
        nonlinear, cosmo_par=cosmo_par, cosmo_par_fid=cosmo_par_fid)

    data_vec = BispectrumMultipole(grs_ingredients, survey_par, data_spec)
    
    return data_vec

def get_fn(info):
    file_tools.mkdir_p(info['plot_dir'])
    return os.path.join(info['plot_dir'], info['run_name'] + '.npy')

#@profiler
def get_galaxy_bis_mult(info):
    data_vec = get_data_vector(info)
    return data_vec.get('galaxy_bis')

def save_galaxy_bis_mult(info):
    galaxy_bis = get_galaxy_bis_mult(info)
    fn = get_fn(info)
    file_tools.save_file_npy(fn, galaxy_bis)
    print('Saved galaxy_bis_mult to file: {}'.format(fn))

def get_plot_dir(info):
    plot_dir = info['plot_dir']
    subdir_name = get_subdir_name_for_bis_mult(info)
    plot_dir = os.path.join(plot_dir, subdir_name)
    return plot_dir

#TODO need to test that refactoring is successful by running this
def get_subdir_name_for_bis_mult(info):

    bis_mult_info = info['BispectrumMultipole']
    nk = bis_mult_info['nk']

    lmax = bis_mult_info['multipole_info']['lmax']

    ori_info = bis_mult_info['triangle_orientation_info']
    do_folded_signal = ori_info['do_folded_signal']
    nbin_cos_theta1 = ori_info['nbin_cos_theta1']
    nbin_phi12 = ori_info['nbin_phi12']

    cosmo_name = os.path.splitext(os.path.basename(info['cosmo_par_file']))[0]
    
    subdir_name = 'cosmo_%s/nk_%s/lmax_%s/do_folded_signal_%s/theta_phi_%s_%s/'\
        %(cosmo_name, nk, lmax, do_folded_signal, nbin_cos_theta1, nbin_phi12)

    return subdir_name    

if __name__ == '__main__':
    """
    Example usage:
        python3 -m lss_theory.scripts.get_bis_mult ./lss_theory/inputs/get_bis_mult.yaml 
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
    
    save_galaxy_bis_mult(info)
       
    data_vec = get_data_vector(info)

    bis_plotter = BisMultPlotter(data_vec, data_spec, plot_dir=get_plot_dir(info), do_run_checks=False)
    bis_plotter.make_plots()


