import numpy as np
import argparse
import yaml
import os
import pickle
import matplotlib.pyplot as plt

from lss_theory.params.cosmo_par import CosmoPar
from lss_theory.params.survey_par import SurveyPar

from lss_theory.data_vector import PowerSpectrum3DSpec, Bispectrum3DRSDSpec
from lss_theory.data_vector import PowerSpectrum3D, Bispectrum3DRSD
from lss_theory.data_vector import GRSIngredientsCreator

from lss_theory.covariance.bis_var import Bispectrum3DVariance
from lss_theory.plotting.bis_sn_plotter import BisSNPlotter
from lss_theory.utils import file_tools


def get_data_vec_ps(info):
    cosmo_par_file = info['cosmo_par_file']
    cosmo_par_fid_file = info['cosmo_par_fid_file']
    survey_par_file = info['survey_par_file']
    data_spec_dict = info['PowerSpectrum3D'] 

    cosmo_par = CosmoPar(cosmo_par_file)
    cosmo_par_fid = CosmoPar(cosmo_par_fid_file)

    survey_par = SurveyPar(survey_par_file)
    data_spec = PowerSpectrum3DSpec(survey_par, data_spec_dict)

    creator = GRSIngredientsCreator()
    option = 'Camb'
    grs_ingredients = creator.create(option, survey_par, data_spec,
        cosmo_par=cosmo_par, cosmo_par_fid=cosmo_par_fid)

    data_vec = PowerSpectrum3D(grs_ingredients, survey_par, data_spec)
    
    return data_vec

def get_ps(info):
    data_vec = get_data_vec_ps(info)
    galaxy_ps = data_vec.get('galaxy_ps')
    return galaxy_ps

def get_fn(info):
    file_tools.mkdir_p(info['plot_dir'])
    return os.path.join(info['plot_dir'], info['run_name'] + '.npy')

def get_b3d_rsd(info):
    
    cosmo_par_file = info['cosmo_par_file']
    cosmo_par_fid_file = info['cosmo_par_fid_file']
    survey_par_file = info['survey_par_file']
    data_spec_dict = info['Bispectrum3D'] 

    cosmo_par = CosmoPar(cosmo_par_file)
    cosmo_par_fid = CosmoPar(cosmo_par_fid_file)

    survey_par = SurveyPar(survey_par_file)
    data_spec = Bispectrum3DRSDSpec(survey_par, data_spec_dict)

    creator = GRSIngredientsCreator()
    option = 'Camb'
    grs_ingredients = creator.create(option, survey_par, data_spec,
        cosmo_par=cosmo_par, cosmo_par_fid=cosmo_par_fid)

    b3d_rsd= Bispectrum3DRSD(grs_ingredients, survey_par, data_spec)

    return b3d_rsd

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
    data_spec = Bispectrum3DRSDSpec(survey_par, data_spec_dict)
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
    print('info = {}'.format(info))

    data_spec_bis = get_data_spec_bis(info)
    data_vec = get_b3d_rsd(info)

    cosmo_par_fid_file = info['cosmo_par_fid_file']
   
    info['cosmo_par_file'] = cosmo_par_fid_file 
    data_vec2 = get_b3d_rsd(info)
    
    if info['plot_with_error'] is True:
        info['cosmo_par_file'] = cosmo_par_fid_file 
        p3d = get_data_vec_ps(info)
        bis_var = Bispectrum3DVariance(p3d, data_spec_bis, info['Bispectrum3DVariance'])

        #HACK
        iz = 0
        ib = 0
        itri = 0
        print('bis var for iz = %s, ib = %s, itri = %s: '%(iz, ib, itri), bis_var.bis_error[ib, iz, itri]**2)
    else:
        bis_var = None
    
    bis_plotter = BisSNPlotter(data_vec, data_spec_bis, data_vec2=data_vec2, \
        bis_var=bis_var, plot_dir=info['plot_dir'], do_run_checks=False)
    bis_plotter.make_plots()
