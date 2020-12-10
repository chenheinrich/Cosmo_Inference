import numpy as np
import argparse
import yaml
import os
import pickle

from theory.params.cosmo_par import CosmoPar
from theory.params.survey_par import SurveyPar
from theory.data_vector.data_spec import DataSpec
from theory.data_vector.data_vector import DataVector, P3D, B3D
from theory.utils import file_tools

def get_ps(info):
    
    cosmo_par_file = info['cosmo_par_file']
    cosmo_par_fid_file = info['cosmo_par_fid_file']
    survey_par_file = info['survey_par_file']
    data_spec_dict = info['PowerSpectrum3D'] 

    cosmo_par = CosmoPar(cosmo_par_file)
    cosmo_par_fid = CosmoPar(cosmo_par_fid_file)

    survey_par = SurveyPar(survey_par_file)
    data_spec = DataSpec(survey_par, data_spec_dict)

    data_vec = P3D(cosmo_par, cosmo_par_fid, survey_par, data_spec)
    galaxy_ps = data_vec.get('galaxy_ps')
    return galaxy_ps

def get_fn(info):
    file_tools.mkdir_p(info['plot_dir'])
    return os.path.join(info['plot_dir'], info['run_name'] + '.npy')

#TODO to move to a different file
def compare_galaxy_ps(d1, d2):
    if np.allclose(d1, d2):
        print('Passed!')
    else:
        diff = d2 - d1
        frac_diff = diff/d1
        max_diff = np.max(np.abs(diff))
        max_frac_diff = np.max(np.abs(frac_diff))
        print('max_diff', max_diff)
        print('max_frac_diff', max_frac_diff)


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

    ps = get_ps(info)
    fn = get_fn(info)
    file_tools.save_file_npy(fn, ps)

    #TODO optional comparison
    d1 = np.load(fn)
    fn2 = './data/ps_base/ref.pickle'
    results = pickle.load(open(fn2, "rb"))
    d2 = results['galaxy_ps']
    compare_galaxy_ps(d1, d2)

    #TODO need to plot and make sure we can reproduce previous galaxy ps results + debug!
    #TODO NEXT: need to add plotting routines and unit tests (from old module, that can be tested now)
    