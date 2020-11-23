import numpy 
import argparse
import yaml
import os

from theory.params.cosmo_par import CosmoPar
from theory.params.survey_par import SurveyPar
from theory.data_vector.data_spec import DataSpec
from theory.data_vector.data_vector import DataVector, P3D, B3D

def get_fn_data_vec(info):
    return os.path.join(info['plot_dir'], info['run_name'] + '.png')

def main(config_file):

    with open(config_file) as file:
        info = yaml.load(file, Loader=yaml.FullLoader)
    
    print('info', info)
    
    cosmo_par_file = info['cosmo_par_file']
    cosmo_par_fid_file = info['cosmo_par_fid_file']
    survey_par_file = info['survey_par_file']
    #TODO might want to just specify data_spec in a block 
    data_spec_dict = info['PowerSpectrum3D'] 
    fn_data_vec = get_fn_data_vec(info)

    cosmo_par = CosmoPar(cosmo_par_file)
    cosmo_par_fid = CosmoPar(cosmo_par_fid_file)

    survey_par = SurveyPar(survey_par_file)
    data_spec = DataSpec(survey_par, data_spec_dict)

    data_vec = P3D(cosmo_par, cosmo_par_fid, survey_par, data_spec)
    data_vec.calculate()
    data_vec.save(fn_data_vec)

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
    main(command_line_args.config_file)
