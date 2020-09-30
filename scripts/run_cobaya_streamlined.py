import sys
import os

from cobaya.run import run
from cobaya.yaml import yaml_load_file

from scripts.generate_ref import generate_ref
from scripts.generate_covariance import generate_covariance
from scripts.generate_data import generate_data

CWD = os.getcwd()
args = {
    'output_dir': CWD + '/data/ps_base_minimal/',
    'model_name': None,
    'model_yaml_file': CWD + '/inputs/cosmo_pars/planck2018_fiducial.yaml',
    'cobaya_yaml_file': CWD + '/inputs/cobaya_pars/ps_base_minimal.yaml',
    'theory_name': "theories.base_classes.ps_base.ps_base.PowerSpectrumBase"
}

info = yaml_load_file(args['cobaya_yaml_file'])

# generate reference cosmology results for AP
generate_ref(args)

# generate inverse covariance if it doesn't already exist
invcov_path = os.path.join(args['output_dir'], 'invcov.npy')
if not os.path.exists(invcov_path):
    #  pulls same survey par file as used during sampling to add shot noise to covariance
    theory_name = args['theory_name']
    args['input_survey_pars'] = info['theory'][theory_name]['survey_pars_file_name']
    generate_covariance(args)
    # TODO additional checks that some criterias are satisfied?
print('Skip making inverse covariance. Found invcov file at\n    {}'.format(invcov_path))

# generate simulated data vector w/ fnl = 1.0
args['model_yaml_file'] = CWD + '/inputs/cosmo_pars/planck2018_fnl_1p0.yaml'
generate_data(args)

for k, v in {"-f": "force", "-r": "resume", "-d": "debug"}.items():
    if k in sys.argv:
        info[v] = True

updated_info, sampler = run(info)
