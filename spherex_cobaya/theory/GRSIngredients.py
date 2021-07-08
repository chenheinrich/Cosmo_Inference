from cobaya.theory import Theory
import numpy as np

from spherex_cobaya.utils.log import LoggedError, class_logger
#HACK
#from spherex_cobaya.params import SurveyPar
from spherex_cobaya.params_generator import TheoryParGenerator

from lss_theory.data_vector import GRSIngredients as GRSIng
from lss_theory.data_vector import PowerSpectrum3DSpec
from lss_theory.params.cosmo_par import CosmoPar
from lss_theory.params.survey_par import SurveyPar
from lss_theory.params.other_par import OtherPar

from lss_theory.data_vector.cosmo_interface import CosmoInterfaceCreator

def make_dictionary_for_base_params():
    base_params = {
        'fnl': {'prior': {'min': 0, 'max': 5},
                'ref': {'dist': 'norm', 'loc': 1.0, 'scale': 0.5},
                'propose': 0.001,
                'latex': 'f_{\rm{NL}}',
                },
        #'derived_param': {'derived': True},
    }
    return base_params

def make_dictionary_for_bias_params(survey_par, \
    fix_to_default=False, include_latex=True,
    prior_name=None, prior_fractional_delta=0.03):
    """Returns a nested dictionary of galaxy bias parameters, 
    with keys (e.g. prior, ref, propose, latex, value) needed
    by cobaya sampler.

    Args:
        survey_par_file: path to survey parameter file. 
        fix_to_default (optional): boolean to fix parameters 
            to default values given by the survey_par_file.
        include_latex (optional): boolean to include latex
            string for the parameters (e.g. set to false if 
            just want a value returned).
        prior_name (optional): 'tight_prior' or 'uniform'
        prior_fractional_delta (optional): a float between 0 and 1
            to set the standard deviation of distributions 
            as a fraction of the bias value; default is 0.03.

    """
    
    prior_name = (prior_name or 'uniform')

    bias_default = survey_par.get_galaxy_bias_array()
    nsample = survey_par.get_nsample()
    nz = survey_par.get_nz ()
    
    bias_params = {}
    for isample in range(nsample):
        for iz in range(nz):
            
            key = 'gaussian_bias_s%s_z%s' % (isample + 1, iz + 1)
            
            latex = 'b_g^{%i}(z_{%i})' % (isample + 1, iz + 1)
            default_value = bias_default[isample, iz]
            
            scale = default_value * prior_fractional_delta
            scale_ref = scale/10.0

            if prior_name == 'uniform':
                delta = 1.0
                prior = {'min': max(0.5, default_value - delta), 'max': default_value + delta}
            elif prior_name == 'tight_prior':
                prior = {'dist': 'norm', 'loc': default_value, 'scale': scale}
            
            if fix_to_default is True:
                if include_latex is True:
                    value = {'value': default_value, 
                            'latex': latex
                            }
                else: 
                    value = default_value
            else:
                value = {'prior': prior,
                        'ref': {'dist': 'norm', 'loc': default_value, 'scale': scale_ref},
                        'propose': scale_ref,
                        'latex': latex,
                        }

            bias_params[key] = value

    return bias_params

def get_params_for_survey_par(survey_par, fix_to_default=False):

    base_params = make_dictionary_for_base_params()
    bias_params = make_dictionary_for_bias_params(\
        survey_par, fix_to_default=fix_to_default)
 
    base_params.update(bias_params)
    return base_params


class ParGenerator(TheoryParGenerator):

    def __init__(self):
        super().__init__()

    def get_params(self, survey_par, gen_info):
        """Update only the bias parameters according to the specifications."""
        bias_params = make_dictionary_for_bias_params(\
            survey_par, \
            **gen_info['bias']
        )
        return bias_params

class GRSIngredients(Theory):

    nk = 2  # number of k points (to be changed into bins)
    nmu = 2  # number of mu bins

    h = 0.68
    kmin = 1e-3 * h # in 1/Mpc
    kmax = 0.2 * h # in 1/Mpc

    nonlinear = False
    is_reference_model = False

    cosmo_fid_file = './inputs/cosmo_pars/planck2018_fiducial.yaml'
    survey_par_file = './inputs/survey_pars/survey_pars_v28_base_cbe.yaml'
    
    # Note: The following block needs to be reprocessed during initialize()
    # since the survey_par_file specified in the yaml file could be 
    # different than class default
    survey_par = SurveyPar(survey_par_file)
    nz = survey_par.get_nz()
    nsample = survey_par.get_nsample()
    params = get_params_for_survey_par(survey_par, fix_to_default=True)

    def initialize(self):
        """called from __init__ to initialize"""
        self.logger = class_logger(self)

        self.survey_par = SurveyPar(self.survey_par_file)
        self.nz = self.survey_par.get_nz()
        self.nsample = self.survey_par.get_nsample()
        self.z = self.survey_par.get_zmid_array()

        self.cosmo_par_fid = CosmoPar(self.cosmo_fid_file) 

        self.cosmo_creator = CosmoInterfaceCreator()

        # Note: This currently calls camb so needs to be done in the
        # initialize function (rather than e.g.. calculate())
        # something about fortran references counted differently
        # than in python and can cause malloc problems.
        self.cosmo_fid = self._get_cosmo_fid()

        self.data_spec = self._get_data_spec()

        print('Done setting up GRSIngredients')

    def _get_cosmo_fid(self):
        cosmo_fid = self.cosmo_creator.create('Camb', self.z, self.nonlinear, \
            cosmo_par=self.cosmo_par_fid)
        return cosmo_fid

    def _get_data_spec(self):
        data_spec_dict = {
            'nk': self.nk, # number of k points (to be changed into bins)
            'nmu': self.nmu, # number of mu bins
            'kmin': self.kmin, # equivalent to 0.001 h/Mpc
            'kmax': self.kmax, # equivalent to 0.2 h/Mpc
        }
        data_spec = PowerSpectrum3DSpec(self.survey_par, data_spec_dict)
        return data_spec

    def initialize_with_provider(self, provider):
        """
        Initialization after other components initialized, using Provider class
        instance which is used to return any dependencies (see calculate below).
        """
        self.provider = provider

    def get_requirements(self):
        """
        Return dictionary of derived parameters or other quantities that are needed
        by this component and should be calculated by another theory class.
        """
        return {}

    def must_provide(self, **requirements):
        
        #TODO to work with model, may need (simplify this):
        z_list_2 = self.survey_par.get_zlo_array()
        z_list_3 = self.survey_par.get_zhi_array()

        z_list = self.survey_par.get_zmid_array()
        k_max = 8.0
        nonlinear = (False, True)

        spec_Pk = {
            'z': z_list,
            'k_max': k_max,  # 1/Mpc
            'nonlinear': nonlinear,
        }

        if 'grs_ingredients' in requirements:
            return {
                'Pk_interpolator': spec_Pk,
                'Cl': {'tt': 2500},
                'H0': None,
                'angular_diameter_distance': {'z': z_list},
                'Hubble': {'z': z_list},
                'omegam': None,
                'As': None,
                'ns': None,
                'fsigma8': {'z': z_list},
                'sigma8': None,
                'CAMBdata': None, # needed for test
                # TODO to work with model.py, may need (simplify this)
                'comoving_radial_distance': {'z': np.hstack((z_list, z_list_2, z_list_3))},
                #'comoving_radial_distance': {'z': z_list}
            }

    def calculate(self, state, want_derived=True, **params_values_dict):

        self.logger.debug('About to get grs_ingredients')
        #TODO Not going through grs ingredient creator, may not work well.

        option = 'Cobaya'
        cosmo = self.cosmo_creator.create(option, self.z, self.nonlinear,
            cosmo_par=None, \
            provider=self.provider)

        other_par = OtherPar(params_values_dict)
        grs_ingredients = GRSIng(cosmo, self.cosmo_fid, self.survey_par, \
            self.data_spec, other_par)

        state['grs_ingredients'] = grs_ingredients 

    def get_grs_ingredients(self):
        return self._current_state['grs_ingredients']
