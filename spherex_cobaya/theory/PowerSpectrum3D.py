from cobaya.theory import Theory
from cobaya.yaml import yaml_load_file
import numpy as np
import time
import sys
import os
import pickle
import matplotlib.pyplot as plt
import pathlib
import logging

from spherex_cobaya.utils.log import LoggedError, class_logger
from spherex_cobaya.utils import constants
#HACK
#from spherex_cobaya.params import SurveyPar
from spherex_cobaya.params_generator import TheoryParGenerator

from lss_theory.data_vector import PowerSpectrum3D as PowerSpectrum3D_standalone
from lss_theory.data_vector import GRSIngredientsCreator
from lss_theory.data_vector import PowerSpectrum3DSpec
from lss_theory.params.cosmo_par import CosmoPar
from lss_theory.params.survey_par import SurveyPar

logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.ticker').disabled = True

#TODO might want to delete some auxiliary functions
# if repeated in other files

def make_dictionary_for_base_params():
    base_params = {
        'fnl': {'prior': {'min': 0, 'max': 5},
                'ref': {'dist': 'norm', 'loc': 1.0, 'scale': 0.5},
                'propose': 0.001,
                'latex': 'f_{\rm{NL}}',
                },
        #'derived_param_p3d': {'derived': True},
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
            
            key = 'gaussian_bias_sample_%s_z_%s' % (isample + 1, iz + 1)
            
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

#TODO could refactor in the future the common code
# with Bispectrum3DBase and Bispectrum3DRSD
class PowerSpectrum3D(Theory):

    nk = 2  # number of k points (to be changed into bins)
    nmu = 2  # number of mu bins

    h = 0.68
    kmin = 1e-3 * h # in 1/Mpc
    kmax = 0.2 * h # in 1/Mpc

    def initialize(self):
        """called from __init__ to initialize"""
        
        self.logger = class_logger(self)

        self.data_spec_dict = {
            'nk': self.nk, # number of k points (to be changed into bins)
            'nmu': self.nmu, # number of mu bins
            'kmin': self.kmin, # equivalent to 0.001 h/Mpc
            'kmax': self.kmax, # equivalent to 0.2 h/Mpc
        }

        print('Done setting up PowerSpectrum3D')

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
        if 'galaxy_ps' in requirements:
            return {
                'grs_ingredients': None 
            }

    def calculate(self, state, want_derived=True, **params_values_dict):

        grs_ingredients = self.provider.get_grs_ingredients()
        self.survey_par = grs_ingredients._survey_par 
        self.data_spec = PowerSpectrum3DSpec(self.survey_par, self.data_spec_dict)

        data_vec = PowerSpectrum3D_standalone(grs_ingredients, self.survey_par, self.data_spec)
        galaxy_ps = data_vec.get('galaxy_ps')

        self.logger.debug('About to galaxy_ps')
        state['galaxy_ps'] = galaxy_ps

        # TODO placeholder for any derived paramter from this module
        #state['derived'] = {'derived_param_p3d': 1.0}

    def get_galaxy_ps(self):
        return self._current_state['galaxy_ps']
