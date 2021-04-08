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

from spherelikes.utils.log import LoggedError, class_logger
from spherelikes.utils import constants
#HACK
#from spherelikes.params import SurveyPar
from spherelikes.params_generator import TheoryParGenerator

from galaxy_3d.theory.data_vector import PowerSpectrum3D as PowerSpectrum3D_standalone
from galaxy_3d.theory.data_vector import GRSIngredientsCreator
from galaxy_3d.theory.data_vector import PowerSpectrum3DSpec
from galaxy_3d.theory.params.cosmo_par import CosmoPar
from galaxy_3d.theory.params.survey_par import SurveyPar

from galaxy_3d.theory.data_vector import Bispectrum3DRSD as Bispectrum3DRSD_standalone
from galaxy_3d.theory.data_vector import Bispectrum3DRSDSpec

logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.ticker').disabled = True


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

class GRSIngredients(Theory):

    cosmo_par_fid_file = './inputs/cosmo_pars/planck2018_fiducial.yaml'
    cosmo_par_fid = CosmoPar(cosmo_par_fid_file)

    survey_par_file = './inputs/survey_pars/survey_pars_v28_base_cbe.yaml'
    survey_par = SurveyPar(survey_par_file)

    nz = survey_par.get_nz()
    nsample = survey_par.get_nsample()
    #HACK is this ok?
    params = get_params_for_survey_par(survey_par, fix_to_default=True)

    nk = 2  # 21  # 211  # number of k points (to be changed into bins)
    nmu = 2  # 5  # number of mu bins

    h = 0.68
    kmin = 1e-3 * h # in 1/Mpc
    kmax = 0.2 * h # in 1/Mpc

    #grs (might take away do_test, do_test_plot, test_plot_names stuff)
    is_reference_model = False

    def initialize(self):
        """called from __init__ to initialize"""
        self.logger = class_logger(self)
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

    #TODO NEXT: clean up this part for GRSIngredients!!
    def must_provide(self, **requirements):
        z_list = self.survey_par.get_zmid_array()
        #HACK (to work with model)
        z_list_2 = self.survey_par.get_zlo_array()
        z_list_3 = self.survey_par.get_zhi_array()
        self.logger.debug('z_list = {}'.format(z_list))
        #z_list = self.z_list #TODO find way to pass this properly
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
                'CAMBdata': None,
                #HACK (to work with model.py)
                'comoving_radial_distance': {'z': np.hstack((z_list, z_list_2, z_list_3))},
            }

    def calculate(self, state, want_derived=True, **params_values_dict):

        nonlinear = False # TODO to make an input later

        self.logger.debug('Getting survey parameters from file: {}'\
            .format(self.survey_par_file))
        self.survey_par = SurveyPar(self.survey_par_file)

        self.data_spec_dict = {
            'nk': self.nk, # number of k points (to be changed into bins)
            'nmu': self.nmu, # number of mu bins
            'kmin': self.kmin, # equivalent to 0.001 h/Mpc
            'kmax': self.kmax, # equivalent to 0.2 h/Mpc
        }
        self.logger.debug('self.data_spec_dict: {}'.format(self.data_spec_dict))

        self.data_spec = PowerSpectrum3DSpec(self.survey_par, self.data_spec_dict)
        cosmo_par_fid = CosmoPar(self.cosmo_par_fid_file) 

        self.logger.debug('About to get grs_ingredients')
        
        creator = GRSIngredientsCreator()
        grs_ingredients = creator.create('Cobaya',\
            self.survey_par, self.data_spec, nonlinear,\
            cosmo_par_fid=cosmo_par_fid, \
            provider=self.provider, **params_values_dict)

        state['grs_ingredients'] = grs_ingredients #TODO instance of a class (ok?)

    def get_grs_ingredients(self):
        return self._current_state['grs_ingredients']