import numpy as np

from cobaya.theory import Theory

from spherelikes.utils.log import LoggedError, class_logger
from galaxy_3d.theory.data_vector import PowerSpectrum3DSpec
from galaxy_3d.theory.params.survey_par import SurveyPar
from galaxy_3d.theory.params.cosmo_par import CosmoPar
from galaxy_3d.theory.data_vector import GRSIngredientsCreator
from galaxy_3d.theory.data_vector.grs_ingredients import GRSIngredients

from spherelikes.theory.GRSIngredients import get_params_for_survey_par
class TestPS(Theory):

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

        self.nonlinear = False # TODO to make an input later

        self.logger.debug('Getting survey parameters from file: {}'\
            .format(self.survey_par_file))

        self.survey_par = SurveyPar(self.survey_par_file)
        #TODO inconsistency, using survey par in must_provide, 
        # is that the class default, or what's created in calculate?

        self.data_spec_dict = {
            'nk': self.nk, # number of k points (to be changed into bins)
            'nmu': self.nmu, # number of mu bins
            'kmin': self.kmin, # equivalent to 0.001 h/Mpc
            'kmax': self.kmax, # equivalent to 0.2 h/Mpc
        }
        self.logger.debug('self.data_spec_dict: {}'.format(self.data_spec_dict))

        self.data_spec = PowerSpectrum3DSpec(self.survey_par, self.data_spec_dict)

        self.cosmo_par_fid = CosmoPar(self.cosmo_par_fid_file) 

        from theory.data_vector.cosmo_interface import CosmoInterfaceCreator

        self.z = self.survey_par.get_zmid_array()
        
        self.cosmo_creator = CosmoInterfaceCreator()
        option_fid = 'Camb'
        self.cosmo_fid = self.cosmo_creator.create(option_fid, self.z, self.nonlinear, \
            cosmo_par=self.cosmo_par_fid)

        print('Done setting up TestPS')

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

        if 'galaxy_ps' in requirements:
            return {
                'H0': None,
                'CAMBdata': None,
                #'grs_ingredients': None
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

        self.logger.debug('About to get grs_ingredients')
    
        option = 'Cobaya'
        print('initializing cobaya cosmo')
        cosmo = self.cosmo_creator.create(option, self.z, self.nonlinear,
            cosmo_par=None, \
            provider=self.provider)
        
        #TODO this works if we pass cosmo object instead of creating it using cosmo_par_fid inside create()
        # Don't understand why.
        #grs_creator = GRSIngredientsCreator(self.survey_par, self.data_spec, self.nonlinear, self.cosmo_par_fid)
        #grs_ingredients = grs_creator.create('Cobaya',\
        #    self.survey_par, self.data_spec, self.nonlinear,\
        #    cosmo_par_fid=self.cosmo_par_fid, \
        #    #cosmo_par_fid=cosmo, \
        #    provider=self.provider, **params_values_dict)

        grs_ingredients = GRSIngredients(cosmo, self.cosmo_fid, self.survey_par, \
            self.data_spec, **params_values_dict)
        
        shape = (15, 11, 21, 5)
        galaxy_ps = np.zeros(shape)
        state['galaxy_ps'] = galaxy_ps

        #state['grs_ingredients'] = grs_ingredients #TODO instance of a class (ok?)

    def calculate_ps(self, state, want_derived=True, **params_values_dict):

        nonlinear = False # TODO to make an input later

        self.data_spec_dict = {
            'nk': self.nk, # number of k points (to be changed into bins)
            'nmu': self.nmu, # number of mu bins
            'kmin': self.kmin, # equivalent to 0.001 h/Mpc
            'kmax': self.kmax, # equivalent to 0.2 h/Mpc
        }
        self.logger.debug('self.data_spec_dict: {}'.format(self.data_spec_dict))

        self.logger.debug('About to get grs_ingredients')
        
        grs_ingredients = self.provider.get_grs_ingredients()
        self.survey_par = grs_ingredients._survey_par #TODO decide if this is ok

        #self.data_spec = PowerSpectrum3DSpec(self.survey_par, self.data_spec_dict)
        #TODO this is not cosmology dependent

        #self.logger.debug('About to get PowerSpectrum3D_standalone')
        #data_vec = PowerSpectrum3D_standalone(grs_ingredients, self.survey_par, self.data_spec)

        #galaxy_ps = data_vec.get('galaxy_ps')

        self.logger.debug('About to galaxy_ps')

        shape = (15, 11, 21, 5)
        galaxy_ps = np.zeros(shape)
        state['galaxy_ps'] = galaxy_ps

        # TODO placeholder for any derived paramter from this module
        #state['derived'] = {'derived_param_p3d': 1.0}

    def get_galaxy_ps(self):
        return self._current_state['galaxy_ps']