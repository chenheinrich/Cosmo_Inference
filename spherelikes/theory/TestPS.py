import numpy as np

from cobaya.theory import Theory

from spherelikes.utils.log import LoggedError, class_logger

class TestPS(Theory):

    def initialize(self):
        """called from __init__ to initialize"""
        self.logger = class_logger(self)
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
        if 'galaxy_ps' in requirements:
            return {
                'H0': None,
                'CAMBdata': None,
                }

    def calculate(self, state, want_derived=True, **params_values_dict):

        galaxy_ps = 0.0
        state['galaxy_ps'] = galaxy_ps

    def get_galaxy_ps(self):
        return self._current_state['galaxy_ps']