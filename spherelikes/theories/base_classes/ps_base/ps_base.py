from cobaya.theory import Theory
import numpy as np


class PowerSpectrumBase(Theory):

    params = {'nuisance_in': {'prior': {'min': 0, 'max': 1}, 'propose': 0.01, 'ref': 0.9},
              'tt_sum': {'derived': True}}

    z_list = [0.25, 0.5, 0.75, 1]
    k_list = np.linspace(0.001, 0.02, 21)

    def initialize(self):
        """called from __init__ to initialize"""

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
        return {'nuisance_in': None}

    def must_provide(self, **requirements):
        z_list = self.z_list
        k_max = 0.01
        nonlinear = (False, True)
        spec_Pk = {
            'z': z_list,
            'k_max': k_max,  # 1/Mpc
            'nonlinear': nonlinear,
        }
        if 'A' in requirements:
            return {"Pk_interpolator": spec_Pk, 'Cl': {'tt': 2500}, 'H0': None}

    def get_can_provide_params(self):
        return ['tt_sum']

    def calculate(self, state, want_derived=True, **params_values_dict):
        H0 = self.provider.get_param('H0')
        Cl = self.provider.get_Cl()
        Pk_interpolator = self.provider.get_Pk_interpolator(nonlinear=False)
        Pk_interpolator_nl = self.provider.get_Pk_interpolator(nonlinear=True)

        tt = Cl['tt']
        tt_sum = np.sum(tt)
        state['A'] = tt * 1.0
        state['derived'] = {'tt_sum': tt_sum}

        Pk = np.array(Pk_interpolator(self.z_list, self.k_list))
        print('Pk', Pk)  # Pk is nz x nk numpy array

        # TODO: Check robustness of the interpolator (or whatever method adopted)
        # 1) use get_Pk_grid to cross-check results of interpolation
        #    Code: Pk = self.provider.get_Pk_grid(nonlinear=False)
        #    Pk_grid is a tuple of three arrays (z, k, Pk), where z, k are 1-d, and Pk is nz x nk.
        # 2) plot out P(k) interpolated and before interpolation
        # 3) look at linear vs nonlinear
        #    understand problem w/ diff = -0.5 for linear scales w/ the interpolator

    def get_A(self, normalization=1):
        return self._current_state['A'] * normalization
