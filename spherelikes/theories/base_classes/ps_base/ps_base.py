from cobaya.theory import Theory
import numpy as np


class PowerSpectrumBase(Theory):

    params = {
        'fnl': {'prior': {'min': 0, 'max': 5}, 'propose': 0.1, 'ref': 1.0},
        'gaussian_bias_1': {'prior': {'min': 0.8, 'max': 2.0}, 'propose': 0.1, 'ref': 1.0, 'latex': 'b_g^{1}'},
        'gaussian_bias_2': {'prior': {'min': 0.8, 'max': 2.0}, 'propose': 0.1, 'ref': 1.0, 'latex': 'b_g^{2}'},
        # TODO how to make this variable number of bins?
        'derived_param': {'derived': True}
    }

    n_sample = 2
    nz = 4
    nk = 21
    z_list = [0.25, 0.5, 0.75, 1]
    k_list = np.linspace(0.001, 0.02, nk)

    _delta_c = 1.686
    _k0 = 0.05  # 1/Mpc

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
        return {}

    def must_provide(self, **requirements):
        z_list = self.z_list
        k_max = 0.01
        nonlinear = (False, True)
        spec_Pk = {
            'z': z_list,
            'k_max': k_max,  # 1/Mpc
            'nonlinear': nonlinear,
        }
        if 'galaxy_ps' in requirements:
            return {
                'Pk_interpolator': spec_Pk,
                # 'Pk_grid': spec_Pk,
                'Cl': {'tt': 2500},
                'H0': None,
                'angular_diameter_distance': {'z': z_list},
                'Hubble': {'z': z_list},
                'omegam': None,
                'As': None,
                'ns': None,
            }

    def get_can_provide_params(self):
        return ['derived_param']

    def calculate(self, state, want_derived=True, **params_values_dict):
        cl = self.provider.get_Cl()

        # matter_power = self.calculate_matter_power()
        state['matter_power'] = 1.0  # matter_power

        galaxy_transfer = self.calculate_galaxy_transfer(**params_values_dict)
        state['galaxy_transfer'] = 1.0  # galaxy_transfer

        # AP_factor = self.calculate_AP_factor()
        state['AP_factor'] = 1.0  # AP_factor

        # TODO NEXT write a loop here, figure out data format first!!
        # galaxy_ps = AP_factor * matter_power[nz, nk] *\
        #    galaxy_transfer[j1, nk] * galaxy_transfer[j2, nk]
        galaxy_ps = 1.0
        state['galaxy_ps'] = galaxy_ps

        # TODO placeholder for any derived paramter from this module
        state['derived'] = {'derived_param': 1.0}

    def get_galaxy_ps(self):
        return self._current_state['galaxy_ps']

    def calculate_matter_power(self):
        """ matter_power is nz x nk numpy array"""
        Pk_interpolator = self.provider.get_Pk_interpolator(nonlinear=False)
        matter_power = np.array(Pk_interpolator(self.z_list, self.k_list))
        return matter_power

        # TODO: Check robustness of the interpolator (or whatever method adopted)
        # 1) use get_Pk_grid to cross-check results of interpolation
        #    Code: Pk = self.provider.get_Pk_grid(nonlinear=False)
        #    Pk_grid is a tuple of three arrays (z, k, Pk), where z, k are 1-d, and Pk is nz x nk.
        # 2) plot out P(k) interpolated and before interpolation
        # 3) look at linear vs nonlinear
        #    understand problem w/ diff = -0.5 for linear scales w/ the interpolator

    def calculate_galaxy_transfer(self, **params_values_dict):
        galaxy_transfer = \
            self._calculate_rsd_kaiser() * \
            self._calculate_nl_damping() * \
            self._calculate_rsd_nl() * \
            self._calculate_galaxy_bias(**params_values_dict)
        return galaxy_transfer

    def _calculate_rsd_kaiser(self):
        # TODO to complete
        # omegam = self.provider.get_param('omegam')
        return 1.0

    def _calculate_nl_damping(self):
        return 1.0

    def _calculate_rsd_nl(self):
        return 1.0

    # TODO might want this in bispectrum class too...
    # could make into a separate theory class that returns galaxy bias
    # similarly for galaxy transfer and alpha (alpha needed by bispectrum too)
    def _calculate_galaxy_bias(self, **params_values_dict):
        gaussian_bias_per_sample = self._calculate_gaussian_bias_array(
            **params_values_dict)
        gaussian_bias_per_sample = gaussian_bias_per_sample[:, None]
        alpha = self._calculate_alpha()
        galaxy_bias = np.array([
            gaussian_bias_per_sample[j]
            + 2.0 * params_values_dict['fnl']
            * (gaussian_bias_per_sample[j] - 1.0) * self._delta_c / alpha
            for j in range(self.n_sample)
        ])
        print('galaxy_bias.shape', galaxy_bias.shape)
        assert galaxy_bias.shape == (self.n_sample, self.nz, self.nk)

        # TODO NEXT decide on data format here!!! and how to multiply to get the right format.
        # galaxy_bias.shape = (n_sample, 1); alpha.shape = (nz, nk)
        # want galaxy_bias.shape = (n_sample, nz, nk)  ?
        # TODO add unit test check galaxy_bias.shape
        # galaxy_power: (n_spectra_per_z x nz x nk x nmu) numpy array
        # matter_power: (nz x nk) numpy array
        # galaxy_transfer: (n_sample x nz x nk x nmu)

        return galaxy_bias

    def _calculate_gaussian_bias_array(self, **params_values_dict):
        """"Returns a 1-d numpy array of gaussian galaxy bias, for each galaxy sample ."""
        keys = ['gaussian_bias_%s' % (i) for i in range(1, self.n_sample + 1)]
        gaussian_bias = np.array([params_values_dict[key] for key in keys])
        print('==>gaussian_bias', gaussian_bias)
        # TODO add unit test check gaussian_bias.shape == (self.n_sample, )
        return gaussian_bias

    def _calculate_alpha(self):
        # TODO double check w/ Jesus Torrado that this is the right way to get T(k)
        # option 1:
        # params, results = self.provider.get_CAMB_transfers()
        # a = results.get_matter_transfer_data()
        # normalized to 1 at low-k; T(k)/k^2; should be cst at low-k
        # t = a.transfer_z('delta_tot')
        # k = a.transfer_z('k/h')
        # H0 = self.provider.get_param('H0')
        # fnl = params_values_dict['fnl']

        initial_power = self._calculate_initial_power()
        initial_power = np.transpose(initial_power[:, None])
        alpha = (5.0 / 3.0) * \
            np.sqrt(self.calculate_matter_power() /
                    self._calculate_initial_power())
        return alpha

    def _calculate_initial_power(self):
        # TODO want to make sure this corresponds to the same as camb module
        # Find out how to get it from camb itself (we might have other parameters
        # like nrun, nrunrun; and possibly customized initial power one day)
        """Returns 1-d numpy array of initial power spectrum evaluated at k_list."""
        k0 = self._k0  # 1/Mpc
        As = self.provider.get_param('As')
        ns = self.provider.get_param('ns')
        k_array = np.array(self.k_list)
        initial_power = (2.0 * np.pi**2) / (k_array**3) * \
            As * (k_array / k0)**(ns - 1.0)
        return initial_power

    # Common ingredients needed by bispectrum:
    # galaxy bias, alpha, rsd factor
    # TODO might want to think about how to interface, so only calculate once.
    # Do this later at the design doc level.
