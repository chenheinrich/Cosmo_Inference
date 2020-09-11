from cobaya.theory import Theory
import numpy as np
import time
import sys


class PowerSpectrumBase(Theory):

    params = {
        'fnl': {'prior': {'min': 0, 'max': 5}, 'propose': 0.1, 'ref': 1.0},
        'gaussian_bias_1': {'prior': {'min': 0.8, 'max': 2.0}, 'propose': 0.1, 'ref': 1.0, 'latex': 'b_g^{1}'},
        'gaussian_bias_2': {'prior': {'min': 0.8, 'max': 2.0}, 'propose': 0.1, 'ref': 1.0, 'latex': 'b_g^{2}'},
        # TODO how to make this variable number of bins?
        'derived_param': {'derived': True}
    }

    n_sample = 2  # number of galaxy samples
    nz = 4  # number of z bins
    nk = 21  # number of k points (to be changed into bins)
    nmu = 10  # number of mu bins
    z_list = [0.25, 0.5, 0.75, 1]  # TODO to change to SPHEREx numbers
    sigz = [0.01, 0.01]  # TODO to change to SPHEREx numbers
    z_array = np.array(z_list)
    k_array = np.linspace(0.001, 0.02, nk)
    mu_edges = np.linspace(0, 1, nmu + 1)
    mu_array = (mu_edges[:-1] + mu_edges[1:]) / 2.0

    _delta_c = 1.686
    _k0 = 0.05  # 1/Mpc
    _fraction_recon = 0.5  # reconstruction fration

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
                'fsigma8': {'z': z_list},
                'sigma8': None,
            }

    def get_can_provide_params(self):
        return ['derived_param']

    def calculate(self, state, want_derived=True, **params_values_dict):

        state['matter_power'] = self._calculate_matter_power()  # (nz, nk)
        state['galaxy_transfer'] = self._calculate_galaxy_transfer(
            **params_values_dict)  # (n_sample, nz, nk, nmu)
        state['AP_factor'] = self._calculate_AP_factor()  # (nz,)
        # (n_sample, nz, nk, nmu)
        state['galaxy_ps'] = self._calculate_galaxy_ps(state)
        # TODO placeholder for any derived paramter from this module
        state['derived'] = {'derived_param': 1.0}

    def get_galaxy_ps(self):
        return self._current_state['galaxy_ps']

    def _calculate_galaxy_ps(self, state):
        print('entered _calculate_galaxy_ps')
        n_ps = int(self.n_sample * (self.n_sample + 1) / 2)
        galaxy_ps = np.zeros((n_ps, self.nz, self.nk, self.nmu))
        jj = 0
        for j1 in range(self.n_sample):
            for j2 in range(j1, self.n_sample):
                galaxy_ps[jj] = \
                    state['AP_factor'].reshape((self.nz, 1, 1)) \
                    * state['matter_power'].reshape((self.nz, self.nk, 1)) \
                    * state['galaxy_transfer'][j1, :, :, :] \
                    * state['galaxy_transfer'][j2, :, :, :]
                jj = jj + 1
        return galaxy_ps

    def _calculate_matter_power(self):
        """ Returns matter_power in a 2-d numpy array of shape (nz, nk)."""
        Pk_interpolator = self.provider.get_Pk_interpolator(nonlinear=False)
        matter_power = np.array(Pk_interpolator(self.z_list, self.k_array))
        return matter_power

        # TODO: Check robustness of the interpolator (or whatever method adopted)
        # 1) use get_Pk_grid to cross-check results of interpolation
        #    Code: Pk = self.provider.get_Pk_grid(nonlinear=False)
        #    Pk_grid is a tuple of three arrays (z, k, Pk), where z, k are 1-d, and Pk is nz x nk.
        # 2) plot out P(k) interpolated and before interpolation
        # 3) look at linear vs nonlinear
        #    understand problem w/ diff = -0.5 for linear scales w/ the interpolator

    def _calculate_galaxy_transfer(self, **params_values_dict):
        bias = self._calculate_galaxy_bias(
            **params_values_dict)  # (n_sample, nz, nk)
        kaiser = self._calculate_rsd_kaiser(bias)  # (n_sample, nz, nk, mu)
        galaxy_transfer = \
            bias[:, :, :, np.newaxis] * kaiser * \
            self._calculate_nl_damping() * \
            self._calculate_rsd_nl()
        return galaxy_transfer

    def _calculate_AP_factor(self):
        # TODO to complete
        DA = self.provider.get_angular_diameter_distance(self.z_list)
        Hubble = self.provider.get_Hubble(self.z_list)
        # want Hubble and DA here from fiducial cosmology
        return np.ones(self.nz)

    def _calculate_rsd_kaiser(self, bias):
        """Returns the Kaiser factor in (n_sample, nz, nk, nmu) numpy array.

        Note: we return one factor of kaiser = (1+f(z)/b_j(k,z) mu^2).
        """
        f = self._calculate_growth_rate()  # shape (nz,) # time: 0.001 sec
        # self._calculate_growth_rate_approx()  # time:  1.71661376953125e-05 sec
        # TODO choose best method
        kaiser = (1.0 + f[:, np.newaxis] / bias)[:, :, :, np.newaxis]\
            * self.mu_array ** 2
        print('==>kaiser.shape', kaiser.shape)
        assert kaiser.shape == (self.n_sample, self.nz, self.nk, self.nmu)
        # TODO turn shape test into unit test
        # print('==>kaiser.shape', kaiser.shape)
        # assert kaiser.shape == (self.n_sample, self.nz, self.nk, self.nmu)
        # TODO turn into unit test
        # iz = 1
        # print('a[0,iz=%i,:]' % iz, a[iz, :])
        # print('expect:', f[iz] / bias[0, iz, :])
        return kaiser

    def _calculate_nl_damping(self):
        """Return non-linear damping factor in 3-d numpy array of shape (nz, nk, nmu).
        Note: This is factor for galaxy density (not power spectrum)
        so it is exp(-1/4 * arg), where arg = k^2 * [Sig_perp^2 + mu^2 * (Sig_para^2 - Sig_perp^2)]
        where Sig_perp(z) = c_rec * D(z) * Sig0 and Sig_para(z) = Sig_perp * (1+f(z)).
        """
        self._Sig0 = 11.0  # Mpc/h # TODO need to change unit
        Sig_perp = self._fraction_recon * self._Sig0 * self._calculate_growth()  # (nz,)
        # might be able to avoid calculating f(z) twice # TODO optimize later
        Sig_para = Sig_perp * (1.0 + self._calculate_growth_rate())  # (nz,)
        arg = self.k_array[np.newaxis, :, np.newaxis] \
            * ((Sig_perp**2)[:, np.newaxis, np.newaxis]
               + (self.mu_array**2)[np.newaxis, np.newaxis, :]
               * ((Sig_para ** 2)[:, np.newaxis, np.newaxis]
                   - (Sig_perp**2)[:, np.newaxis, np.newaxis])
               )
        # TODO add unit test for shape
        assert arg.shape == (self.nz, self.nk, self.nmu)
        ans = np.exp(-0.25 * arg)
        return ans

    def _calculate_growth(self):
        """Returns growth D(z) normalized to unity at z = 0 as a 1-d numpy array."""
        # TODO find a better way to implement D(z), or call another name
        # like sigma8_ratio or something to be clear.
        sigma8 = self._calculate_sigma8()
        sigma8_z0 = self.provider.get_param('sigma8')
        print('==>sigma8_z0', sigma8_z0)
        return sigma8 / sigma8_z0

    def _calculate_rsd_nl(self):  # TODO find a better name
        """Returns the blurring due to redshift error exp(-arg^2/2)
        where arg = (1+z) * sigz * k_parallel/H(z). """
        k_parallel = self.k_array[:, np.newaxis] \
            * self.mu_array  # (nk, nmu)
        Hubble = self.provider.get_Hubble(self.z_list)
        # fac = Hubble.reshape((self.nz, 1, 1)) / k_parallel
        k_parallel_over_H = k_parallel[np.newaxis, :, :] \
            / Hubble.reshape((self.nz, 1, 1))
        # TODO turn into unit test
        assert k_parallel_over_H.shape == (self.nz, self.nk, self.nmu)
        print('==>k_parallel_over_H.shape', k_parallel_over_H.shape)
        arg = np.array(self.sigz).reshape(self.n_sample, 1, 1, 1) \
            * (1.0 + self.z_array.reshape(1, self.nz, 1, 1)) \
            * k_parallel_over_H
        # TODO turn into unit test
        print('==>arg.shape', arg.shape)
        assert arg.shape == (self.n_sample, self.nz, self.nk, self.nmu)
        ans = np.exp(- arg ** 2 / 2.0)
        return ans

    def _calculate_galaxy_bias(self, **params_values_dict):
        """Returns galaxy bias in a 3-d numpy array of shape (n_sample, nz, nk)."""
        gaussian_bias_per_sample = self._calculate_gaussian_bias_array(
            **params_values_dict)
        gaussian_bias_per_sample = gaussian_bias_per_sample[:, np.newaxis]
        alpha = self._calculate_alpha()
        galaxy_bias = np.array([
            gaussian_bias_per_sample[j]
            + 2.0 * params_values_dict['fnl']
            * (gaussian_bias_per_sample[j] - 1.0) * self._delta_c / alpha
            for j in range(self.n_sample)
        ])
        print('==>galaxy_bias', galaxy_bias)
        assert galaxy_bias.shape == (self.n_sample, self.nz, self.nk)
        # TODO add unit test for galaxy_bias shape
        # TODO add test for galaxy_bias content (perhaps w/ a plot)

        return galaxy_bias

    def _calculate_gaussian_bias_array(self, **params_values_dict):
        """"Returns a 1-d numpy array of gaussian galaxy bias, for each galaxy sample ."""
        keys = ['gaussian_bias_%s' % (i) for i in range(1, self.n_sample + 1)]
        gaussian_bias = np.array([params_values_dict[key] for key in keys])
        print('==>gaussian_bias', gaussian_bias)
        # TODO add unit test check gaussian_bias.shape == (self.n_sample, )
        return gaussian_bias

    def _calculate_alpha(self):
        """Returns alpha as 2-d numpy array with shape (nz, nk) """
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
        initial_power = np.transpose(initial_power[:, np.newaxis])
        alpha = (5.0 / 3.0) * \
            np.sqrt(self._calculate_matter_power() /
                    self._calculate_initial_power())
        # TODO add unit test for alpha shape
        return alpha

    def _calculate_initial_power(self):
        # TODO want to make sure this corresponds to the same as camb module
        # Find out how to get it from camb itself (we might have other parameters
        # like nrun, nrunrun; and possibly customized initial power one day)
        """Returns 1-d numpy array of initial power spectrum evaluated at self.k_array."""
        k0 = self._k0  # 1/Mpc
        As = self.provider.get_param('As')
        ns = self.provider.get_param('ns')
        initial_power = (2.0 * np.pi**2) / (self.k_array**3) * \
            As * (self.k_array / k0)**(ns - 1.0)
        return initial_power

    def _calculate_growth_rate_approx(self):
        """Returns f(z) = Omega_m(z)^0.55 in a 1-d numpy array"""
        growth_rate_index = 0.55
        omegam = self.provider.get_param('omegam')
        f = (omegam * (1.0 + self.z_array) ** 3) ** growth_rate_index
        return f

    def _calculate_sigma8(self):
        # TODO checkout self.provider.get_sigma_R()
        # (https://cobaya.readthedocs.io/en/latest/theory_camb.html#theories.camb.camb.get_fsigma8)
        # same answer? (probably better to use self.provider.get_sigma_R() and no camb transfer)
        params, results = self.provider.get_CAMB_transfers()
        data = results.get_matter_transfer_data()
        return data.sigma_8

    def _calculate_growth_rate(self):
        """Returns f(z) from camb"""
        sigma8 = self._calculate_sigma8()
        fsigma8 = self.provider.get_fsigma8(self.z_list)
        print('==>fsigma8', fsigma8)
        f = fsigma8 / sigma8
        print('==>f (accurate)', f)
        return f

        # Common ingredients needed by bispectrum:
        # galaxy bias, alpha, rsd factor
        # TODO might want to think about how to interface, so only calculate once.
        # Do this later at the design doc level.
