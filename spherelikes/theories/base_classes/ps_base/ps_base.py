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

logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.ticker').disabled = True


def get_params(nsample, nz):
    base_params = {
        'fnl': {'prior': {'min': 0, 'max': 5},
                'ref': {'dist': 'norm', 'loc': 1.0, 'scale': 0.5},
                'propose': 0.001,
                'latex': 'f_{\rm{NL}}'
                },
        'derived_param': {'derived': True},
    }
    biases = {}
    for isample in range(nsample):
        for iz in range(nz):
            key = 'gaussian_bias_sample_%s_z_%s' % (isample + 1, iz + 1)
            value = {'prior': {'min': 0.8, 'max': 2.0},
                     'ref': {'dist': 'norm', 'loc': 1.1, 'scale': 0.2},
                     'propose': 0.001,
                     'latex': 'b_g^{%i}(z_{%i})' % (isample + 1, iz + 1)
                     }
            biases[key] = value
    params = {**base_params, **biases}
    return params


class PowerSpectrumBase(Theory):

    nsample = 5
    nz = 11
    params = get_params(nsample, nz)

    nk = 1  # 21  # 211  # number of k points (to be changed into bins)
    nmu = 1  # 5  # number of mu bins

    survey_pars_file_name = 'inputs/survey_pars_v28_base_cbe.yaml'
    data_dir = 'data/ps_base/'
    model_name = 'ref'
    plot_dir = 'plots/'

    is_reference_model = False
    do_test = False
    do_test_plot = False
    test_plot_names = None

    _delta_c = 1.686
    _k0 = 0.05  # 1/Mpc
    _fraction_recon = 0.  # 0 for fully damped, no recon; 1 for fully reconstructed, no damping

    def initialize(self):
        """called from __init__ to initialize"""
        self.logger = class_logger(self)
        self._setup_survey_pars()
        self._setup_results_fid()
        self._setup_tests()
        print('Done setting up PowerSpectrumBase')

    def initialize_with_provider(self, provider):
        """
        Initialization after other components initialized, using Provider class
        instance which is used to return any dependencies (see calculate below).
        """
        self.provider = provider

    def get_dk(self, k):
        """Return dk array.

        Note: This assumes that k is uniformly spaced in log space. 
        It computes dk by taking the difference in linear space between the
        log bin-centers, assuming that the first and last bin are only half 
        the usual log bin size."""

        logk = np.log(k)
        logk_mid = (logk[:-1] + logk[1:]) / 2
        dk = np.zeros(k.size)
        dk[1:-1] = np.exp(logk_mid)[1:] - np.exp(logk_mid)[0:-1]
        dk[0] = np.exp(logk_mid[0]) - np.exp(logk[0])
        dk[-1] = np.exp(logk[-1]) - np.exp(logk_mid[-1])
        return dk

    def _setup_survey_pars(self):
        """kmin and kmax preserved, dk calcula"""
        path = os.path.join(self.survey_pars_file_name)
        pars = yaml_load_file(path)
        self.z_lo = np.array(pars['zbin_lo'])
        self.z_hi = np.array(pars['zbin_hi'])
        self.z = (self.z_lo + self.z_hi) / 2.0
        self.z_list = list(self.z)
        self.sigz = np.array(pars['sigz_over_one_plus_z'])[:, np.newaxis] \
            * (1.0 + self.z[np.newaxis, :])
        self.nsample = len(pars['sigz_over_one_plus_z'])
        self.nz = self.z.size
        self.k = np.logspace(np.log10(1e-5), np.log10(5.0), self.nk)
        self.dk = self.get_dk(self.k)
        self.mu_edges = np.linspace(0, 1, self.nmu + 1)
        self.mu = (self.mu_edges[:-1] + self.mu_edges[1:]) / 2.0
        self.dmu = self.mu_edges[1:] - self.mu_edges[:-1]
        assert self.mu.size == self.dmu.size, ('mu and dmu not the same size: {}, {}'.format(
            self.mu.size, self.dmu.size))

    def get_requirements(self):
        """
        Return dictionary of derived parameters or other quantities that are needed
        by this component and should be calculated by another theory class.
        """
        return {}

    def must_provide(self, **requirements):
        z_list = self.z_list
        z_more = np.hstack((self.z_lo, self.z_hi, self.z))
        k_max = 8.0
        nonlinear = (False, True)
        spec_Pk = {
            'z': z_list,
            'k_max': k_max,  # 1/Mpc
            'nonlinear': nonlinear,
        }
        if 'galaxy_ps' in requirements:
            return {
                # 'Pk_grid': spec_Pk,
                'Pk_interpolator': spec_Pk,
                'Cl': {'tt': 2500},
                'H0': None,
                'angular_diameter_distance': {'z': z_list},
                'Hubble': {'z': z_list},
                'comoving_radial_distance': {'z': z_more},
                'omegam': None,
                'As': None,
                'ns': None,
                'fsigma8': {'z': z_list},
                'sigma8': None,
                'CAMBdata': None,
            }
        if 'galaxy_transfer' in requirements:
            return{
                # 'Pk_grid': spec_Pk,
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
            }
        if 'AP_factor' in requirements:
            return{
                'angular_diameter_distance': {'z': z_list},
                'Hubble': {'z': z_list},
            }

    def get_can_provide_params(self):
        return ['derived_param']

    def calculate(self, state, want_derived=True, **params_values_dict):

        nonlinear = False

        self.logger.debug('Calculating k and mu actual using AP factors.')
        self.k_actual_perp, self.k_actual_para, self.k_actual = self._calc_k_actual()
        self.mu_actual = self._calc_mu_actual()

        print('Calculating matter power')
        state['matter_power'] = self._calc_matter_power(
            nonlinear=nonlinear)

        print('Calculating galaxy transfer')
        state['galaxy_transfer'] = self._calc_galaxy_transfer(
            **params_values_dict)

        print('Calculating AP factor')
        state['AP_factor'] = self._calc_AP_factor()

        print('Calculating galaxy_ps')
        state['galaxy_ps'] = self._calc_galaxy_ps(state)

        # TODO placeholder for any derived paramter from this module
        state['derived'] = {'derived_param': 1.0}

        print('Running tests')
        self.run_tests(params_values_dict, state)

    def get_galaxy_ps(self):
        return self._current_state['galaxy_ps']

    def get_galaxy_transfer(self):
        return self._current_state['galaxy_transfer']

    def get_AP_factor(self):
        return self._current_state['AP_factor']

    def _calc_k_actual(self):
        """Returns a tuple of three 3-d numpy arrays of shape (nz, nk, nmu) for
        (k_perp, k_para, k) using AP factors:
            k_perp = k_perp|ref * D_A(z)|ref / D_A(z),
            k_para = k_para|ref * (1/H(z))|ref / (1/H(z)),
        where
            k// = mu * k,
            kperp = sqrt(k^2  - k//^2) = k sqrt(1 - mu^2).
        """
        k_perp_ref = self.k[:, np.newaxis] * \
            np.sqrt(1. - (self.mu**2)[np.newaxis, :])
        k_para_ref = self.k[:, np.newaxis] * self.mu[np.newaxis, :]

        ap_perp, ap_para = self._calc_AP_factors_perp_and_para()

        k_perp = k_perp_ref[np.newaxis, :, :] * \
            ap_perp[:, np.newaxis, np.newaxis]
        k_para = k_para_ref[np.newaxis, :, :] * \
            ap_para[:, np.newaxis, np.newaxis]

        k = np.sqrt(k_perp ** 2 + k_para ** 2)

        return k_perp, k_para, k

    def _calc_mu_actual(self):
        """Returns a 3d numpy array of shape (nz, nk, nmu) for the actual mu.
        """
        return self.k_actual_para / self.k_actual

    def _setup_results_fid(self):
        self.logger.info('is_reference_model = {}'.format(
            self.is_reference_model))
        if self.is_reference_model is True:
            self.logger.info(
                '... calculating instead of loading fiducial results.')
        else:
            self._make_path_fid()
            self._load_results_fid()

    def _setup_tests(self):
        self._setup_plot_dir()

    def _setup_plot_dir(self):
        pathlib.Path(
            self.plot_dir).mkdir(parents=True, exist_ok=True)

    def _make_path_fid(self):
        self.fname_fid = os.path.join(
            self.data_dir, self.model_name + '.pickle')

    def _load_results_fid(self):
        try:
            self.results_fid = pickle.load(open(self.fname_fid, "rb"))
        except FileNotFoundError as e:
            raise LoggedError(
                self.logger,
                '%s. \n' % e
                + 'Reference results needed by AP effects do not exist! '
                + 'You must run generate_ref.py first to make reference model.')

    def _get_var_fid(self, name_of_variable):
        # TODO want to document structure of results later
        if self.is_reference_model is True:
            raise NotImplementedError
        elif name_of_variable not in ['Hubble', 'angular_diameter_distance']:
            msg = "_get_var_fid: name_of_variable can only be: Hubble, angular_diameter_distance."
            raise ValueError(msg)
        else:
            try:
                self._check_z_fid()
            except ValueError as e:
                raise LoggedError(self.logger, e)
            else:
                var = self.results_fid[name_of_variable]
                msg = '{}.size = {}, expected {}'.format(
                    name_of_variable, var.size, self.results_fid['aux']['z'].size)
                assert var.size == self.results_fid['aux']['z'].size, (msg)
                # TODO write unit test to check against runing get_fid_model.py
                return var

    def _check_z_fid(self):
        z_fid = self.results_fid['aux']['z']
        is_same = np.all(self.z == z_fid)
        if not is_same:
            raise ValueError(
                'fiducial cosmology results (Hubble or angular diameter distance) \
                    do not have the same redshift values: z = {} vs z_fid = {}.'
                .format(self.z, z_fid))

    def _calc_galaxy_ps(self, state):
        """Returns 4-d numpy array of shape (n_ps, nz, nk, nmu) for the galaxy power spectra.

        Note: n_ps is the number of unique auto- and cross- power spectra bewteen nsample of
        galaxy samples:
            n_ps = nsample * (nsample + 1) / 2.
        """
        self.nps = int(self.nsample * (self.nsample + 1) / 2)
        galaxy_ps = np.zeros((self.nps, self.nz, self.nk, self.nmu))
        jj = 0
        for j1 in range(self.nsample):
            for j2 in range(j1, self.nsample):
                galaxy_ps[jj] = state['AP_factor'][:, np.newaxis, np.newaxis] \
                    * state['matter_power'] \
                    * state['galaxy_transfer'][j1, :, :, :] \
                    * state['galaxy_transfer'][j2, :, :, :]
                jj = jj + 1
        assert jj == self.nps
        return galaxy_ps

    def _calc_matter_power(self, nonlinear=False):
        """ Returns 3-d numpy array of shape (nz, nk, nmu) for the matter power spectrum.
        Default is linear matter power."""

        Pk_interpolator = self.provider.get_Pk_interpolator(
            nonlinear=nonlinear)

        matter_power = np.zeros_like(self.k_actual)
        # for every fixed z and fixed mu, there is an array of k scaled by AP factors.
        for iz in range(self.nz):
            for imu in range(self.nmu):
                a = Pk_interpolator(self.z_list[iz], np.log(
                    self.k_actual[iz, :, imu]))
                matter_power[iz, :, imu] = np.exp(a)
        assert matter_power.shape == (self.nz, self.nk, self.nmu)
        return matter_power

    def _calc_matter_power_from_grid(self):
        """ Returns 2-d numpy array of shape (nz, nk) for the matter power spectrum."""
        Pk_grid = self.provider.get_Pk_grid(
            nonlinear=False)  # returns (k, z, Pk)
       # self.k = Pk_grid[0]  # update k and nk
       # self.nk = self.k.size
        assert np.all(self.z == Pk_grid[1]), (self.z, Pk_grid[1])
        matter_power = Pk_grid[2]  # 2-d numpy array of shape (nz, nk)
        # add test to check
        return matter_power

    def _calc_galaxy_transfer(self, **params_values_dict):
        """Returns 4-d numpy array of shape (nsample, nz, nk, mu) for the galaxy transfer functions."""

        # TODO need to apply nl damping to wiggles and not to broadband

        bias = self._calc_galaxy_bias(**params_values_dict)
        print('bias.shape', bias.shape)
        print('Calculating kaiser')
        kaiser = self._calc_rsd_kaiser(bias)

        print('Calculating fog')
        fog = self._calc_fog()

        galaxy_transfer = bias * kaiser * fog
        return galaxy_transfer

    def _calc_AP_factors_perp_and_para(self):
        """Returns a tuple with two 1d numpy arrays: (DA(z)|fid / DA(z), H(z)/H(z)|fid).
        """
        if self.is_reference_model is True:
            ap_perp = np.ones(self.nz)
            ap_para = np.ones(self.nz)
        else:
            DA = self.provider.get_angular_diameter_distance(self.z_list)
            Hubble = self.provider.get_Hubble(self.z_list)
            Hubble_fid = self._get_var_fid('Hubble')
            DA_fid = self._get_var_fid('angular_diameter_distance')
            ap_perp = DA_fid / DA
            ap_para = Hubble / Hubble_fid
        return (ap_perp, ap_para)

    def _calc_AP_factor(self):
        """Returns 1-d numpy array of shape (nz,) for the Alcock-Pacynzski factors."""
        ap_perp, ap_para = self._calc_AP_factors_perp_and_para()
        ap = ap_perp ** 2 * (ap_para)
        return ap

    def _calc_rsd_kaiser(self, bias):
        """Returns a 4-d numpy array of shape (nsample, nz, nk, nmu) for RSD Kaiser factor.

        Note: we return one factor of
            kaiser = (1 + f(z)/b_j(k,z) * mu^2)
        for the galaxy density, not power spectrum.
        """
        f = self._calc_growth_rate()
        kaiser = 1.0 + f[np.newaxis, :, np.newaxis, np.newaxis] / bias \
            * (self.mu_actual ** 2)[np.newaxis, :, :, :]
        assert kaiser.shape == (self.nsample, self.nz, self.nk, self.nmu)
        return kaiser

    def _calc_nl_damping(self):
        """Return 3-d numpy array of shape (nz, nk, nmu) for the non-linear damping.

        Note: This is the factor for the galaxy density (not power spectrum), so
            ans = exp(-1/4 * arg),
        where
            arg = k^2 * [Sig_perp^2 + mu^2 * (Sig_para^2 - Sig_perp^2)]
        and
            Sig_perp(z) = c_rec * D(z) * Sig0 and Sig_para(z) = Sig_perp * (1+f(z)).
        """

        # 11 Mpc/h for sigma8 = 0.8 at present day (assume h = 0.68)
        self._Sig0 = 11.0 / 0.68

        sig0_scaled = self._Sig0 / 0.8 * self._calc_sigma8()

        Sig_perp = (1.0 - self._fraction_recon) * sig0_scaled
        Sig_para = Sig_perp * (1.0 + self._calc_growth_rate())

        arg = self.k[np.newaxis, :, np.newaxis] \
            * ((Sig_perp**2)[:, np.newaxis, np.newaxis]
               + (self.mu**2)[np.newaxis, np.newaxis, :]
               * ((Sig_para ** 2)[:, np.newaxis, np.newaxis]
                   - (Sig_perp**2)[:, np.newaxis, np.newaxis])
               )
        assert arg.shape == (self.nz, self.nk, self.nmu)
        ans = np.exp(-0.25 * arg)

        return ans

    # TODO not completed - might delete
    def _calc_growth(self, norm_kind='z=0'):
        """Returns a 1-d numpy array for growth D(z) normalized to unity at z = 0 by default.

        Note: use norm_kind='matter_domination' to get D(z) normalized to 1/(1+z) at matter domination."""

        _, results = self.provider.get_CAMB_transfers()
        data = results.get_matter_transfer_data()
        # normalized to 1 at low-k; T(k)/k^2; should be cst at low-k
        k = data.transfer_z('k/h')
        k_fixed = 0.1
        ind = np.max(np.where(k < k_fixed)[0])

        T0 = data.transfer_z('delta_tot')[ind]
        T_list = np.array([data.transfer_z('delta_tot', z_index=z_index)[ind]
                           for z_index in range(self.nz)])
        D = T_list / T0
        if norm_kind == 'matter_domination':
            wanted_D_high_z = 1.0 / (1 + self.z_list[-1])
            current_D_high_z = D[-1]
            D = D / current_D_high_z * wanted_D_high_z
        return D

    def _calc_fog(self):  # TODO find a better name
        """Returns 4-d numpy array of shape (nsample, nz, nk, nmu) for the blurring
        due to redshift error exp(-arg^2/2) where arg = (1+z) * sigz * k_parallel * c /H(z). """

        k_parallel = self.k[:, np.newaxis] * \
            self.mu[np.newaxis, :]  # this would have been reference k w/ new hubble

        Hubble = self.provider.get_Hubble(self.z_list)
        arg = self.sigz[:, :, np.newaxis, np.newaxis] \
            * (1.0 + self.z[np.newaxis, :, np.newaxis, np.newaxis]) \
            * self.k_actual_para[np.newaxis, :, :, :] \
            * constants.c_in_km_per_sec  \
            / Hubble[np.newaxis, :, np.newaxis, np.newaxis]

        if (self.do_test) and (not self.is_reference_model) == True:
            Hubble_ref = self._get_var_fid('Hubble')
            arg_ref = self.sigz[self.nsample, self.nz, np.newaxis, np.newaxis] \
                * (1.0 + self.z[np.newaxis, self.nz, np.newaxis, np.newaxis]) \
                * k_parallel[np.newaxis, np.newaxis, :, :]\
                * constants.c_in_km_per_sec  \
                / Hubble_ref[np.newaxis, :, np.newaxis, np.newaxis]
            assert np.allclose(arg_ref, arg)
        # TODO make unit test
        # expect suppression at k ~ 0.076 1/Mpc for H = 75km/s/Mpc at z = 0.25 w/ sigma_z = 0.003
        assert arg.shape == (self.nsample, self.nz, self.nk, self.nmu)

        ans = np.exp(- arg ** 2 / 2.0)
        return ans

    def _calc_galaxy_bias(self, **params_values_dict):
        """Returns galaxy bias in a 4-d numpy array of shape (nsample, nz, nk, nmu)."""

        gaussian_bias = self._calc_gaussian_bias_array(
            **params_values_dict)[:, :, np.newaxis, np.newaxis]

        alpha = self._calc_alpha()[np.newaxis, :, :, :]

        galaxy_bias = gaussian_bias \
            + 2.0 * params_values_dict['fnl'] * self._delta_c\
            * (gaussian_bias - 1.0) / alpha

        expected_shape = (self.nsample, self.nz, self.nk, self.nmu)
        msg = ('galaxy_bias.shape = {}, expected ({})'
               .format(galaxy_bias.shape, expected_shape))
        assert galaxy_bias.shape == expected_shape, msg

        return galaxy_bias

    def _calc_gaussian_bias_array(self, **params_values_dict):
        """"Returns a 1-d numpy array of shape (nsample, 1) for gaussian galaxy bias,
        one for each galaxy sample.
        """
        gaussian_bias = np.zeros((self.nsample, self.nz))
        for isample in range(self.nsample):
            for iz in range(self.nz):
                key = 'gaussian_bias_sample_%s_z_%s' % (isample + 1, iz + 1)
                gaussian_bias[isample, iz] = params_values_dict[key]
        return gaussian_bias

    def _calc_alpha(self):
        """Returns alpha as 3-d numpy array with shape (nz, nk, nmu) """
        initial_power = self._calc_initial_power()
        alpha = (5.0 / 3.0) \
            * np.sqrt(self._calc_matter_power() / initial_power)
        assert alpha.shape == (self.nz, self.nk, self.nmu)
        return alpha

    def _calc_initial_power(self):
        # TODO want to make sure this corresponds to the same as camb module
        # Find out how to get it from camb itself (we might have other parameters
        # like nrun, nrunrun; and possibly customized initial power one day)
        """Returns 3-d numpy array of shape (nz, nk, nmu) of initial power spectrum
        evaluated at self.k_actual. 

        Note: There is a z and mu dependence because the k_actual at which we 
        need to evaluate the initial power is different for different z and mu.
        """
        k0 = self._k0  # 1/Mpc
        As = self.provider.get_param('As')
        ns = self.provider.get_param('ns')
        initial_power = (2.0 * np.pi**2) / (self.k_actual**3) * \
            As * (self.k_actual / k0)**(ns - 1.0)
        return initial_power

    def _calc_sigma8(self):
        sigma8 = np.flip(
            self.provider.get_CAMBdata().get_sigma8()
        )
        return sigma8

    def _calc_growth_rate(self):
        """Returns f(z) from camb"""
        sigma8 = self._calc_sigma8()
        fsigma8 = self.provider.get_fsigma8(self.z_list)
        f = fsigma8 / sigma8
        return f

    def run_tests(self, params_values_dict, state):
        if self.do_test is True:
            # add more tests here if needed
            if self.do_test_plot is True:
                self._make_plots(params_values_dict, state,
                                 names=self.test_plot_names)

    def _make_plots(self, params_values_dict, state, names=None):
        if names is None:
            names = [
                'Hubble',
                'angular_diameter_distance',
                'matter_power',
                # 'matter_power_from_grid',
                # 'galaxy_ps',
                'gaussian_bias',
                'alpha',
                'bias',
                'growth_rate',
                'sigma8',
                'galaxy_transfer_components',
                'galaxy_transfer',
            ]
        plotter = Plotter(self, params_values_dict, state)
        plotter.make_plots(names)


class Plotter():

    def __init__(self, theory, params_values_dict, state):
        self.logger = class_logger(self)
        self.theory = theory
        self.z = self.theory.z
        self.k = self.theory.k
        self.mu = self.theory.mu
        self.sigz = self.theory.sigz
        self.nz = self.theory.nz
        self.nk = self.theory.nk
        self.nmu = self.theory.nmu
        self.nsample = self.theory.nsample
        self.nps = self.theory.nps
        self.params_values_dict = params_values_dict
        self.galaxy_ps = state['galaxy_ps']

    def make_plots(self, names):
        for name in names:
            function_name = 'plot_' + name
            function = getattr(self, function_name, None)
            if callable(function):
                self.logger.info('Calling function {}'.format(function_name))
                function()
            else:
                msg = "Function {} is not implemented!".format(function_name)
                raise NotImplementedError(msg)

    def plot_angular_diameter_distance(self):
        y1 = self.theory.provider.get_angular_diameter_distance(self.z)
        y_list = [y1]
        if self.theory.is_reference_model is False:
            y2 = self.theory._get_var_fid('angular_diameter_distance')
            y_list.append(y2)
        self.plot_1D('z', y_list, '$D_A(z)$', 'angular_diameter_distance',
                     legend=['current model', 'fid. model'])

    def plot_Hubble(self):
        y1 = self.theory.provider.get_Hubble(self.z)
        y_list = [y1]
        if self.theory.is_reference_model is False:
            y2 = self.theory._get_var_fid('Hubble')
            y_list.append(y2)
        self.plot_1D('z', y_list, '$H(z)$', 'Hubble',
                     legend=['current model', 'fid. model'])

    def plot_alpha(self):
        data = self.theory._calc_alpha()
        axis_names = ['z', 'k']
        ylatex = '$\\alpha(k)$'
        yname = 'alpha'
        plot_type = 'loglog'
        self.plot_2D(axis_names, data, ylatex, yname, plot_type=plot_type)

    def plot_growth(self):
        y_list = [self.theory._calc_growth()]
        axis_name = 'a'
        ylatex = '$D(a)$'
        yname = 'growth'
        plot_type = 'plot'
        self.plot_1D(axis_name, y_list, ylatex, yname,
                     plot_type=plot_type)

    def plot_sigma8(self):
        y_list = [self.theory._calc_sigma8()]
        axis_name = 'z'
        ylatex = '$\sigma_8(z)$'
        yname = 'sigma8'
        plot_type = 'plot'
        self.plot_1D(axis_name, y_list, ylatex, yname,
                     plot_type=plot_type)

    def plot_matter_power(self):

        data_nl = self.theory._calc_matter_power(nonlinear=True)
        data_lin = self.theory._calc_matter_power(nonlinear=False)

        axis_names = ['z', 'k']
        plot_type = 'loglog'

        yname = 'matter_power_nonlinear'
        ylatex = '$P_m^{\mathrm{NL}}(z,k)$ [Mpc$^3$]'
        self.plot_2D(axis_names, data_nl, ylatex, yname, plot_type=plot_type)

        yname = 'matter_power_linear'
        ylatex = '$P_m^{\mathrm{lin}}(z,k)$ [Mpc$^3$]'
        self.plot_2D(axis_names, data_lin, ylatex, yname, plot_type=plot_type)

    def plot_matter_power_from_grid(self):
        data = self.theory._calc_matter_power_from_grid()
        axis_names = ['z', 'k']
        ylatex = '$P_m(z,k)$ from grid'
        yname = 'matter_power_from_grid'
        plot_type = 'loglog'
        self.plot_2D(axis_names, data, ylatex, yname, plot_type=plot_type)

    def plot_galaxy_transfer(self):
        data = self.theory._calc_galaxy_transfer(
            **self.params_values_dict)
        axis_names = ['sample', 'z', 'k', 'mu']
        yname = 'galaxy_transfer'
        plot_type = 'loglog'

        isamples = self._get_indices(self.nsample, n_wanted=2)
        izs = self._get_indices(self.nz, n_wanted=2)
        for isample in isamples:
            for iz in izs:
                axis_names_in = ['k', 'mu']
                yname_in = yname + '_isample%s_iz%s' % (isample, iz)
                data_in = data[isample, iz, :, :]
                ylatex = 'galaxy transfer function $T_g(k, \mu)$'
                self.plot_2D(axis_names_in, data_in, ylatex,
                             yname_in, plot_type=plot_type)

    def plot_gaussian_bias(self):
        y_list = [self.theory._calc_gaussian_bias_array(
            **self.params_values_dict)]
        axis_name = 'sample'
        ylatex = '$b_{Gauss}$'
        yname = 'gaussian_bias'
        plot_type = 'loglog'
        self.plot_1D(axis_name, y_list, ylatex, yname,
                     plot_type=plot_type)

    def plot_bias(self):
        isample = 0
        y_list = self.theory._calc_galaxy_bias(
            **self.params_values_dict)[isample, :, :]
        axis_name = ['z', 'k']
        ylatex = '$b_g(z, k)$'
        yname = 'bias_isample%s' % isample
        plot_type = 'loglog'
        self.plot_2D(axis_name, y_list, ylatex, yname,
                     plot_type=plot_type)

    def plot_growth_rate(self):
        isample = 0
        y_list = [self.theory._calc_growth_rate()]
        axis_name = 'z'
        ylatex = '$f(z)$'
        yname = 'growth_rate'
        plot_type = 'plot'
        ylim = [0.5, 1]
        self.plot_1D(axis_name, y_list, ylatex, yname,
                     plot_type=plot_type, ylim=ylim)

    def plot_galaxy_transfer_components(self):
        matter_power_lin = self.theory._calc_matter_power(
            nonlinear=False)[np.newaxis, :, :, :]
        bias = self.theory._calc_galaxy_bias(**self.params_values_dict)
        kaiser = self.theory._calc_rsd_kaiser(
            bias)  # (nsample, nz, nk, mu)
        nl_damping = self.theory._calc_nl_damping()[np.newaxis, :, :, :]
        fog = self.theory._calc_fog()

        axis_names = ['ps', 'z', 'k', 'mu']
        yname = 'galaxy_ps'
        plot_type = 'loglog'

        galaxy_ps_1 = matter_power_lin * bias**2
        galaxy_ps_2 = galaxy_ps_1 * kaiser**2
        galaxy_ps_3 = galaxy_ps_2
        galaxy_ps_4 = galaxy_ps_3 * fog**2
        galaxy_ps_steps = [galaxy_ps_1, galaxy_ps_2, galaxy_ps_3, galaxy_ps_4]

        dict_steps = {0: "$P_m(k) b^2$",
                      1: "$P_m(k)$ (b * kaiser)$^2$",
                      2: "$P_m(k)$ (b * kaiser)$^2$ [need to implement damping]",
                      3: "$P_m(k)$ (b * kaiser * fog)$^2$",
                      }
        # ips_list = self._get_indices(self.nps, n_wanted=2)
        # izs = self._get_indices(self.nz, n_wanted=2)
        ips_list = [0]
        izs = [0]
        for istep in range(4):
            for ips in ips_list:
                for iz in izs:
                    axis_names_in = ['k', 'mu']
                    yname_in = yname + '_istep%s_ips%s_iz%s' % (istep, ips, iz)
                    self.logger.debug('istep = {}'.format(istep))
                    data_in = galaxy_ps_steps[istep][ips, iz, :, :]
                    ylatex = 'Making $P_g(k, \mu)$ step %s: %s' % (
                        istep, dict_steps[istep])

                    if istep == 3:
                        ylim = [1e-5, 1e10]
                    else:
                        ylim = None

                    if istep == 0:
                        self.plot_1D('k', [data_in[:, 0]], ylatex,
                                     yname_in, plot_type=plot_type, ylim=ylim)
                    else:
                        self.plot_2D(axis_names_in, data_in, ylatex,
                                     yname_in, plot_type=plot_type, ylim=ylim)

        components = [kaiser, fog]
        ynames = ['kaiser', 'fog']
        for (component, yname) in zip(components, ynames):
            for ips in ips_list:
                for iz in izs:
                    axis_names_in = ['k', 'mu']
                    yname_in = yname + '_ips%s_iz%s' % (ips, iz)
                    data_in = component[ips, iz, :, :]
                    ylatex = '%s' % (yname)
                    if yname is 'fog':
                        ylim = [1e-10, 10]
                    else:
                        ylim = None
                    self.plot_2D(axis_names_in, data_in, ylatex,
                                 yname_in, plot_type=plot_type, ylim=ylim)

                    # Print out expected suppression at a k close to 0.2 to compare with plot by eye
                    if self.theory.is_reference_model is False:  # test not implemented when running fiducial model
                        if (ips == 0):
                            isample = 0
                            self.predict_fog(isample, iz)

    def predict_fog(self, isample, iz):
        idx = np.min(np.where(self.k >= 0.2)[0])
        k_ref = self.k[idx]
        mu_ref = self.mu[3]
        Hubble_ref = self.theory._get_var_fid('Hubble')
        arg = self.sigz[isample, iz] * \
            (1.0 + self.z[iz]) * k_ref * mu_ref * \
            constants.c_in_km_per_sec / Hubble_ref[iz]
        fog = np.sqrt(np.exp(-arg**2))
        self.logger.info('expect fog = {} for k_ref = {}, mu_ref = {}, iz = {}'.format(
            fog, k_ref, mu_ref, iz))

    def plot_galaxy_ps(self):
        data = self.galaxy_ps
        axis_names = ['ps', 'z', 'k', 'mu']
        yname = 'galaxy_ps'
        plot_type = 'loglog'

        ips_list = self._get_indices(self.nps, n_wanted=2)
        izs = self._get_indices(self.nz, n_wanted=2)
        for ips in ips_list:
            for iz in izs:
                axis_names_in = ['k', 'mu']
                yname_in = yname + '_ips%s_iz%s' % (ips, iz)
                data_in = data[ips, iz, :, :]
                ylatex = 'galaxy power spectrum $P_g(k, \mu)$'
                self.plot_2D(axis_names_in, data_in, ylatex,
                             yname_in, plot_type=plot_type)

    def plot_2D(self, *args, **kwargs):
        self.make_plot_2D_fixed_axis('col', * args, **kwargs)
        self.make_plot_2D_fixed_axis('row', * args, **kwargs)

    def make_plot_2D_fixed_axis(self, fixed_axis_name, axis_names, data, ylatex, yname,
                                plot_type='plot', k=None, z=None, ylim=None):
        """
        Make 1D plot of selected rows of the input 2-d numpy array data.
        If data has more than 5 rows, 5 indices equally spaced are selected.

        [More doc needed here on input args.]
        """

        FIXED, VARIED = self._get_fixed_and_varied_axes(fixed_axis_name)

        shape = data.shape
        indices = self._get_indices(shape[FIXED], n_wanted=5)

        if fixed_axis_name == 'row':
            y_list = [data[i, :] for i in indices]
        elif fixed_axis_name == 'col':
            y_list = [data[:, i] for i in indices]

        x = self.get_xarray(axis_names[FIXED], k=k, z=z)
        legend = ['%s = %.2e' % (axis_names[FIXED], x[i])
                  for i in indices]

        self.plot_1D(axis_names[VARIED], y_list, ylatex, yname,
                     legend=legend, plot_type=plot_type, k=k, z=z, ylim=ylim)

    def _get_fixed_and_varied_axes(self, fixed_axis):
        if fixed_axis == 'row':
            FIXED = 0
            VARIED = 1
        elif fixed_axis == 'col':
            FIXED = 1
            VARIED = 0
        else:
            raise ValueError(
                "_get_fixed_and_varied_axes: fixed_axis can only be: row or col.")
        return FIXED, VARIED

    def _get_indices(self, n_tot, n_wanted=5):
        delta = 1 if n_tot <= n_wanted else int(np.floor(
            n_tot / np.float(n_wanted)))
        indices = np.arange(0, n_tot, delta)
        self.logger.debug('delta = {}'.format(delta))
        self.logger.debug('indices = {}'.format(indices))
        return indices

    def get_xarray(self, dim, k=None, z=None):
        if dim == 'z':
            x = self.z if z is None else z
        elif dim == 'k':
            x = self.k if k is None else k
        elif dim == 'mu':
            x = self.mu
        elif dim == 'sample':
            x = np.arange(self.nsample)
        elif dim == 'a':
            x = self.z if z is None else z
            x = 1.0 / (1.0 + x)
        else:
            msg = "get_xarray can only take input values for dim: 'z', 'k', 'mu', 'sample' or 'a'."
            raise ValueError(msg)
        return x

    def get_xlabel(self, dim):
        if dim == 'z':
            xlabel = '$z$'
        elif dim == 'k':
            xlabel = '$k$ [1/Mpc]'
        elif dim == 'mu':
            xlabel = '$\mu$'
        elif dim == 'sample':
            xlabel = 'galaxy sample number'
        elif dim == 'a':
            xlabel = '$a$'
        else:
            msg = "get_xarray can only take input values for dim: 'z', 'k', 'mu', 'sample' or 'a'."
            raise ValueError(msg)
        return xlabel

    def plot_1D(self, dimension, y_list, ylatex, yname,
                legend='', plot_type='plot', k=None, z=None,
                ylim=None, xlim=None):

        allowed_plot_types = ['plot', 'loglog', 'semilogx', 'semilogy']

        x = self.get_xarray(dimension, k=k, z=z)
        xlabel = self.get_xlabel(dimension)

        fig, ax = plt.subplots()

        for y in y_list:
            if plot_type in allowed_plot_types:
                getattr(ax, plot_type)(x, y)
            else:
                msg = "plot_type can only be one of the following: {}".format(
                    allowed_plot_types)
                raise ValueError(msg)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylatex)
        if ylim is not None:
            ax.set_ylim(ylim)
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.legend(legend)

        plot_name = 'plot_%s_vs_%s.png' % (yname, dimension)
        plot_name = os.path.join(self.theory.plot_dir, plot_name)
        fig.savefig(plot_name)
        self.logger.info('Saved plot = {}'.format(plot_name))
        plt.close()
