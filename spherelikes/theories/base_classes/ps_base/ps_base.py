from cobaya.theory import Theory
import numpy as np
import time
import sys
import os
import pickle
import matplotlib.pyplot as plt
import pathlib
import logging
from spherelikes import paths

logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.ticker').disabled = True


class PowerSpectrumBase(Theory):

    params = {
        'fnl': {'prior': {'min': 0, 'max': 5}, 'propose': 0.1, 'ref': 1.0},
        'gaussian_bias_1': {'prior': {'min': 0.8, 'max': 2.0}, 'propose': 0.1, 'ref': 1.0, 'latex': 'b_g^{1}'},
        'gaussian_bias_2': {'prior': {'min': 0.8, 'max': 2.0}, 'propose': 0.1, 'ref': 1.0, 'latex': 'b_g^{2}'},
        # TODO how to make this variable number of bins?
        'derived_param': {'derived': True}
    }

    nsample = 2  # number of galaxy samples
    nz = 4  # number of z bins
    nk = 211  # number of k points (to be changed into bins)
    nmu = 10  # number of mu bins
    z_list = [0.25, 0.5, 0.75, 1]  # TODO to change to SPHEREx numbers
    sigz = [0.01, 0.01]  # TODO to change to SPHEREx numbers
    z = np.array(z_list)
    k = np.logspace(np.log10(1e-5), np.log10(5.0), nk)
    mu_edges = np.linspace(0, 1, nmu + 1)
    mu = (mu_edges[:-1] + mu_edges[1:]) / 2.0

    model_dir = paths.model_dir
    test_dir = paths.test_dir
    model_name = 'model_test'
    is_fiducial_model = False
    do_test = False
    do_test_plot = False
    test_plot_names = None

    _delta_c = 1.686
    _k0 = 0.05  # 1/Mpc
    _fraction_recon = 0.5  # reconstruction fration

    def initialize(self):
        """called from __init__ to initialize"""
        self._setup_results_fid()
        self._setup_tests()

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
        state['matter_power'] = self._calculate_matter_power(
            nonlinear=nonlinear)
        state['galaxy_transfer'] = self._calculate_galaxy_transfer(
            **params_values_dict)
        state['AP_factor'] = self._calculate_AP_factor()
        state['galaxy_ps'] = self._calculate_galaxy_ps(state)
        # TODO placeholder for any derived paramter from this module
        state['derived'] = {'derived_param': 1.0}

        self._run_tests(params_values_dict)

    def get_galaxy_ps(self):
        return self._current_state['galaxy_ps']

    def get_galaxy_transfer(self):
        return self._current_state['galaxy_transfer']

    def get_AP_factor(self):
        return self._current_state['AP_factor']

    def _setup_results_fid(self):
        print('is_fiducial_model = ', self.is_fiducial_model)
        if self.is_fiducial_model is True:
            print('Skipping loading of fiducial results since we are calculating it.')
        else:
            self._make_path_fid()
            self._load_results_fid()

    def _setup_tests(self):  # TODO might not need this function
        self._setup_test_dir()

    def _setup_test_dir(self):
        pathlib.Path(
            self.test_dir).mkdir(parents=True, exist_ok=True)

    def _make_path_fid(self):
        self.model_dir = os.path.join(self.model_dir, self.model_name)
        self.fname_fid = os.path.join(
            self.model_dir, self.model_name + '.pickle')

    def _load_results_fid(self):
        self.results = pickle.load(open(self.fname_fid, "rb"))

    def _get_var_fid(self, name_of_variable):
        # TODO want to document structure of results later
        if name_of_variable not in ['Hubble', 'angular_diameter_distance']:
            print('Error: name can only be: Hubble, angular_diameter_distance')
            # throw error
        is_same_z = self._check_z_fid()
        # if not is_same:
        # error handling goes here (decide how)
        print('is_same_z = ', is_same_z)
        var = self.results[name_of_variable]
        assert var.size == self.results['aux']['z'].size, (
            var.size, self.results['aux']['z'].size)
        # TODO check against runing get_fid_model.py 68.8920532   80.38958972  94.11747952 109.75555669
        print('fid %s:' % name_of_variable, var)
        return var

    def _check_z_fid(self):
        z = self.results['aux']['z']
        is_same = np.all(self.z == z)
        return is_same

    def _calculate_galaxy_ps(self, state):
        """Returns 4-d numpy array of shape (n_ps, nz, nk, nmu) for the galaxy power spectra.

        Note: n_ps is the number of unique auto- and cross- power spectra bewteen nsample of
        galaxy samples:
            n_ps = nsample * (nsample + 1) / 2.
        """
        n_ps = int(self.nsample * (self.nsample + 1) / 2)
        galaxy_ps = np.zeros((n_ps, self.nz, self.nk, self.nmu))
        jj = 0
        for j1 in range(self.nsample):
            for j2 in range(j1, self.nsample):
                galaxy_ps[jj] = \
                    state['AP_factor'].reshape((self.nz, 1, 1)) \
                    * state['matter_power'].reshape((self.nz, self.nk, 1)) \
                    * state['galaxy_transfer'][j1, :, :, :] \
                    * state['galaxy_transfer'][j2, :, :, :]
                jj = jj + 1
        return galaxy_ps

    def _calculate_matter_power(self, nonlinear=False):
        """ Returns 2-d numpy array of shape (nz, nk) for the matter power spectrum.
        Default is linear matter power."""

        Pk_interpolator = self.provider.get_Pk_interpolator(
            nonlinear=nonlinear)
        matter_power = np.exp(np.array(
            Pk_interpolator(self.z_list, np.log(self.k))))
        return matter_power

        # TODO: Check robustness of the interpolator (or whatever method adopted)
        # 1) use get_Pk_grid to cross-check results of interpolation
        #    Code:
        #   Pk = self.provider.get_Pk_grid(nonlinear=False)
        #    Pk_grid is a tuple of three arrays (z, k, Pk), where z, k are 1-d, and Pk is nz x nk.
        # 2) plot out P(k) interpolated and before interpolation
        # 3) look at linear vs nonlinear
        #    understand problem w/ diff = -0.5 for linear scales w/ the interpolator

    def _calculate_matter_power_from_grid(self):
        """ Returns 2-d numpy array of shape (nz, nk) for the matter power spectrum."""
        Pk_grid = self.provider.get_Pk_grid(
            nonlinear=False)  # returns (k, z, Pk)
        self.k = Pk_grid[0]  # update k and nk
        self.nk = self.k.size
        assert np.all(self.z == Pk_grid[1]), (self.z, Pk_grid[1])
        matter_power = Pk_grid[2]  # 2-d numpy array of shape (nz, nk)
        # add test to check
        return matter_power

    def _calculate_galaxy_transfer(self, **params_values_dict):
        """Returns 4-d numpy array of shape (nsample, nz, nk, mu) for the galaxy transfer functions."""

        bias = self._calculate_galaxy_bias(
            **params_values_dict)  # (nsample, nz, nk)
        kaiser = self._calculate_rsd_kaiser(bias)  # (nsample, nz, nk, mu)
        galaxy_transfer = \
            bias[:, :, :, np.newaxis] * kaiser * \
            self._calculate_nl_damping() * \
            self._calculate_rsd_nl()
        return galaxy_transfer

    def _calculate_AP_factor(self):
        """Returns 1-d numpy array of shape (nz,) for the Alcock-Pacynzski factors."""

        if self.is_fiducial_model is True:
            ap = np.ones(self.nz)
        else:
            DA = self.provider.get_angular_diameter_distance(self.z_list)
            Hubble = self.provider.get_Hubble(self.z_list)
            Hubble_fid = self._get_var_fid('Hubble')
            DA_fid = self._get_var_fid('angular_diameter_distance')
            ap = (DA_fid / DA) ** 2 * (Hubble / Hubble_fid)
        return ap

    def _calculate_rsd_kaiser(self, bias):
        """Returns a 4-d numpy array of shape (nsample, nz, nk, nmu) for RSD Kaiser factor.

        Note: we return one factor of 
            kaiser = (1 + f(z)/b_j(k,z) * mu^2) 
        for the galaxy density, not power spectrum.
        """
        f = self._calculate_growth_rate()  # shape (nz,) # time: 0.001 sec
        # self._calculate_growth_rate_approx()  # time:  1.71661376953125e-05 sec
        # TODO choose best method
        kaiser = (1.0 + f[:, np.newaxis] / bias)[:, :, :, np.newaxis]\
            * self.mu ** 2
        print('==>kaiser.shape', kaiser.shape)
        assert kaiser.shape == (self.nsample, self.nz, self.nk, self.nmu)
        # TODO turn shape test into unit test
        # print('==>kaiser.shape', kaiser.shape)
        # assert kaiser.shape == (self.nsample, self.nz, self.nk, self.nmu)
        # TODO turn into unit test
        # iz = 1
        # print('a[0,iz=%i,:]' % iz, a[iz, :])
        # print('expect:', f[iz] / bias[0, iz, :])
        return kaiser

    def _calculate_nl_damping(self):
        """Return 3-d numpy array of shape (nz, nk, nmu) for the non-linear damping.

        Note: This is the factor for the galaxy density (not power spectrum), so
            ans = exp(-1/4 * arg), 
        where 
            arg = k^2 * [Sig_perp^2 + mu^2 * (Sig_para^2 - Sig_perp^2)]
        and 
            Sig_perp(z) = c_rec * D(z) * Sig0 and Sig_para(z) = Sig_perp * (1+f(z)).
        """

        self._Sig0 = 11.0  # Mpc/h # TODO need to change unit
        Sig_perp = self._fraction_recon * self._Sig0 * self._calculate_growth()  # (nz,)
        # might be able to avoid calculating f(z) twice # TODO optimize later
        Sig_para = Sig_perp * (1.0 + self._calculate_growth_rate())  # (nz,)
        arg = self.k[np.newaxis, :, np.newaxis] \
            * ((Sig_perp**2)[:, np.newaxis, np.newaxis]
               + (self.mu**2)[np.newaxis, np.newaxis, :]
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
        k_parallel = self.k[:, np.newaxis] \
            * self.mu  # (nk, nmu)
        Hubble = self.provider.get_Hubble(self.z_list)
        # fac = Hubble.reshape((self.nz, 1, 1)) / k_parallel
        k_parallel_over_H = k_parallel[np.newaxis, :, :] \
            / Hubble.reshape((self.nz, 1, 1))
        # TODO turn into unit test
        assert k_parallel_over_H.shape == (self.nz, self.nk, self.nmu)
        print('==>k_parallel_over_H.shape', k_parallel_over_H.shape)
        arg = np.array(self.sigz).reshape(self.nsample, 1, 1, 1) \
            * (1.0 + self.z.reshape(1, self.nz, 1, 1)) \
            * k_parallel_over_H
        # TODO turn into unit test
        print('==>arg.shape', arg.shape)
        assert arg.shape == (self.nsample, self.nz, self.nk, self.nmu)
        ans = np.exp(- arg ** 2 / 2.0)
        return ans

    def _calculate_galaxy_bias(self, **params_values_dict):
        """Returns galaxy bias in a 3-d numpy array of shape (nsample, nz, nk)."""
        gaussian_bias_per_sample = self._calculate_gaussian_bias_array(
            **params_values_dict)
        gaussian_bias_per_sample = gaussian_bias_per_sample[:, np.newaxis]
        alpha = self._calculate_alpha()
        galaxy_bias = np.array([
            gaussian_bias_per_sample[j]
            + 2.0 * params_values_dict['fnl']
            * (gaussian_bias_per_sample[j] - 1.0) * self._delta_c / alpha
            for j in range(self.nsample)
        ])
        print('==>galaxy_bias.shape', galaxy_bias.shape)
        assert galaxy_bias.shape == (self.nsample, self.nz, self.nk)
        # TODO add unit test for galaxy_bias shape
        # TODO add test for galaxy_bias content (perhaps w/ a plot)

        return galaxy_bias

    def _calculate_gaussian_bias_array(self, **params_values_dict):
        """"Returns a 1-d numpy array of gaussian galaxy bias, for each galaxy sample.
        """
        keys = ['gaussian_bias_%s' % (i) for i in range(1, self.nsample + 1)]
        gaussian_bias = np.array([params_values_dict[key] for key in keys])
        print('==>gaussian_bias', gaussian_bias)
        # TODO add unit test check gaussian_bias.shape == (self.nsample, )
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
        """Returns 1-d numpy array of initial power spectrum evaluated at self.k."""
        k0 = self._k0  # 1/Mpc
        As = self.provider.get_param('As')
        ns = self.provider.get_param('ns')
        initial_power = (2.0 * np.pi**2) / (self.k**3) * \
            As * (self.k / k0)**(ns - 1.0)
        return initial_power

    def _calculate_growth_rate_approx(self):
        """Returns f(z) = Omega_m(z)^0.55 in a 1-d numpy array"""
        growth_rate_index = 0.55
        omegam = self.provider.get_param('omegam')
        f = (omegam * (1.0 + self.z) ** 3) ** growth_rate_index
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

    def _run_tests(self, params_values_dict):
        if self.do_test is True:
            # TODO add more tests
            if self.do_test_plot is True:
                self._make_plots(params_values_dict,
                                 names=self.test_plot_names)

    def _make_plots(self, params_values_dict, names=None):
        if names is None:
            names = [
                'Hubble',
                'angular_diameter_distance',
                'matter_power',
                # 'matter_power_from_grid',
                'alpha',
                'galaxy_transfer'
            ]
        plotter = Plotter(self, params_values_dict)
        plotter.make_plots(names)
        # could remove this in the future if it causes problems
        plt.close('all')


class Plotter():

    def __init__(self, theory, params_values_dict):
        self.theory = theory
        self.z = self.theory.z
        self.k = self.theory.k
        self.mu = self.theory.mu
        self.nz = self.theory.nz
        self.nk = self.theory.nk
        self.nmu = self.theory.nmu
        self.nsample = self.theory.nsample
        self.params_values_dict = params_values_dict

    def make_plots(self, names):
        for name in names:
            function_name = 'plot_' + name
            function = getattr(self, function_name, None)
            if callable(function):
                print('Calling function {}'.format(function_name))
                function()
            else:
                print('Function {} is not implemented!'.format(function_name))
        # sys.exit()

    def plot_angular_diameter_distance(self):
        y1 = self.theory.provider.get_angular_diameter_distance(self.z)
        y_list = [y1]
        if self.theory.is_fiducial_model is False:
            y2 = self.theory._get_var_fid('angular_diameter_distance')
            y_list.append(y2)
        self.plot_1D('z', y_list, '$D_A(z)$', 'plot_angular_diameter_distance.png',
                     legend=['current model', 'fid. model'])

    def plot_Hubble(self):
        y1 = self.theory.provider.get_Hubble(self.z)
        y_list = [y1]
        if self.theory.is_fiducial_model is False:
            y2 = self.theory._get_var_fid('Hubble')
            y_list.append(y2)
        self.plot_1D('z', y_list, '$H(z)$', 'plot_Hubble.png',
                     legend=['current model', 'fid. model'])

    def plot_alpha(self):
        data = self.theory._calculate_alpha()
        axis_names = ['z', 'k']
        ylatex = '$\\alpha(k)$'
        yname = 'alpha'
        plot_type = 'loglog'
        self.plot_2D(axis_names, data, ylatex, yname, plot_type=plot_type)

    def plot_matter_power(self):

        data_nl = self.theory._calculate_matter_power(nonlinear=True)
        data_lin = self.theory._calculate_matter_power(nonlinear=False)

        axis_names = ['z', 'k']
        plot_type = 'loglog'

        yname = 'matter_power_nonlinear'
        ylatex = '$P_m^{\mathrm{NL}}(z,k)$ [Mpc$^3$]'
        self.plot_2D(axis_names, data_nl, ylatex, yname, plot_type=plot_type)

        yname = 'matter_power_linear'
        ylatex = '$P_m^{\mathrm{lin}}(z,k)$ [Mpc$^3$]'
        self.plot_2D(axis_names, data_lin, ylatex, yname, plot_type=plot_type)

    def plot_matter_power_from_grid(self):
        data = self.theory._calculate_matter_power_from_grid()
        axis_names = ['z', 'k']
        ylatex = '$P_m(z,k)$ from grid'
        yname = 'matter_power_from_grid'
        plot_type = 'loglog'
        self.plot_2D(axis_names, data, ylatex, yname, plot_type=plot_type)

    def plot_galaxy_transfer(self):
        data = self.theory._calculate_galaxy_transfer(
            **self.params_values_dict)
        axis_names = ['sample', 'z', 'k', 'mu']
        yname = 'galaxy_transfer'
        plot_type = 'loglog'

        isamples = range(self.nsample)
        izs = range(self.nz)
        for isample in isamples:
            for iz in izs:
                axis_names_in = ['k', 'mu']
                yname_in = yname + '_isample%s_iz%s' % (isample, iz)
                data_in = data[isample, iz, :, :]
                ylatex = 'galaxy transfer function $T_g(k, \mu)$'
                self.plot_2D(axis_names_in, data_in, ylatex,
                             yname_in, plot_type=plot_type)

    def plot_galaxy_power(self):  # TODO to be completed
        data = self.theory._calculate_galaxy_power()
        axis_names = ['ps', 'z', 'k', 'mu']
        yname = 'galaxy_power'
        plot_type = 'loglog'

        isamples = 0  # range(self.nsample)
        izs = range(self.nz)
        for isample in isamples:
            for iz in izs:
                axis_names_in = ['k', 'mu']
                yname_in = yname + '_ips%s_iz%s' % (ips, iz)
                data_in = data[isample, iz, :, :]
                ylatex = 'galaxy power spectrum $P_g(k, \mu)$'
                self.plot_2D(axis_names_in, data_in, ylatex,
                             yname_in, plot_type=plot_type)

    def plot_2D(self, *args, **kwargs):
        self.make_plot_2D_fixed_axis('col', * args, **kwargs)
        self.make_plot_2D_fixed_axis('row', * args, **kwargs)

    def make_plot_2D_fixed_axis(self, fixed_axis_name, axis_names, data, ylatex, yname, plot_type='plot'):
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

        x = self.get_xarray(axis_names[FIXED])
        legend = ['%s = %.2e' % (axis_names[FIXED], x[i])
                  for i in indices]

        plot_name = 'plot_%s_vs_%s.png' % (yname, axis_names[VARIED])
        self.plot_1D(axis_names[VARIED], y_list, ylatex,
                     plot_name, legend=legend, plot_type=plot_type)

    def _get_fixed_and_varied_axes(self, fixed_axis):
        if fixed_axis == 'row':
            FIXED = 0
            VARIED = 1
        elif fixed_axis == 'col':
            FIXED = 1
            VARIED = 0
        else:
            print('Error: fixed_axis can only be row or col')
        return FIXED, VARIED

    def _get_indices(self, n_tot, n_wanted=5):
        delta = 1 if n_tot <= n_wanted else int(np.floor(
            n_tot / np.float(n_wanted)))
        indices = np.arange(0, n_tot, delta)
        # TODO add logging/logger ability to print only in debug mode
        print('delta = {}'.format(delta))
        print('indices = {}'.format(indices))
        return indices

    def get_xarray(self, dim):
        if dim == 'z':
            x = self.z
        elif dim == 'k':
            x = self.k
        elif dim == 'mu':
            x = self.mu
        elif dim == 'sample':
            x = np.arange(self.nsample)
        else:
            print('Error: get_xarray can only take dim=z, k, mu or sample.')
            # TODO raise custom Error
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
        else:
            print('error: ...')
        return xlabel

    def plot_1D(self, z_or_k, y_list, ylatex, plot_name, legend=None, plot_type='plot'):

        allowed_plot_types = ['plot', 'loglog', 'semilogx', 'semilogy']

        x = self.get_xarray(z_or_k)
        xlabel = self.get_xlabel(z_or_k)

        fig, ax = plt.subplots()

        for y in y_list:
            if plot_type in allowed_plot_types:
                getattr(ax, plot_type)(x, y)
            else:
                print(
                    "Error: plot_type can only be one of the following: {}".format(allowed_plot_types))
                # Error handling for plot

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylatex)
        ax.legend(legend)

        plot_name = os.path.join(self.theory.test_dir, plot_name)
        fig.savefig(plot_name)
        print('Saved plot = {}'.format(plot_name))
        plt.close()
