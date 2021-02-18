import numpy as np
import copy
import sys

from theory.data_vector.cosmo_product import CosmoProductCreator
from theory.data_vector.cosmo_product import CosmoProduct_FromCobayaProvider
from theory.data_vector.cosmo_product import CosmoProduct_FromCamb
from theory.utils import constants
from theory.utils.errors import NameNotAllowedError
from theory.utils.logging import class_logger

class GRSIngredientsCreator():

    """Create an instance of the GRSIngredients class given different input options"""

    def create(self, option, survey_par, data_spec, \
            cosmo_par=None, cosmo_par_fid=None, \
            provider=None, z=None, nonlinear=None, \
            **params_values_dict):
        
        cosmo_creator = CosmoProductCreator()
        z = survey_par.get_zmid_array()

        option_fid = 'FromCamb'
        cosmo_fid = cosmo_creator.create(option_fid, cosmo_par=cosmo_par_fid, z=z)

        cosmo = cosmo_creator.create(option, cosmo_par=cosmo_par, \
            provider=provider, z=z, nonlinear=nonlinear)

        print('testing get_angular...', cosmo.get_angular_diameter_at_z([1.0]))
    
        params_values_dict = self._get_params_values_dict(option, params_values_dict, cosmo_par)

        return GRSIngredients(cosmo, cosmo_fid, survey_par, data_spec, **params_values_dict)

    def _get_params_values_dict(self, option, params_values_dict, cosmo_par):
        if option == 'FromCobayaProvider':
            return params_values_dict
        elif option == 'FromCamb':
            return self._get_params_values_dict_from_cosmo_par(cosmo_par)
        else:
            raise ValueError('GRSIngredientsCreator: \
                option can only be "FromCobayaProvider" or "FromCamb".')
        #TODO Might be able to simplify multiple cosmo parameter passing channels
        # cosmo_par and **params_values_dict.

    @staticmethod
    def _get_params_values_dict_from_cosmo_par(cosmo_par):
        params_values_dict = {
            'As': cosmo_par.As,\
            'ns': cosmo_par.ns,\
            'nrun': cosmo_par.nrun,\
            'fnl': cosmo_par.fnl,\
        }
        return params_values_dict

class GRSIngredients(object):

    def __init__(self, cosmo, cosmo_fid, survey_par, data_spec, **params_values_dict):
       
        """
        Args:
            params_values_dict: has gaussian_bias_sample_<i>_z_<j> where i and j start at 1.
        """

        self._logger = class_logger(self)

        self._cosmo = cosmo
        self._cosmo_fid = cosmo_fid
        self._survey_par = survey_par
        self._d = data_spec
        self._params_values_dict = params_values_dict
        
        ap_perp, ap_para = self._get_AP_perp_and_para()
        self._k_actual, self._mu_actual = self._d.get_k_and_mu_actual(ap_perp, ap_para)
        self._k_actual_perp, self._k_actual_para = self._d.get_k_actual_perp_and_para(ap_perp, ap_para)

        self._allowed_ingredients = [\
            'alpha', \
            'alpha_without_AP', \
            'galaxy_bias', \
            'galaxy_bias_without_AP', \
            'galaxy_bias_20', \
            'gaussian_bias', \
            'AP', \
            'matter_power_with_AP',\
            'matter_power_without_AP',\
            'kaiser', \
            'fog', \
            'fog_using_ref_cosmology',\
            'sigp', \
            'fnl', \
        ]
        self._ingredients = {'fnl': params_values_dict['fnl'] }

    def _get_matter_power_at_z_and_ks(self, z, ks):
        matter_power = self._cosmo.get_matter_power_at_z_and_k(z, ks)
        return matter_power

    @property
    def allowed_ingredients(self):
        return self._allowed_ingredients

    @property
    def k_actual(self):
        """3d numpy array of shape (nz, nk, nmu)"""
        return self._k_actual

    @property
    def mu_actual(self):
        """1d numpy array of shape (nmu)"""
        return self._mu_actual

    def get(self, name):
        if name in self._allowed_ingredients:
            if name not in self._ingredients.keys():
                getattr(self, '_calc_'+name)()
            return self._ingredients[name]
        else:
            raise NameNotAllowedError(name, self._allowed_ingredients)

    def _calc_F2(self):
        """
        docstring
        """
        pass

    def _get_k_and_mu_actual(self):
        AP_perp, AP_para = self._get_AP_perp_and_para()
        return self._d.get_k_and_mu_actual(AP_perp, AP_para)

    def _get_survey_volume_array(self):
        """Returns 1d numpy array of shape (nz,) for the volume of the 
        redshift bins.
        """
         #TODO abolish zmid calculations somewhere
        zhi = self._survey_par.get_zhi_array()
        zlo = self._survey_par.get_zlo_array()

        d_hi = self._cosmo.get_comoving_radial_distance_at_z(zhi)
        d_lo = self._cosmo.get_comoving_radial_distance_at_z(zlo)

        V_array = (4.0 * np.pi)/3.0 * (d_hi**3 - d_lo**3)

        return V_array

    def _test_get_volume(self):
       
        zlo = np.array([0.65, 0.75, 0.85, 0.95, 1.05, 1.75])
        zhi = np.array([0.75, 0.85, 0.95, 1.05, 1.15, 1.85])
        zmid = np.array([0.70, 0.80, 0.9, 1.0, 1.1, 1.8])

        V_array = np.zeros_like(zhi)
        for i in range(zhi.size):
            d_hi = self._cosmo.get_comoving_radial_distance_at_z(zhi[i])
            d_lo = self._cosmo.get_comoving_radial_distance_at_z(zlo[i])
            V_array[i] = (4.0 * np.pi)/3.0 * (d_hi**3 - d_lo**3) 
        
        DESI_area = 14000
        fsky = DESI_area/180/180*np.pi/4.0
        h = self._get_H0_fid()/100.0

        V_array = V_array * h**3 * fsky/1e9

        print('fsky', fsky)
        print('V_array:', V_array)
        print('Expected V (DESI) =', [2.63, 3.15, 3.65, 4.10, 4.52, 6.43], \
            '(Gpc/h)^3 [up to cosmology differences]')
        # https://arxiv.org/pdf/1611.00036.pdf Table 2.3

        return V_array

    def _get_number_density_array(self):
        """Returns 2d numpy array of shape (nsample, nz) for the number density 
        in (1/Mpc)^3 (not (h/Mpc)^3 as in input yaml)."""
        h = self._get_H0_fid()/100.0
        number_density_invMpc = h**3 * self._survey_par.get_number_density_array()
        self._logger.debug('h = {}'.format(h))
        return number_density_invMpc

    def _calc_AP(self):
        """Returns AP factor in an array same size as self._d.z."""
        assert 'AP' not in self._ingredients.keys()
        AP_perp, AP_para = self._get_AP_perp_and_para()
        AP = (AP_perp)**2 * (AP_para)
        self._ingredients['AP'] = AP

    def _calc_alpha(self):
        """Returns alpha as 3-d numpy array with shape (nz, nk, nmu) """

        assert 'alpha' not in self._ingredients.keys()
        
        initial_power = self._get_initial_power(self._k_actual)
        matter_power = self.get('matter_power_with_AP')

        expected_shape = self._d.shape[1:]

        assert initial_power.shape == expected_shape, \
            (initial_power.shape, expected_shape)

        assert matter_power.shape == expected_shape, \
            (matter_power.shape, expected_shape)

        alpha = (5.0/3.0) * np.sqrt( matter_power/initial_power )

        assert alpha.shape == expected_shape, \
            (alpha.shape, expected_shape)

        self._ingredients['alpha'] = alpha

    def _calc_alpha_without_AP(self):
        """Returns alpha without AP effect as 2d numpy array with shape (nz, nk) """

        assert 'alpha' not in self._ingredients.keys()
        
        initial_power = self._get_initial_power(self._d.k)
        matter_power = self.get('matter_power_without_AP')

        expected_shape = (self._d.nz, self._d.nk)

        assert initial_power.shape == (self._d.nk,), \
            (initial_power.shape, (self._d.nk))

        assert matter_power.shape == expected_shape, \
            (matter_power.shape, expected_shape)

        alpha = (5.0/3.0) * np.sqrt(matter_power/initial_power[np.newaxis, :])

        assert alpha.shape == expected_shape, \
            (alpha.shape, expected_shape)

        self._ingredients['alpha_without_AP'] = alpha

    def _calc_galaxy_bias(self):
        """Returns galaxy bias in a 4-d numpy array of shape (nsample, nz, nk, nmu)."""

        assert 'galaxy_bias' not in self._ingredients.keys()

        delta_c = 1.686

        gaussian_bias = self.get('gaussian_bias')[:, :, np.newaxis, np.newaxis]

        alpha = self.get('alpha')[np.newaxis, :, :, :]

        galaxy_bias = gaussian_bias + 2.0*self.get('fnl')*delta_c*(gaussian_bias-1.0)/alpha

        expected_shape = self._d.transfer_shape
        msg = ('galaxy_bias.shape = {}, expected ({})'
               .format(galaxy_bias.shape, expected_shape))
        assert galaxy_bias.shape == expected_shape, msg

        self._ingredients['galaxy_bias'] = galaxy_bias

    def _calc_galaxy_bias_without_AP(self):
        """Returns galaxy bias in a 4-d numpy array of shape (nsample, nz, nk)."""

        assert 'galaxy_bias_without_AP' not in self._ingredients.keys()

        delta_c = 1.686

        gaussian_bias = self.get('gaussian_bias')[:, :, np.newaxis]

        alpha = self.get('alpha_without_AP')[np.newaxis, :, :]

        galaxy_bias = gaussian_bias + 2.0*self.get('fnl')*delta_c*(gaussian_bias-1.0)/alpha

        expected_shape = (self._d.nsample, self._d.nz, self._d.nk)

        msg = ('galaxy_bias.shape = {}, expected ({})'
               .format(galaxy_bias.shape, expected_shape))

        assert galaxy_bias.shape == expected_shape, msg

        self._ingredients['galaxy_bias_without_AP'] = galaxy_bias

    def _calc_galaxy_bias_20(self, fnl=None):
        """Returns 2d numpy array of shape (nsample, nz) for second-order galaxy 
        bias, a number for each galaxy sample and redshift
        """
        b1 = self.get('gaussian_bias')
        b2 = self._get_b2_Lezeyras_2016_for_b1(b1)
        self._ingredients['galaxy_bias_20'] = b2 

    @staticmethod
    def _get_b2_Lezeyras_2016_for_b1(b1):
        """Returns b2 given b1 using GR fit from Lezeyras et al. 2016
        https://arxiv.org/pdf/1511.01096.pdf Eq. 5.2 
        (Also used in SPHEREx forecast) defined with 
        delta_g(x) = b1 * delta_m(x) + 0.5 * b2 * delta_m^2(x).
        """
        b2 = 0.412 - 2.143 * b1 + 0.929 * b1 * b1 + 0.008 * b1 * b1 * b1
        return b2

    def _calc_gaussian_bias(self):
        """"Returns a 2-d numpy array of shape (nsample, nz) for gaussian galaxy bias,
        a number for each galaxy sample and redshift.
        """
        assert 'gaussian_bias' not in self._ingredients.keys()

        gaussian_bias = np.zeros((self._d.nsample, self._d.nz))
        gaussian_bias_default = self._survey_par.get_galaxy_bias_array()
        if 'gaussian_bias_sample_1_z_1' not in self._params_values_dict.keys():
            gaussian_bias = copy.copy(gaussian_bias_default)
        else:
            for isample in range(self._d.nsample):
                for iz in range(self._d.nz):
                    key = 'gaussian_bias_sample_%s_z_%s' % (isample + 1, iz + 1)
                    gaussian_bias[isample, iz] = self._params_values_dict[key]

        self._ingredients['gaussian_bias'] = gaussian_bias

    def _calc_sigp(self):
        """"Returns a 2-d numpy array of shape (nsample, nz) for sigmap,
        where sigp = sigma_z * (1+z) * c / H
        """
        
        Hubble = self._cosmo.get_Hubble_at_z(self._d.z)
        sigp = self._survey_par.get_sigz_array() \
            * (1.0 + self._d.z[np.newaxis, :]) \
            * constants.c_in_km_per_sec\
            / Hubble[np.newaxis, :]

        self._ingredients['sigp'] = sigp

        # TODO check: expect suppression at k ~ 0.076 1/Mpc 
        # for H = 75km/s/Mpc at z = 0.25 w/ sigma_z = 0.003

    def _calc_matter_power_with_AP(self, nonlinear=False):
        """ Returns 3-d numpy array of shape (nz, nk, nmu) for the matter power spectrum.
        Default is linear matter power. Note that the power spectrum itself is evaluated 
        at z, but the k also has a dependence on z and mu due to the AP factor varying 
        with z and mu."""

        assert 'matter_power_with_AP' not in self._ingredients.keys()

        ks = self._k_actual
        zs = self._d.z
        matter_power = np.zeros_like(ks)
        for iz in range(zs.size):
            for imu in range(self._d.nmu): 
                p = self._get_matter_power_at_z_and_ks(zs[iz], ks[iz, :, imu])
                matter_power[iz, :, imu] = p
        self._ingredients['matter_power_with_AP'] = matter_power

    def _calc_matter_power_without_AP(self, nonlinear=False):
        """ Returns 3-d numpy array of shape (nz, nk) for the matter power spectrum.
        Default is linear matter power."""

        assert 'matter_power_without_AP' not in self._ingredients.keys()

        ks = self._d.k
        zs = self._d.z
        matter_power = np.zeros((zs.size, ks.size))
        for iz in range(zs.size):
            p = self._get_matter_power_at_z_and_ks(zs[iz], ks)
            matter_power[iz, :] = p
        self._ingredients['matter_power_without_AP'] = matter_power

    def _calc_kaiser(self):
        """Returns a 4-d numpy array of shape (nsample, nz, nk, nmu) for RSD Kaiser factor.

        Note: we return one factor of
            kaiser = (1 + f(z)/b_j(k,z) * mu^2)
        for the galaxy density, not power spectrum.
        """
        f = self._get_f()
        bias = self.get('galaxy_bias')
        
        kaiser = 1.0 + f[np.newaxis, :, np.newaxis, np.newaxis] / bias \
            * (self._mu_actual ** 2)[np.newaxis, :, :, :]

        expected_shape = self._d.transfer_shape
        assert kaiser.shape == expected_shape, (kaiser.shape, expected_shape)
        
        self._ingredients['kaiser'] = kaiser

    def _calc_fog(self):  
        """Returns 4-d numpy array of shape (nsample, nz, nk, nmu) for the blurring
        due to redshift error exp(-arg^2/2) where arg = (1+z) * sigz * k_parallel * c /H(z). """

        Hubble = self._cosmo.get_Hubble_at_z(self._d.z)
        arg = self._survey_par.get_sigz_array()[:, :, np.newaxis, np.newaxis] \
            * (1.0 + self._d.z[np.newaxis, :, np.newaxis, np.newaxis]) \
            * self._k_actual_para[np.newaxis, :, :, :] \
            * constants.c_in_km_per_sec\
            / Hubble[np.newaxis, :, np.newaxis, np.newaxis]

        # TODO check: expect suppression at k ~ 0.076 1/Mpc 
        # for H = 75km/s/Mpc at z = 0.25 w/ sigma_z = 0.003

        fog = np.exp(- arg ** 2 / 2.0)

        expected_shape = self._d.transfer_shape
        assert fog.shape == expected_shape, (fog.shape, expected_shape)

        self._ingredients['fog'] = fog


    def _calc_fog_using_reference_cosmology(self):  
        """Returns 4-d numpy array of shape (nsample, nz, nk, nmu) for the blurring
        due to redshift error exp(-arg^2/2) where arg = (1+z) * sigz * k_parallel * c /H(z). """

        k_ref_para = self._d.k[:, np.newaxis] * self._d.mu[np.newaxis, :] 
        Hubble_ref = self._cosmo_fid.get_Hubble_at_z()
        arg_ref = self._survey_spec.get_sigz_array[:, :, np.newaxis, np.newaxis] \
            * (1.0 + self._d.z[np.newaxis, :, np.newaxis, np.newaxis]) \
            * k_ref_para[np.newaxis, np.newaxis, :, :]\
            * constants.c_in_km_per_sec  \
            / Hubble_ref[np.newaxis, :, np.newaxis, np.newaxis]

        fog_ref = np.exp(- arg_ref ** 2 / 2.0)

        expected_shape = self._d.transfer_shape
        assert fog_ref.shape == expected_shape, (fog_ref.shape, expected_shape)

        #TODO could make this a unit test instead:
        fog = self.get('fog')
        assert np.allclose(fog_ref, fog)

        self._ingredients['fog_using_reference_cosmology'] = fog_ref

    def _get_AP_perp_and_para(self):
        """Returns tuple of two elements for the AP factor in the 
        perpendicular and parallel direction to line-of-sight.
        Each element is an array the same size as self._d.z."""

        z = self._d.z

        D = self._cosmo.get_angular_diameter_at_z(z)
        D_fid = self._cosmo_fid.get_angular_diameter_at_z(z)

        H = self._cosmo.get_Hubble_at_z(z)
        H_fid = self._cosmo_fid.get_Hubble_at_z(z)

        return (D_fid/D, H/H_fid)

    def _get_H0(self):
        """Returns H0 in km/s/Mpc units."""
        H0 = self._cosmo.get_H0()
        return H0
    
    def _get_H0_fid(self):
        """Returns H0 in km/s/Mpc units."""
        H0 = self._cosmo_fid.get_H0()
        return H0

    def _get_sigma8_now(self):
        sigma8_now = self._cosmo.get_sigma8_now()
        print('sigma8 today:', sigma8_now)
        return sigma8_now

    def _get_f(self):
        f = self._cosmo.get_f_array()
        return f

    def _get_sigma8(self):
        sigma8 = self._cosmo.get_sigma8_array()
        return sigma8

    def _get_initial_power(self, k):
        # TODO want to make sure this corresponds to the same as camb module
        """Returns 3-d numpy array of shape (nz, nk, nmu) of initial power spectrum 
        evaluated at self._k_actual:

        initial_power = (2pi^2)/k^3 * P,
        where ln P = ln A_s + (n_s-1) * ln(k/k_0_scalar) + n_{run}/2 * ln(k/k_0_scalar)^2 

        Note: There is a z and mu dependence because the k_actual at which we 
        need to evaluate the initial power is different for different z and mu.
        """

        k_pivot_in_invMpc = 0.05 

        #TODO how to handle this?
        As = self._params_values_dict['As']
        ns = self._params_values_dict['ns']
        nrun = self._params_values_dict['nrun']
    
        print('As, ns, nrun', As, ns, nrun)

        lnk = np.log(k / k_pivot_in_invMpc)

        initial_power = np.log(As) + (ns - 1.0) * lnk  \
            + 0.5 * nrun * lnk ** 2 
        initial_power = (2.0 * np.pi**2)/(k**3) * np.exp(initial_power)
        
        return initial_power

class GRSIngredientsForBispectrum(GRSIngredients): #No AP

    def __init__(self, cosmo_par, cosmo_par_fid, survey_par, data_spec, **params_values_dict):
       super().__init__(cosmo_par, cosmo_par_fid, survey_par, data_spec, **params_values_dict)

