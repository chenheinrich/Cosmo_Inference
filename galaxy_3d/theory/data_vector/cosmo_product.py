import numpy as np
import logging
import sys
import collections

import camb 
from spherelikes.utils.log import LoggedError, class_logger

class RequestedRedshiftError(Exception):

    def __init__(self, input_zs, zmax):
        msg = 'Redshift requested for power spectrum is out of range!\n'
        msg += '    Supported redshift range z = [0, {}], input z = {}'.format(zmax, input_zs)
        self.msg = msg
        super().__init__(self.msg)

class RequestedWavenumberError(Exception):

    def __init__(self, input_ks, kmax):
        msg = 'Wavenumber requested for power spectrum is out of range!\n'
        msg += '    Supported wavenumber range k = [0, {}], input k = {}'.format(kmax, input_ks)
        self.msg = msg
        super().__init__(self.msg)

class CosmoProductCreator():
    """Creates an instance of the subclass of CosmoProduct, 
    either the CosmoProduct_FromCamb subclass or the
    CosmoProduct_FromCobayaProvider subclass.

    Args:
        option: A string, either "FromCobayaProvider" or "FromCamb".
        cosmo_par (optional): an instance of CosmoPar constaining 
            cosmological parameters.
        provider (optional): an instance of Cobaya provider, which
            contains access functions to camb results.
        z (optional): an numpy array of redshifts.
        nonlinear (optional): a boolean for calculating the nonlinear
            matter power spectrum (default is linear).
    """

    def create(self, option, cosmo_par=None, provider=None, z=None, nonlinear=False):
        if option == 'FromCobayaProvider':
            try:
                print('Creating the CosmoProduct_FromCobayaProvider subclass')
                return CosmoProduct_FromCobayaProvider(\
                    provider=provider, nonlinear=nonlinear)
            except Exception as e:
                print('Error: {}'.format(e))
                sys.exit() #TODO
        elif option == 'FromCamb':
            try:
                print('Creating the CosmoProduct_FromCamb subclass')
                return CosmoProduct_FromCamb(cosmo_par=cosmo_par, z=z) #TODO needs nonlinear?
            except Exception as e:
                print('Error: {}'.format(e))
                sys.exit() #TODO
        else:
            raise ValueError('CosmoProductCreator: option can only be "FromCobayaProvider" or "FromCamb".')

class CosmoProduct(object):

    def __init__(self):
        self.logger = class_logger(self)

    def get_Hubble_at_z(self, zs):
        raise NotImplementedError

    def get_angular_diameter_at_z(self, zs):
        raise NotImplementedError

    def get_H0(self):
        raise NotImplementedError

    def get_sigma8_now(self):
        raise NotImplementedError

    def get_f_array(self):
        raise NotImplementedError

    def get_sigma8_array(self):
        raise NotImplementedError

    def get_matter_power_at_z_and_k(self, zs, ks):
        raise NotImplementedError


class CosmoProduct_FromCobayaProvider(CosmoProduct):

    def __init__(self, provider=None, nonlinear=False):
        super().__init__()
        self._provider = provider
        self._Pk_interpolator = self._provider.get_Pk_interpolator(
            nonlinear=nonlinear)
    
    def get_Hubble_at_z(self, zs):
        return self._provider.get_Hubble(zs)
        
    def get_angular_diameter_at_z(self, zs):
        return self._provider.get_angular_diameter_distance(zs)

    def get_H0(self):
        return self._provider.get('H0')

    def get_sigma8_now(self):
        return self._provider.get('sigma8')

    def get_f_array(self, zs):
        sigma8 = self._calc_sigma8()
        fsigma8 = self._provider.get_fsigma8(zs) #self.zs #TODO check this
        f = fsigma8 / sigma8
        return f

    def _calc_sigma8(self):
        sigma8 = np.flip(
            self._provider.get_CAMBdata().get_sigma8()
        )
        self.logger.debug('_calc_sigma8 returns: {}'.format(sigma8))
        return sigma8

    def get_sigma8_array(self):
        return self._calc_sigma8()

    def get_matter_power_at_z_and_k(self, zs, ks):
        return self._Pk_interpolator(zs, ks)


class CosmoProduct_FromCamb(CosmoProduct):

    """Sample Usage:
    cosmo = CosmoProduct_FromCamb(cosmo_par, z)
    if you need to update redsfhit 
    """
    def __init__(self, cosmo_par=None, z=[0]):
        self.logger = class_logger(CosmoProduct_FromCamb)
        super().__init__()

        self._cosmo_par = cosmo_par

        self._want_redshift_zero = (0 in z)
        if not self._want_redshift_zero:
            self._z_with_zero = self._get_z_with_zero(z)

        self._zmax = np.max(self._z_with_zero)
        self._kmax = 1.2
        self._nonlinear = False
        self._camb_pars = self._get_camb_pars()
        
        self._results = self._get_camb_results()
        self._matter_power_interpolator = self._get_matter_power_interpolator()
    
    def _get_z_with_zero(self, z):
        z = self._format_z(z)
        assert len(z.shape) == 1
        if 0 not in z:
            z = np.insert(z, 0, 0)
            self.logger.info('Inserted 0 in redshift array: z = {}'.format(z))
        return z

    def _format_z(self, z):
        if isinstance(z, list):
            z = np.array(z)
        assert len(z.shape) == 1
        return z

    def _get_camb_results(self):
        results = camb.get_results(self._camb_pars)
        return results

    def _get_camb_pars(self):
        pars = camb.CAMBparams()
        self._set_cosmo_pars(pars)
        self._set_matter_power_pars(pars)
        return pars

    def _set_cosmo_pars(self, pars):
        pars.set_cosmology(\
            #H0=67.5, 
            cosmomc_theta=self._cosmo_par.cosmomc_theta,\
            ombh2=self._cosmo_par.ombh2,\
            omch2=self._cosmo_par.omch2,\
            mnu=self._cosmo_par.mnu,\
            omk=self._cosmo_par.omegak,\
            tau=self._cosmo_par.tau
        )
        pars.InitPower.set_params(\
            As=self._cosmo_par.As,\
            ns=self._cosmo_par.ns,\
            r=0)

        # what about nrun, w0, wa?
        return pars

    def _set_matter_power_pars(self, pars):
        pars.set_matter_power(redshifts = self._z_with_zero, kmax=1.2)
        pars.NonLinear = camb.model.NonLinear_none

    def _get_matter_power_interpolator(self):   

        self._zmax = np.max(self._z_with_zero)

        matter_power_interpolator = camb.get_matter_power_interpolator(\
            self._camb_pars, nonlinear=self._nonlinear, \
            hubble_units=False, k_hunit=False, kmax=self._kmax, zmax=self._zmax)

        return matter_power_interpolator

    def get_matter_power_at_z_and_k(self, zs, ks):
        """
        Return matter power spectrum 
        """
        
        try:
            if np.any(zs > self._zmax) or np.any(zs < 0):
                raise RequestedRedshiftError(zs, self._zmax)
            if np.any(ks > self._kmax) or np.any(ks < 0):
                raise RequestedWavenumberError(ks, self._kmax)
        except (RequestedRedshiftError, RequestedWavenumberError) as e:
            self.logger.error(e.msg)
            sys.exit(0)
        
        Pk = self._matter_power_interpolator

        zs_is_scalar = not isinstance(zs, collections.Sequence)
        if zs_is_scalar:
            matter_power = np.array([Pk.P(zs, ks)])
        else:
            matter_power = np.array([Pk.P(z, ks) for z in zs])

        expected_shape = (np.array(zs).size, np.array(ks).size)
        assert matter_power.shape == expected_shape, (matter_power.shape, expected_shape)

        return matter_power

    def get_angular_diameter_at_z(self, zs):
        """
        Get angular diameter distance at any redshift zs. Scalar or array.
        """
        return self._results.angular_diameter_distance(zs)

    def get_comoving_radial_distance_at_z(self, zs):
        """
        Get comoving radial distance at any redshift zs. Scalar or array.
        """
        return self._results.comoving_radial_distance(zs)

    def get_Hubble_at_z(self, zs):
        """
        Get Hubble rate at any redshift zs in km/s/Mpc units. Scalar or array.
        """
        return self._results.hubble_parameter(zs)

    def get_H0(self):
        H0 = self.get_Hubble_at_z(0)
        return H0

    def get_sigma8_now(self):
        sigma8_now = self._results.get_sigma8_0()
        sigma8 = self._get_sigma8_array()
        assert sigma8[0] == sigma8_now, (sigma8[0], sigma8_now)
        return sigma8_now

    def get_sigma8_array(self):
        sigma8 = self._get_fsigma8_array()
        if self._want_redshift_zero is False:
            sigma8 = sigma8[1:]
        return sigma8

    def _get_sigma8_array(self):
        sigma8 = self._make_increasing_with_redshifts(self._results.get_sigma8())
        sigma8_now = self._results.get_sigma8_0()
        assert sigma8[0] == sigma8_now, (sigma8[0], sigma8_now)
        return sigma8
        
    def get_f_array(self):
        """
        Get logarithmic growth rate f = fsigma8/sigma8 at redshift zs. Scalar or array.
        
        Details: fsigma8 is defined as in the Planck 2015 parameter paper 
        (see https://arxiv.org/pdf/1502.01589.pdf) in terms of the 
        velocity-density correlation: sigma_{vd}^2/\sigma{dd} for 8ℎ−1Mpc spheres, 
        "where  v = −∇ · v_N/H, where v_N is the Newtonian-gauge peculiar velocity 
        of the baryons and dark matter, and d is the total matter density perturbation."
        
        "This definition assumes that the observed galaxies follow the flow of the 
        cold matter, not including massive neutrino velocity effects."
        
        """
        
        fsigma8 = self._get_fsigma8_array()
        sigma8 = self._get_sigma8_array()
        f = fsigma8/sigma8 
        if self._want_redshift_zero is False:
            f = f[1:]
        return f
        
    def _get_fsigma8_array(self):
        fsigma8 = self._results.get_fsigma8()
        fsigma8 = self._make_increasing_with_redshifts(fsigma8)
        return fsigma8
    
    def _make_increasing_with_redshifts(self, array):
        """Assumes array is decreasing with redshifts"""
        return np.flip(array)
