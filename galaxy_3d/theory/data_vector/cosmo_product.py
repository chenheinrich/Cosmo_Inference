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

    def create(self, option, zs, cosmo_par=None, provider=None, nonlinear=False):
        if option == 'FromCobayaProvider':
            try:
                print('About to create the CosmoProduct_FromCobayaProvider subclass')
                return CosmoProduct_FromCobayaProvider(zs, \
                    provider, nonlinear=nonlinear)
            except Exception as e:
                print('Error creating CosmoProduct_FromCobayaProvider: {}'.format(e))
                sys.exit() #TODO
        elif option == 'FromCamb':
            try:
                #TODO to polish error handling
                print('About to create the CosmoProduct_FromCamb subclass')
                assert (cosmo_par is not None), ('cosmo_par cannot be None \
                    when choosing "FromCamb" option for CosmoProduct')
                return CosmoProduct_FromCamb(zs, cosmo_par) #TODO needs nonlinear?
            except Exception as e:
                print('Error creating the CosmoProduct_FromCamb: {}'.format(e))
                sys.exit() #TODO
        else:
            raise ValueError('CosmoProductCreator: option can only be "FromCobayaProvider" or "FromCamb".')

class CosmoProduct(object):

    def __init__(self, zs):
        self.logger = class_logger(self)
        self._z = zs

        self._want_redshift_zero = (0 in zs)
        if not self._want_redshift_zero:
            self._z_with_zero = self._get_z_with_zero(zs)

    def get_param(self, paramname):
         raise NotImplementedError

    def get_Hubble_at_z(self, zs):
        raise NotImplementedError

    def get_angular_diameter_at_z(self, zs):
        raise NotImplementedError

    def get_H0(self):
        raise NotImplementedError

    def get_sigma8_now(self):
        raise NotImplementedError

    def get_matter_power_at_z_and_k(self, zs, ks):
        raise NotImplementedError

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
        self.logger.debug('fsigma8.shape = {}'.format(fsigma8.shape))
        self.logger.debug('sigma8.shape = {}'.format(sigma8.shape))
        
        f = fsigma8/sigma8 

        self.logger.debug('z = {}'.format(self._z_with_zero))
        self.logger.debug('f = {}'.format(f))
        self.logger.debug('fsigma8 = {}'.format(fsigma8))
        self.logger.debug('sigma8 = {}'.format(sigma8))
        
        if self._want_redshift_zero is False:
            f = f[1:]
        return f

    def get_sigma8_array(self):
        sigma8 = self._get_fsigma8_array()
        if self._want_redshift_zero is False:
            sigma8 = sigma8[1:]
        return sigma8
    
    def _get_fsigma8_array(self):
        raise NotImplementedError

    def _get_sigma8_array(self):
        raise NotImplementedError

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

    def _make_increasing_with_redshifts(self, array):
        """Assumes array is decreasing with redshifts"""
        return np.flip(array)

class CosmoProduct_FromCobayaProvider(CosmoProduct):

    def __init__(self, zs, provider, nonlinear=False):
        super().__init__(zs)

        self._provider = provider
        self._Pk_interpolator = self._provider.get_Pk_interpolator(
            nonlinear=nonlinear)

    def get_Hubble_at_z(self, zs):
        return self._provider.get_Hubble(zs)
        
    def get_angular_diameter_at_z(self, zs):
        return self._provider.get_angular_diameter_distance(zs)

    def get_param(self, paramname):
        return self._provider.get_param(paramname)
        
    def get_H0(self):
        return self._provider.get('H0')

    def get_sigma8_now(self):
        return self._provider.get('sigma8')

    def _get_fsigma8_array(self):
        fsigma8 = self._provider.get_fsigma8(self._z_with_zero) 
        return fsigma8

    def _get_sigma8_array(self):
        """Returns sigma8 array at self._z_with_zero."""
        sigma8 = self._provider.get_CAMBdata().get_sigma8()
        sigma8_array = self._make_increasing_with_redshifts(sigma8)
        return sigma8_array

    def get_matter_power_at_z_and_k(self, zs, ks):
        return self._Pk_interpolator(zs, ks)


class CosmoProduct_FromCamb(CosmoProduct):

    """Sample Usage:
    cosmo = CosmoProduct_FromCamb(zs, cosmo_par)
    if you need to update redsfhit 
    """
    def __init__(self, zs, cosmo_par):

        super().__init__(zs)

        self._cosmo_par = cosmo_par

        self._zmax = np.max(self._z_with_zero)
        self._kmax = 1.2
        self._nonlinear = False
        self._camb_pars = self._get_camb_pars()
        
        self._results = self._get_camb_results()
        self._matter_power_interpolator = self._get_matter_power_interpolator()

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

    
    #TODO to be implemented so we can call "As = self._cosmo.get_param('As')"
    #TODO reexamine if this is the best thing to do
    def get_param(self, paramname):
        return getattr(self._cosmo_par, paramname)

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
        #TODO turn into unit test so not doing that during sampling
        #sigma8 = self._get_sigma8_array()
        #assert sigma8[0] == sigma8_now, (sigma8[0], sigma8_now)
        return sigma8_now
        
    def _get_fsigma8_array(self):
        fsigma8 = self._results.get_fsigma8()
        fsigma8 = self._make_increasing_with_redshifts(fsigma8)
        return fsigma8

    def _get_sigma8_array(self):
        sigma8 = self._results.get_sigma8()
        sigma8 = self._make_increasing_with_redshifts(sigma8)
        return sigma8

    

    
