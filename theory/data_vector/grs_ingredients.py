import numpy as np

from theory.cosmo.cosmo_product import CosmoProduct_FromCobayaProvider
from theory.cosmo.cosmo_product import CosmoProduct_FromCamb

#TODO do this last!

def get_k_and_mu_actual(k, mu, ap_perp, ap_para, z=None):
    """Return four 3-d numpy arrays of shape (nz, nk, mu) 
    for the actual values of k_perp, k_para, k and mu
    given the two AP factors in directions perpendicular to 
    and parallel to the line-of-sigh, ap_perp and ap_para, 
    each specified as a 1-d numpy array of size self._d.size:
        k_perp = k_perp|ref * D_A(z)|ref / D_A(z),
        k_para = k_para|ref * (1/H(z))|ref / (1/H(z)),
    where
        k// = mu * k,
        kperp = sqrt(k^2  - k//^2) = k sqrt(1 - mu^2).
    """
    
    if z is not None:
        assert ap_perp.shape == z.shape, (ap_perp.shape, z.shape)
        assert ap_para.shape == z.shape, (ap_para.shape, z.shape)

    k_perp_ref = k[:, np.newaxis] * \
        np.sqrt(1. - (mu**2)[np.newaxis, :])
    k_para_ref = k[:, np.newaxis] * mu[np.newaxis, :]

    k_actual_perp = k_perp_ref[np.newaxis, :, :] * \
        ap_perp[:, np.newaxis, np.newaxis]

    k_actual_para = k_para_ref[np.newaxis, :, :] * \
        ap_para[:, np.newaxis, np.newaxis]

    k_actual = np.sqrt(k_actual_perp**2 + k_actual_para**2)

    mu_actual = k_actual_para/k_actual

    return (k_actual_perp, k_actual_para, k_actual, mu_actual)


class GRSIngredients(object):

    def __init__(self, cosmo_par, cosmo_par_fid, survey_par, data_spec):
       
        self._cosmo_par = cosmo_par
        self._cosmo_par_fid = cosmo_par_fid
        self._survey_par = survey_par
        self._d = data_spec
        
        self._cosmo = self._get_cosmo_product(self._cosmo_par, z=self._d.z)
        self._cosmo_fid = self._get_cosmo_product(self._cosmo_par_fid, z=self._d.z)

        ap_perp = self._get_AP_perp()
        ap_para = self._get_AP_para()
        self._k_actual_perp, self._k_actual_para, self._k_actual, self._mu_actual = \
            get_k_and_mu_actual(self._d.k, self._d.mu, ap_perp, ap_para, z=self._d.z)

        self._state = {}
        self._calculate()

    def _get_cosmo_product(self, cosmo_par, z):
        return CosmoProduct_FromCamb(cosmo_par, z)

    def _calculate(self):
        self._calc_alpha()

    def get_b1(self):
        """
        docstring
        """
        pass

    def get_F2(self):
        """
        docstring
        """
        pass

    def get_matter_power_at_z_and_k(self, z, k):
        matter_power = self._cosmo.get_matter_power_at_z_and_k(z, k)
        # TODO NEXT to test calculating alpha and matter power 
        return matter_power

    def get_AP(self):
        """Returns AP factor in an array same size as self._d.z."""
        AP_perp = self._get_AP_perp()
        AP_para = self._get_AP_para()
        AP = (AP_perp)**2 * (AP_para)
        return AP

    def _get_AP_perp(self):
        """Returns AP factor for k perpendicular to line-of-sight
        in an array same size as self._d.z."""
        z = self._d.z
        D = self._cosmo.get_angular_diameter_at_z(z)
        D_fid = self._cosmo_fid.get_angular_diameter_at_z(z)
        return D_fid/D

    def _get_AP_para(self):
        """Returns AP factor for k parallel to line-of-sight
        in an array same size as self._d.z."""
        z = self._d.z
        H = self._cosmo.get_Hubble_at_z(z)
        H_fid = self._cosmo_fid.get_Hubble_at_z(z)
        return H/H_fid

    def _get_H0(self):
        """Returns H0 in km/s/Mpc units."""
        H0 = self._cosmo.get_H0()
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

    def _get_k_and_mu_actual(self):
        ap_perp = self._get_AP_perp()
        ap_para = self._get_AP_para()
        return self._d.get_k_and_mu_actual(ap_perp, ap_para)

    def _calc_alpha(self):
        """Returns alpha as 3-d numpy array with shape (nz, nk, nmu) """
        initial_power = self._calc_initial_power()
        alpha = (5.0 / 3.0) \
            * np.sqrt(self._calc_matter_power() / initial_power)
        assert alpha.shape == self._d.shape[1:]
        self._state['alpha'] = alpha

    def get_alpha(self):
        return self._state['alpha']


    def _calc_initial_power(self):
        # TODO want to make sure this corresponds to the same as camb module
        """Returns 3-d numpy array of shape (nz, nk, nmu) of initial power spectrum 
        evaluated at self.k_actual:

        initial_power = (2pi^2)/k^3 * P,
        where ln P = ln A_s + (n_s -1) * ln(k/k_0_scalar) + n_{run}/2 * ln(k/k_0_scalar)^2 

        Note: There is a z and mu dependence because the k_actual at which we 
        need to evaluate the initial power is different for different z and mu.
        """

        k_pivot_in_invMpc = 0.05 

        As = self._cosmo_par.As 
        ns = self._cosmo_par.ns
        nrun = self._cosmo_par.nrun

        print('As, ns, nrun', As, ns, nrun)

        k = self._k_actual
        lnk = np.log(k / k_pivot_in_invMpc)

        initial_power = np.log(As) + (ns - 1.0) * lnk  \
            + 0.5 * nrun * lnk ** 2 
        initial_power = (2.0 * np.pi**2)/(k**3) * np.exp(initial_power)
        
        assert initial_power.shape == self._d.shape[1:], (initial_power.shape, self._d.shape[1:])

        return initial_power

    def _calc_matter_power(self, nonlinear=False):
        """ Returns 3-d numpy array of shape (nz, nk, nmu) for the matter power spectrum.
        Default is linear matter power. Note that the power spectrum itself is evaluated 
        at z, but the k also has a dependence on z due to the AP factor varying with z."""

        # for every fixed z and fixed mu, there is an array of k scaled by AP factors.

        matter_power = np.zeros_like(self._k_actual)
        for iz in range(self._d.nz):
            print('iz', iz)
            for imu in range(self._d.nmu): 
                p = self.get_matter_power_at_z_and_k(self._d.z[iz],
                    self._k_actual[iz, :, imu])
                matter_power[iz, :, imu] = p
        assert matter_power.shape == (self._d.nz, self._d.nk, self._d.nmu)
        return matter_power

    def _get_bnl(self):
        pass

    def _get_alpha(self):
        pass


    