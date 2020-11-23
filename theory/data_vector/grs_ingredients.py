import numpy as np

from theory.cosmo.cosmo_product import CosmoProduct_FromCobayaProvider
from theory.cosmo.cosmo_product import CosmoProduct_FromCamb

#TODO do this last!

class GRSIngredients(object):

    def __init__(self, cosmo_par, cosmo_par_fid, survey_par, data_spec):
       
        self._cosmo_par = cosmo_par
        self._cosmo_par_fid = cosmo_par_fid
        self._survey_par = survey_par
        self._d = data_spec
        
        self._cosmo = self._get_cosmo_product(self._cosmo_par, self._d.z)
        self._cosmo_fid = self._get_cosmo_product(self._cosmo_par_fid, self._d.z)

        self._set_k_and_mu_actual()

    def _get_cosmo_product(self, cosmo_par, z):
        return CosmoProduct_FromCamb(cosmo_par, z)

    #TODO might want to make cosmos getter part of GRSIngredients 
    # getter too, so nobody knows about cosmos really.
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

    def get_matter_power(self):
        matter_power = self._cosmo.get_matter_power_for_z_and_k(self._d.z, self._d.k)
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
        D = self._cosmo.get_angular_diameter(z)
        D_fid = self._cosmo_fid.get_angular_diameter(z)
        return D_fid/D

    def _get_AP_para(self):
        """Returns AP factor for k parallel to line-of-sight
        in an array same size as self._d.z."""
        z = self._d.z
        H = self._cosmo.get_Hubble(z)
        H_fid = self._cosmo_fid.get_Hubble(z)
        return H/H_fid

    def _get_H0(self):
        """Returns H0 in km/s/Mpc units."""
        H0 = self._cosmo.get_H0()
        return H0

    def _get_f(self):
        f = self._cosmo.get_f(self._d.z)
        return f

    def _get_sigma8(self):
        sigma8 = self._cosmo.get_sigma8(self._d.z)
        return sigma8

    def _set_k_and_mu_actual(self):
        ap_perp = self._get_AP_perp()
        ap_para = self._get_AP_para()
        self._d.set_k_and_mu_actual(ap_perp, ap_para)

    def _calc_alpha(self):
        """Returns alpha as 3-d numpy array with shape (nz, nk, nmu) """
        initial_power = self._calc_initial_power()
        alpha = (5.0 / 3.0) \
            * np.sqrt(self.get_matter_power() / initial_power)
        assert alpha.shape == self._d.shape[1:]
        return alpha

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

        k = self._d.k_actual
        lnk = np.log(k / k_pivot_in_invMpc)

        initial_power = np.log(As) + (ns - 1.0) * lnk  \
            + 0.5 * nrun * lnk ** 2 
        initial_power = (2.0 * np.pi**2)/(k**3) * np.exp(initial_power)

        return initial_power

    def _get_bnl(self):
        pass

    def _get_alpha(self):
        pass





