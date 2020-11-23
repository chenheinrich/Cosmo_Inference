
from theory.cosmo.cosmo_product import CosmoProduct_FromCobayaProvider
from theory.cosmo.cosmo_product import CosmoProduct_FromCamb

#TODO do this last!

class GRSIngredients(object):

    def __init__(self, cosmo_par, cosmo_par_fid, survey_par, data_spec):
       
        self._cosmo_par = cosmo_par
        self._cosmo_par_fid = cosmo_par_fid
        self._survey_par = survey_par
        self._data_spec = data_spec

        self._z = self._survey_par.get_zmid_array()
        self._k = self._data_spec.k
        
        self._cosmo = self._get_cosmo_product(self._cosmo_par, self._z)
        self._cosmo_fid = self._get_cosmo_product(self._cosmo_par_fid, self._z)

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
        matter_power = self._cosmo.get_matter_power_for_z_and_k(self._z, self._k)
        return matter_power

    def get_AP(self):
        D = self._cosmo.get_angular_diameter(self._z)
        H = self._cosmo.get_Hubble(self._z)
        
        D_fid = self._cosmo_fid.get_angular_diameter(self._z)
        H_fid = self._cosmo_fid.get_Hubble(self._z)

        AP = (D_fid/D)**2 * (H/H_fid)
        return AP

    def _get_H0(self):
        """Returns H0 in km/s/Mpc units."""
        H0 = self._cosmo.get_H0()
        return H0

    def _get_f(self):
        f = self._cosmo.get_f(self._z)
        return f

    def _get_sigma8(self):
        sigma8 = self._cosmo.get_sigma8(self._z)
        return sigma8

    def _get_bnl(self):
        pass

    def _get_alpha(self):
        pass





