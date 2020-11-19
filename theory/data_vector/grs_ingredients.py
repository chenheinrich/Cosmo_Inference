
from theory.cosmo.cosmo_product import CosmoProduct_FromCobayaProvider
from theory.cosmo.cosmo_product import CosmoProduct_FromCambCalculator
from theory.cosmo.cosmo_calculator import CambCalculator

#TODO do this last!

class GRSIngredients(object):

    def __init__(self, cosmo_par, survey_par):
        self._cosmo_par = cosmo_par
        self._survey_par = survey_par
        self._cosmo_product = self._get_cosmo_product()

    def _get_cosmo_product(self):
        #TODO might need to place this somewhere else
        camb_calc = CambCalculator(self._cosmo_par)
        return CosmoProduct_FromCambCalculator(self._cosmo_par, camb_calc)

    #TODO might want to make cosmo_products getter part of GRSIngredients 
    # getter too, so nobody knows about cosmo_products really.
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

    def get_matter_power_for_z_and_k(self, z, k):
        matter_power = self._cosmo_product.get_matter_power_for_z_and_k(z, k)
        angular_diameter = self._cosmo_product.get_angular_diameter_array(z)
        Hubble = self._cosmo_product.get_Hubble_array(z)

        sigma8 = self._cosmo_product.get_sigma8_array()
        f = self._cosmo_product.get_f_array()

        print('angular_diameter', angular_diameter)
        print('Hubble', Hubble)

        print('sigma8', sigma8)
        print('f', f)
        
        return matter_power

    def _get_bnl(self):
        pass

    def _get_alpha(self):
        pass





