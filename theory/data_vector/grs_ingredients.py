
from theory.cosmo.cosmo_product import CosmoProduct

#TODO do this last!

class GRSIngredients(object):

    def __init__(self, cosmo_par, survey_par):
        self._cosmo_par = cosmo_par
        self._survey_par = survey_par
        self._cosmo_product = self._get_cosmo_product()

    def _get_cosmo_product(self):
        return CosmoProduct(self._cosmo_par)

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

    def _get_bnl(self):
        pass

    def _get_alpha(self):
        pass



