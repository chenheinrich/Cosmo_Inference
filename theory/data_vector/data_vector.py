
from theory.data_vector.grs_ingredients import GRSIngredients
from theory.data_vector.data_spec import DataSpec

class DataVector():

    def __init__(self, cosmo_par, survey_par, data_spec):
        self._cosmo_par = cosmo_par
        self._survey_par = survey_par
        self._data_spec = data_spec

        self._grs_ingredients = self._get_grs_ingredients()

    def calculate(self):
        pass

    def save(self, fn):
        pass

    def _get_grs_ingredients(self):
        grs_ing = GRSIngredients(self._cosmo_par, self._survey_par)
        return grs_ing


class P3D(DataVector):

    def __init__(self, cosmo_par, survey_par, ps3d_spec):
        # TODO check that ps3d_spec is instance of the right child class?
        super().__init__(cosmo_par, survey_par, ps3d_spec)
        
    def calculate(self):
        #TODO to be implemented: 
        # use ps2d_specs to get z and k
        z = [0, 0.1, 0.3]
        k = [1e-4, 1e-3, 1e-2, 1e-1]
        matter_power = self._grs_ingredients.get_matter_power_for_z_and_k(z, k)
        print('matter_power = ', matter_power)


        self._galaxy_ps = None

    def get_galaxy_ps(self):
        return self._galaxy_ps


class B3D(DataVector):

    def __init__(self, cosmo_par, survey_par, bs3d_spec):
        # TODO check that bs3d_spec is instance of the right child class?
       super().__init__(cosmo_par, survey_par, bs3d_spec)

    def calculate(self):
        #TODO to be implemented: 
        # using various ingredients self._grs_ingredients to make galaxy_bis 
        self._galaxy_bis = None

    def get_galaxy_bis(self):
        return self._galaxy_bis