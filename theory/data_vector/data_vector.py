import numpy as np

from theory.data_vector.grs_ingredients import GRSIngredients
from theory.data_vector.data_spec import DataSpec
from theory.utils.errors import NameNotAllowedError

class DataVector():

    def __init__(self, cosmo_par, cosmo_par_fid, survey_par, data_spec):
        self._cosmo_par = cosmo_par
        self._cosmo_par_fid = cosmo_par_fid
        self._survey_par = survey_par
        self._data_spec = data_spec

        self._grs_ingredients = self._get_grs_ingredients()

    def calculate(self):
        pass

    def save(self, fn):
        pass

    def _get_grs_ingredients(self):
        grs_ing = GRSIngredients(self._cosmo_par, self._cosmo_par_fid, self._survey_par, self._data_spec)
        return grs_ing


class P3D(DataVector):

    def __init__(self, cosmo_par, cosmo_par_fid, survey_par, ps3d_spec):
        # TODO check that ps3d_spec is instance of the right child class?
        super().__init__(cosmo_par, cosmo_par_fid, survey_par, ps3d_spec)

        self._state = {}
        self._allowed_names = ['galaxy_ps', 'galaxy_transfer']
        
    def get(self, name):
        if name in self._allowed_names:
            if name not in self._state.keys():
                getattr(self, '_calc_'+name)()
            return self._state[name]
        else:
            raise NameNotAllowedError(name, self._allowed_names)

    def _calc_galaxy_ps(self):
        galaxy_ps = np.zeros(self._data_spec.shape)

        AP = self._grs_ingredients.get('AP')
        matter_power = self._grs_ingredients.get('matter_power_with_AP')
        galaxy_transfer = self.get('galaxy_transfer')

        print('AP = ', AP)

        jj = 0
        for j1 in range(self._data_spec.nsample):
            for j2 in range(j1, self._data_spec.nsample):
                galaxy_ps[jj] = AP[:, np.newaxis, np.newaxis] \
                    * matter_power \
                    * galaxy_transfer[j1, :, :, :] \
                    * galaxy_transfer[j2, :, :, :]
                jj = jj + 1
        assert jj == self._data_spec.nps

        self._state['galaxy_ps'] = galaxy_ps

    def _calc_galaxy_transfer(self):

        bias = self._grs_ingredients.get('galaxy_bias')
        kaiser = self._grs_ingredients.get('kaiser')
        fog = self._grs_ingredients.get('fog')

        galaxy_transfer = bias * kaiser * fog

        self._state['galaxy_transfer'] = galaxy_transfer

class B3D(DataVector):

    def __init__(self, cosmo_par, cosmo_par_fid, survey_par, bs3d_spec):
        # TODO check that bs3d_spec is instance of the right child class?
       super().__init__(cosmo_par, cosmo_par_fid, survey_par, bs3d_spec)

    def calculate(self):
        #TODO to be implemented: 
        # using various ingredients self._grs_ingredients to make galaxy_bis 
        self._galaxy_bis = None

    def get_galaxy_bis(self):
        return self._galaxy_bis