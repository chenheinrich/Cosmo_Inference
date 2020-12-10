import numpy as np

from theory.data_vector.grs_ingredients import GRSIngredients
from theory.data_vector.data_spec import DataSpec, TriangleSpecs
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

    def __init__(self, cosmo_par, cosmo_par_fid, survey_par, p3d_spec):
        # TODO check that b3d_spec is instance of the right child class?
        super().__init__(cosmo_par, cosmo_par_fid, survey_par, p3d_spec)

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

    def __init__(self, cosmo_par, cosmo_par_fid, survey_par, b3d_spec):
        # TODO check that b3d_spec is instance of the right child class?
        super().__init__(cosmo_par, cosmo_par_fid, survey_par, b3d_spec)

        self._state = {}
        self._allowed_names = ['galaxy_ps', 'galaxy_transfer']
        
    def get(self, name):
        if name in self._allowed_names:
            if name not in self._state.keys():
                getattr(self, '_calc_'+name)()
            return self._state[name]
        else:
            raise NameNotAllowedError(name, self._allowed_names)

    def _calc_galaxy_bis(self):
        galaxy_bis = np.zeros(self._data_spec.shape)

        AP = self._grs_ingredients.get('AP')
        matter_power = self._grs_ingredients.get('matter_power_with_AP')
        galaxy_transfer = self.get('galaxy_transfer')

        print('AP = ', AP)

        jj = 0
        for j1 in range(self._data_spec.nsample):
            for j2 in range(j1, self._data_spec.nsample):
                galaxy_bis[jj] = AP[:, np.newaxis, np.newaxis] \
                    * matter_power \
                    * galaxy_transfer[j1, :, :, :] \
                    * galaxy_transfer[j2, :, :, :]
                jj = jj + 1
        assert jj == self._data_spec.nps

        self._state['galaxy_bis'] = galaxy_bis

    @staticmethod
    def get_matter_power_quadratic_permutations(matter_power, iz, ik1, ik2, ik3, imu1 = 0, imu2 = 0, imu3 = 0):
        
        mp = matter_power[iz,:,:]
        pk12 = mp[ik1, imu1] * mp[ik2, imu2]
        pk23 = mp[ik2, imu3] * mp[ik2, imu3]
        pk13 = mp[ik1, imu3] * mp[ik1, imu3]

        return pk12, pk23, pk13

    def get_Bmmm_of_z_and_k(self, fnl):
        #TODO figure out prefactors:
        # prefac1 = (b_lin + nl1) * (b_lin + nl2) * (b_lin + nl3)
        # use galaxy_bias to get b_lin+nl1
        alpha = self._grs_ingredients.get('alpha') # shape = (nz, nk, nmu)
        matter_power = self._grs_ingredients.get('matter_power_without_AP') # shape (nz, nk)
        k = self._data_spec.k

        triangle_specs = TriangleSpecs(self._data_spec.nk)
        tri_index_array = triangle_specs.get_tri_index_array()
            
        iz = 0 # TODO iterate over this
        ik1 = tri_index_array[:,0]
        ik2 = tri_index_array[:,1]
        ik3 = tri_index_array[:,2]

        pk12, pk23, pk13 = self.get_matter_power_quadratic_permutations(\
            matter_power, iz, ik1, ik2, ik3) 

        #TODO NEXT use above for get_F2 and alpha
        #TODO get_F2(k, ik1, ik2, ik3); what about if RSD? think through this
        t1 = 2.0 * (alpha3 / (alpha1 * alpha2)) * fnl + \
            2.0 * get_F2(k1_array, k2_array, k3_array)
        t2 = 2.0 * (alpha2 / (alpha1 * alpha3)) * fnl + \
            2.0 * get_F2(k1_array, k3_array, k2_array)
        t3 = 2.0 * (alpha1 / (alpha2 * alpha3)) * fnl + \
            2.0 * get_F2(k2_array, k3_array, k1_array)
        term1 = t1 * pk12 + t2 * pk13 + t3 * pk23
        # TODO 
        # TO CHECK: previously ans2 = prefac2 * (t1 * pmk1 * pmk2 + t2 * pmk1 * pmk3 + t3 * pmk2 * pmk3)
        # factor of 2 missing here, c.f. Eq. 5.5, unless their b10 is different than
        # Elisabeth's b2! --- CHECK LATER
        Bmmm = term1  # + term2  (wrong do not need b20 term2 here here)
        return Bmmm


        #TODO stepping through k1, k2, k3 (eventually mu1, mu2, mu3)
        # shape of Bggg: (nz, nk1=nk, nk2=nk-nk1, nk3=nk3(k1,k2))
        # let Bggg be 1-dimensional for each z, so (nz, ntri)
        # eventually when adding mu, can keep same form (nz, ntri)
        # while really we got (nz, nk1, nmu1, nk2, nmu2, nk3, nmu3)
        # if use only unique triangles w/ k3>=k2>=k1 and satisfying unique triangles

