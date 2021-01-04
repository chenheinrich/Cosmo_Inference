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
        self._allowed_names = ['galaxy_bis', 'Bggg_b10', 'Bggg_b20']
        (self._ik1, self._ik2, self._ik3) = self._get_ik1_ik2_ik3()
        
    def get(self, name):
        if name in self._allowed_names:
            if name not in self._state.keys():
                getattr(self, '_calc_'+name)()
            return self._state[name]
        else:
            raise NameNotAllowedError(name, self._allowed_names)

    def _get_ik1_ik2_ik3(self):
        self._triangle_specs = TriangleSpecs(self._data_spec.k)
        return self._triangle_specs.get_ik1_ik2_ik3()

    def _calc_galaxy_bis(self):
        b10 = self._get_Bggg(term_name='b10')  
        b20 = self._get_Bggg(term_name='b20')  
        self._state['galaxy_bis'] = b10 + b20
        self._state['Bggg_b10'] = b10
        self._state['Bggg_b20'] = b20
        #TODO think about how to make this useful for cobaya when only varying fnl and bias as well.

    def _get_Bggg(self, term_name='b10'):
        """3d numpy array of shape (nb, nz, ntri)"""
    
        nb = self._data_spec.nsample**3
        nz = self._data_spec.nz
        ntri = self._triangle_specs.ntri

        Bggg = np.zeros((nb, nz, ntri))
        
        for iz in range(self._data_spec.nz):
            ib = 0
            for isample1 in range(self._data_spec.nsample):
                for isample2 in range(self._data_spec.nsample):
                    for isample3 in range(self._data_spec.nsample):
                        Bggg[ib, iz, :] =  getattr(self, '_get_Bggg_' + term_name + '_at_iz')(iz, isample1, isample2, isample3)
                        ib = ib + 1

        return Bggg

    def _get_Bggg_b10_at_iz(self, iz, isample1, isample2, isample3):
        imu = 0 #TODO handle later
        bias = self._grs_ingredients.get('galaxy_bias') 
        Bmmm = self._get_Bmmm()
        Bggg_b10 = Bmmm[iz, :] * bias[isample1, iz, self._ik1, imu] * bias[isample2, iz, self._ik2, imu] * bias[isample3, iz, self._ik3, imu]
        return Bggg_b10

    def _get_Bggg_b20_at_iz(self, iz, isample1, isample2, isample3):
        (pk12, pk23, pk13) = self._get_pk12_23_13(iz)
        bias = self._grs_ingredients.get('galaxy_bias') 
        bias_20 = self._grs_ingredients.get('galaxy_bias_20') 
        
        imu = 0 #TODO handle later
        Bggg_b20 = bias[isample1, iz, self._ik1, imu] \
                        * bias[isample2, iz, self._ik2, imu] \
                        * bias_20[isample3, iz] \
                        * pk12 \
                + bias[isample1, iz, self._ik1, imu] \
                        * bias_20[isample2, iz] \
                        * bias[isample3, iz, self._ik3, imu] \
                        * pk13 \
                + bias_20[isample1, iz] \
                        * bias[isample2, iz, self._ik2, imu] \
                        * bias[isample3, iz, self._ik3, imu] \
                        * pk23 
        
        return Bggg_b20
    
    def _get_Bmmm_at_iz(self, iz):
        """Matter bispectrum at redshift index iz: 1d numpy array of shape (ntri,)"""

        (pk12, pk23, pk13) = self._get_pk12_23_13(iz)
        (alpha1, alpha2, alpha3) = self._get_alpha1_alpha2_alpha3(iz)
        (k1_array, k2_array, k3_array) = self._get_k1_k2_k3_array(iz)

        fnl = self._cosmo_par.fnl
        t1 = 2.0 * fnl * alpha3 / (alpha1 * alpha2) + \
                2.0 * self._grs_ingredients.get_F2(k1_array, k2_array, k3_array)
        t2 = 2.0 * fnl * alpha2 / (alpha1 * alpha3) + \
                2.0 * self._grs_ingredients.get_F2(k1_array, k3_array, k2_array)
        t3 = 2.0 * fnl * alpha1 / (alpha2 * alpha3) + \
                2.0 * self._grs_ingredients.get_F2(k2_array, k3_array, k1_array)
        Bmmm = t1 * pk12 + t2 * pk13 + t3 * pk23
        
        return Bmmm

    def _get_Bmmm(self):
        """2d numpy array of shape (nz, ntri)"""
        Bmmm = np.array([self._get_Bmmm_at_iz(iz) for iz in range(self._data_spec.nz)])
        return Bmmm
    
    def _get_pk12_23_13(self, iz):
        matter_power = self._grs_ingredients.get('matter_power_with_AP') # shape (nz, nk)
        (ik1, ik2, ik3) = self._triangle_specs.get_ik1_ik2_ik3()
        pk12, pk23, pk13 = self._grs_ingredients.get_matter_power_quadratic_permutations(\
            matter_power, iz, ik1, ik2, ik3) 
        return pk12, pk23, pk13

    def _get_alpha1_alpha2_alpha3(self, iz):

        imu = 0 #TODO to handle later
        alpha = self._grs_ingredients.get('alpha') # shape = (nz, nk, nmu)

        alpha1 = alpha[iz, self._ik1, imu]
        alpha2 = alpha[iz, self._ik2, imu]
        alpha3 = alpha[iz, self._ik3, imu]

        return (alpha1, alpha2, alpha3)

    def _get_k1_k2_k3_array(self, iz):
        imu = 0 #TODO to handle later
        k1_array = self._grs_ingredients.k_actual[iz, self._ik1, imu]
        k2_array = self._grs_ingredients.k_actual[iz, self._ik2, imu]
        k3_array = self._grs_ingredients.k_actual[iz, self._ik3, imu]
        return (k1_array, k2_array, k3_array)