import numpy as np

from theory.data_vector.grs_ingredients import GRSIngredients
from theory.data_vector.data_spec import DataSpec, DataSpecBispectrumOriented, TriangleSpec, TriangleSpecTheta1Phi12
from theory.utils.errors import NameNotAllowedError
from theory.utils.logging import class_logger

class DataVector():

    def __init__(self, cosmo_par, cosmo_par_fid, survey_par, data_spec):

        self.logger = class_logger(self)

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
        self._allowed_names = ['galaxy_bis', 'Bggg_b10', 'Bggg_b20', \
            'Bggg_b10_primordial', 'Bggg_b10_gravitational']

        self._internal_state = {}
        self._internal_allowed_names = ['Bmmm_prim', 'Bmmm_grav']

        self._triangle_spec = self._get_triangle_spec()
        (self._ik1, self._ik2, self._ik3) = self._get_ik1_ik2_ik3()

    def get(self, name):
        if name in self._allowed_names:
            if name not in self._state.keys():
                #TODO all terms are calculated when either one is asked for
                # may want to simplify since now b10_primordial and gravitational repeat the Bggg_b10 calculation
                if name in ['galaxy_bis', 'Bggg_b10', 'Bggg_b20', 'Bggg_b10_primordial', 'Bggg_b10_gravitational']:
                    getattr(self, '_calc_'+'galaxy_bis')() 
                else:
                    getattr(self, '_calc_'+name)() 
            return self._state[name]
        else:
            raise NameNotAllowedError(name, self._allowed_names)

    def _get(self, name):
        """"Private getter for intermediate states"""
        if name in self._internal_allowed_names:
            if name not in self._internal_state.keys():
                if name in ['Bmmm_prim', 'Bmmm_grav']:
                    self._calc_Bmmm()
            return self._internal_state[name]
        else:
            raise NameNotAllowedError(name, self._internal_allowed_names)
        return 

    def _get_triangle_spec(self):
        return self._data_spec.triangle_spec

    def _get_ik1_ik2_ik3(self):
        return self._triangle_spec.get_ik1_ik2_ik3()

    def _calc_galaxy_bis(self):

        b10_grav = self._get_Bggg(term_name='b10_grav')  
        b10_prim = self._get_Bggg(term_name='b10_prim')  
        b10_tot = b10_grav + b10_prim
        b20 = self._get_Bggg(term_name='b20')  

        #print('b10_grav', b10_grav)
        #print('b10_prim', b10_prim)

        self._state['Bggg_b10_gravitational'] = b10_grav
        self._state['Bggg_b10_primordial'] = b10_prim
        self._state['Bggg_b10'] = b10_tot
        self._state['Bggg_b20'] = b20
        self._state['galaxy_bis'] = b10_tot + b20

    def _get_Bggg(self, term_name='b10'):
        """3d numpy array of shape (nb, nz, ntri)"""
    
        nb = self._data_spec.nsample**3
        nz = self._data_spec.nz
        ntri = self._triangle_spec.ntri

        Bggg = np.zeros((nb, nz, ntri))
        
        for iz in range(self._data_spec.nz):
            ib = 0
            for isample1 in range(self._data_spec.nsample):
                for isample2 in range(self._data_spec.nsample):
                    for isample3 in range(self._data_spec.nsample):
                        Bggg[ib, iz, :] =  getattr(self, '_get_Bggg_' + term_name + '_at_iz')(iz, isample1, isample2, isample3)
                        ib = ib + 1

        return Bggg

    def _get_Bggg_b10_prim_at_iz(self, iz, isample1, isample2, isample3):
       
        return self._get_Bggg_b10_with_Bmmm_at_iz(\
            iz, isample1, isample2, isample3, self._get('Bmmm_prim')[iz,:])

    def _get_Bggg_b10_grav_at_iz(self, iz, isample1, isample2, isample3):

        return self._get_Bggg_b10_with_Bmmm_at_iz(\
            iz, isample1, isample2, isample3, self._get('Bmmm_grav')[iz,:])

    def _get_Bggg_b10_with_Bmmm_at_iz(self, iz, isample1, isample2, isample3, Bmmm_iz):

        imu = 0 #TODO handle later
        bias = self._grs_ingredients.get('galaxy_bias') 
        Bggg_b10 = Bmmm_iz \
            * bias[isample1, iz, self._ik1, imu] \
            * bias[isample2, iz, self._ik2, imu] \
            * bias[isample3, iz, self._ik3, imu]
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
    
    def _calc_Bmmm(self):
        """Matter bispectrum: 2d numpy array of shape (nz, ntri)."""

        Bmmm_prim = np.zeros((self._data_spec.nz, self._data_spec.triangle_spec.ntri))
        Bmmm_grav = np.zeros((self._data_spec.nz, self._data_spec.triangle_spec.ntri))

        for iz in range(self._data_spec.nz):

            (pk12, pk23, pk13) = self._get_pk12_23_13(iz)
            (alpha1, alpha2, alpha3) = self._get_alpha1_alpha2_alpha3(iz)
            (k1_array, k2_array, k3_array) = self._get_k1_k2_k3_array(iz)

            fnl = self._cosmo_par.fnl

            t1_prim = 2.0 * fnl * alpha3 / (alpha1 * alpha2) 
            t1_grav = 2.0 * self._grs_ingredients.get_F2(k1_array, k2_array, k3_array)

            t2_prim = 2.0 * fnl * alpha2 / (alpha1 * alpha3) 
            t2_grav = 2.0 * self._grs_ingredients.get_F2(k1_array, k3_array, k2_array)
            
            t3_prim = 2.0 * fnl * alpha1 / (alpha2 * alpha3) 
            t3_grav = 2.0 * self._grs_ingredients.get_F2(k2_array, k3_array, k1_array)
            
            Bmmm_prim[iz,:] = t1_prim * pk12 + t2_prim * pk13 + t3_prim * pk23
            Bmmm_grav[iz,:] = t1_grav * pk12 + t2_grav * pk13 + t3_grav * pk23

        self._internal_state['Bmmm_prim'] = Bmmm_prim
        self._internal_state['Bmmm_grav'] = Bmmm_grav

    def _get_pk12_23_13(self, iz):
        matter_power = self._grs_ingredients.get('matter_power_with_AP') # shape (nz, nk)
        (ik1, ik2, ik3) = self._triangle_spec.get_ik1_ik2_ik3()
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

    def get_expected_Bggg_b10_equilateral_triangles_single_tracer(self, isample=0, iz=0, imu=0):
        """Returns a 1D numpy array for expected value of Bggg b10 terms 
        for equilateral triangles in single tracer specified by isample."""

        matter_power = self._grs_ingredients.get('matter_power_with_AP')
        Pm = matter_power[iz, :, imu]
        
        bias = self._grs_ingredients.get('galaxy_bias') 
        b = bias[isample, iz, :, imu]

        alpha = self._grs_ingredients.get('alpha') 
        imu = 0 # TODO handle later
        alpha1 = alpha[iz, np.arange(self._data_spec.nk), imu]

        fnl = self._cosmo_par.fnl
        
        F2_equilateral = 0.2857142857142857
        Bmmm_equilateral = 3.0 * (2.0 * F2_equilateral * Pm ** 2)
        Bmmm_equilateral += 3.0 * (2.0 * fnl / alpha1 * Pm ** 2)
        
        Bggg_b10_equilateral_triangles_single_tracer = b ** 3 * Bmmm_equilateral 

        return Bggg_b10_equilateral_triangles_single_tracer

    def get_expected_Bggg_b10_general(self, \
        isample1, isample2, isample3, iz, itri=None, imu=0): 

        if itri is None:
            ik1 = self._ik1
            ik2 = self._ik2
            ik3 = self._ik3
        else:
            iks = self._triangle_spec.tri_index_array[itri]
            ik1 = iks[0]
            ik2 = iks[1]
            ik3 = iks[2]

        bias = self._grs_ingredients.get('galaxy_bias') 
        b_g1 = bias[isample1, iz, ik1, imu] 
        b_g2 = bias[isample2, iz, ik2, imu] 
        b_g3 = bias[isample3, iz, ik3, imu]

        matter_power = self._grs_ingredients.get('matter_power_with_AP')
        Pm = matter_power[iz, :, imu]
        pk12 = Pm[ik1] * Pm[ik2] 
        pk23 = Pm[ik2] * Pm[ik3]
        pk13 = Pm[ik1] * Pm[ik3]

        alpha = self._grs_ingredients.get('alpha') # shape = (nz, nk, nmu)
        alpha1 = alpha[iz, ik1, imu]
        alpha2 = alpha[iz, ik2, imu]
        alpha3 = alpha[iz, ik3, imu]

        k1_array = self._data_spec.k[ik1]
        k2_array = self._data_spec.k[ik2]
        k3_array = self._data_spec.k[ik3]

        fnl = self._cosmo_par.fnl
        t1 = 2.0 * fnl * alpha3 / (alpha1 * alpha2) + \
                2.0 * self._grs_ingredients.get_F2(k1_array, k2_array, k3_array)
        t2 = 2.0 * fnl * alpha2 / (alpha1 * alpha3) + \
                2.0 * self._grs_ingredients.get_F2(k1_array, k3_array, k2_array)
        t3 = 2.0 * fnl * alpha1 / (alpha2 * alpha3) + \
                2.0 * self._grs_ingredients.get_F2(k2_array, k3_array, k1_array)
        Bmmm = t1 * pk12 + t2 * pk13 + t3 * pk23

        Bggg_b10 = Bmmm * b_g1 * b_g2 * b_g3

        return Bggg_b10
    
    def get_expected_Bggg_b20_general(self, \
        isample1, isample2, isample3, iz, itri=None, imu=0): 

        if itri is None:
            ik1 = self._ik1
            ik2 = self._ik2
            ik3 = self._ik3
        else:
            iks = self._triangle_spec.tri_index_array[itri]
            ik1 = iks[0]
            ik2 = iks[1]
            ik3 = iks[2]

        bias = self._grs_ingredients.get('galaxy_bias') 
        bias_20 = self._grs_ingredients.get('galaxy_bias_20') 

        matter_power = self._grs_ingredients.get('matter_power_with_AP')
        Pm = matter_power[iz, :, imu]
        pk12 = Pm[ik1] * Pm[ik2] 
        pk23 = Pm[ik2] * Pm[ik3]
        pk13 = Pm[ik1] * Pm[ik3]
        
        Bggg_b20 = bias[isample1, iz, ik1, imu] \
                        * bias[isample2, iz, ik2, imu] \
                        * bias_20[isample3, iz] \
                        * pk12 \
                + bias[isample1, iz, ik1, imu] \
                        * bias_20[isample2, iz] \
                        * bias[isample3, iz, ik3, imu] \
                        * pk13 \
                + bias_20[isample1, iz] \
                        * bias[isample2, iz, ik2, imu] \
                        * bias[isample3, iz, ik3, imu] \
                        * pk23 

        return Bggg_b20
        

class B3D_RSD(B3D): #No AP

    def __init__(self, cosmo_par, cosmo_par_fid, survey_par, b3d_spec):
        # TODO check that b3d_spec is instance of the right child class?
        super().__init__(cosmo_par, cosmo_par_fid, survey_par, b3d_spec)

        assert isinstance(self._data_spec, DataSpecBispectrumOriented)
        self._triangle_spec = self._data_spec.triangle_spec
        print('self.triangle_spec.nori', self._triangle_spec.nori)
        print('self.triangle_spec.theta1', self._triangle_spec.theta1)

    def _get_Bggg(self, term_name='b10'):
        """3d numpy array of shape (nb, nz, ntri)"""

        allowed_term_name = ['b10_prim', 'b10_grav', 'b20']

        if term_name in allowed_term_name:

            nb = self._data_spec.nsample**3
            nz = self._data_spec.nz
            ntri = self._triangle_spec.ntri
            nori = self._triangle_spec.nori

            Bggg = np.zeros((nb, nz, ntri, nori))
            
            for iz in range(self._data_spec.nz):
                ib = 0
                for isample1 in range(self._data_spec.nsample):
                    for isample2 in range(self._data_spec.nsample):
                        for isample3 in range(self._data_spec.nsample):
                            Bggg[ib, iz, :, :] =  getattr(self, '_get_Bggg_' + term_name + '_at_iz')(iz, isample1, isample2, isample3)
                            ib = ib + 1
            return Bggg
        else:
            raise NotImplementedError

    def _get_Bggg_b10_with_Bmmm_at_iz(self, iz, isample1, isample2, isample3, Bmmm_iz):

        ntri = self._triangle_spec.ntri
        nori = self._triangle_spec.nori

        Bggg_b10 = np.zeros((ntri, nori))

        imu = 0 #TODO handle later

        k1, k2, k3 = self._get_k1_k2_k3_array(iz)

        bias = self._grs_ingredients.get('galaxy_bias') 
        f_of_z = self._get_f_of_z(iz) 

        for iori in range(self._triangle_spec.nori):
            
            mu1 = self._triangle_spec.mu_array[:, iori, 0]
            mu2 = self._triangle_spec.mu_array[:, iori, 1]
            mu3 = self._triangle_spec.mu_array[:, iori, 2]
            assert mu1.size == self._ik1.size
            
            Z1_k1 = bias[isample1, iz, self._ik1, imu] + f_of_z * mu1 ** 2
            Z1_k2 = bias[isample2, iz, self._ik2, imu] + f_of_z * mu2 ** 2
            Z1_k3 = bias[isample3, iz, self._ik3, imu] + f_of_z * mu3 ** 2

            fog = self._get_fog(iz, isample1, isample2, isample3, k1*mu1, k2*mu2, k3*mu3)

            Bggg_b10[:, iori] = Bmmm_iz *  Z1_k1 * Z1_k2 * Z1_k3 * fog

        return Bggg_b10

    def _get_Bggg_b20_at_iz(self, iz, isample1, isample2, isample3):
        (pk12, pk23, pk13) = self._get_pk12_23_13(iz)
        bias = self._grs_ingredients.get('galaxy_bias') 
        bias_20 = self._grs_ingredients.get('galaxy_bias_20') 
        
        imu = 0 #TODO handle later

        ntri = self._triangle_spec.ntri
        nori = self._triangle_spec.nori

        k1, k2, k3 = self._get_k1_k2_k3_array(iz)

        Bggg_b20 = np.zeros((ntri, nori))

        for iori in range(self._triangle_spec.nori):

            mu1 = self._triangle_spec.mu_array[:, iori, 0]
            mu2 = self._triangle_spec.mu_array[:, iori, 1]
            mu3 = self._triangle_spec.mu_array[:, iori, 2]
            assert mu1.size == self._ik1.size

            f_of_z = self._get_f_of_z(iz)
            Z1_k1 = bias[isample1, iz, self._ik1, imu] + f_of_z * mu1 ** 2
            Z1_k2 = bias[isample2, iz, self._ik2, imu] + f_of_z * mu2 ** 2
            Z1_k3 = bias[isample3, iz, self._ik3, imu] + f_of_z * mu3 ** 2

            #TODO need to factor Z1, Z2, Z3 into grs ingredients

            fog = self._get_fog(iz, isample1, isample2, isample3, k1*mu1, k2*mu2, k3*mu3)

            Bggg_b20[:, iori] = fog * (
                    Z1_k1 \
                            * Z1_k2 \
                            * bias_20[isample3, iz] \
                            * pk12 \
                    + Z1_k1 \
                            * bias_20[isample2, iz] \
                            * Z1_k3 \
                            * pk13 \
                    + bias_20[isample1, iz] \
                            * Z1_k2 \
                            * Z1_k3 \
                            * pk23
                    )
                            
        return Bggg_b20

    def _get_fog(self, iz, isample1, isample2, isample3, k1mu1, k2mu2, k3mu3):

        if self._data_spec._debug_sigp is not None:
            sigp = self._data_spec._debug_sigp
            sigp1 = sigp2 = sigp3 = sigp
        else:
            sigp = self._grs_ingredients.get('sigp')
            sigp1, sigp2, sigp3 = (\
            sigp[isample1, iz], \
            sigp[isample2, iz], \
            sigp[isample3, iz]
            )

        self.logger.debug('using sigp1, sigp2, sigp3 = {}, {}, {}'.format(sigp1, sigp2, sigp3))
        
        sigp_kmu_squared = (k1mu1*sigp1)**2 + (k2mu2*sigp2) ** 2 + (k3mu3*sigp3) ** 2
        assert sigp_kmu_squared.shape == k1mu1.shape

        fog = np.exp(- 0.5 * (sigp_kmu_squared))

        return fog
        
    def _get_f_of_z(self, iz):

        if self._data_spec._debug_f_of_z is not None:
            f_of_z = self._data_spec._debug_f_of_z
        else:
            f_of_z = self._grs_ingredients._get_f()[iz] 

        self.logger.debug('f_of_z = {}'.format(f_of_z))

        return f_of_z