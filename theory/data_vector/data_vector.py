import numpy as np

from theory.data_vector.grs_ingredients import GRSIngredients
from theory.data_vector.data_spec import DataSpec, DataSpecPowerSpectrum
from theory.data_vector.data_spec import DataSpecBispectrum, DataSpecBispectrumOriented
from theory.data_vector.triangle_spec import TriangleSpec, TriangleSpecTheta1Phi12
from theory.utils.errors import NameNotAllowedError
from theory.utils.logging import class_logger

from theory.utils.profiler import profiler

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

        assert isinstance(p3d_spec, DataSpecPowerSpectrum)

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

    """ 
    .. module:: B3D

    :Synopsis: :class:`B3D is a base class for 3D galaxy bispectrum calculation, 
        derived from the :class:`DataVector.

    Attributes: 
    
        self._state: A dictionary that stores the main results, 
            retrievable via self.get(key).

        self._allowed_names: A list of strings for a name that can be 
            called with "self.get(name)"; these are the allowed keys 
            of the dictionary "self._state".

        self._internal_state: A dictionary that stores the intermediate 
            results, not meant to be retrievable from outside the class, 
            but for resuse of intermdiate results such as matter bispectra.

        self._internal_allowed_names: A list of strings for a name that 
            can be called with "self._get(name)", these are the allowed 
            keys of the dictionary "self._internal_state".
        
    """

    def __init__(self, cosmo_par, cosmo_par_fid, survey_par, b3d_spec):
        
        """Inits B3D with cosmo_par, cosmo_par_fid, survey_par, b3d_spec.
        
        We check that b3d_spec is an instance of DataSpecBispectrum.

        We setup triangle spec.
        """ #TODO may need to simplify naming DataSpecBispectrum, and other input args

        assert isinstance(b3d_spec, DataSpecBispectrum)

        super().__init__(cosmo_par, cosmo_par_fid, survey_par, b3d_spec)

        self._state = {}
        self._allowed_names = ['galaxy_bis', 'Bggg_b10', 'Bggg_b20', \
            'Bggg_b10_primordial', 'Bggg_b10_gravitational']

        self._internal_state = {}
        self._internal_allowed_names = ['Bmmm_prim', 'Bmmm_grav']

        self._triangle_spec = self._data_spec.triangle_spec
        (self._ik1, self._ik2, self._ik3) = self._get_ik1_ik2_ik3()

    def get(self, name):
        if name in self._allowed_names:
            if name not in self._state.keys():
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

    def _get_ik1_ik2_ik3(self):
        return self._triangle_spec.get_ik1_ik2_ik3()

    def _calc_galaxy_bis(self):

        b10_grav = self._get_Bggg(term_name='b10_grav')  
        b10_prim = self._get_Bggg(term_name='b10_prim')  
        b10_tot = b10_grav + b10_prim
        b20 = self._get_Bggg(term_name='b20')  

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

        bias = self._grs_ingredients.get('galaxy_bias_without_AP') 
        Bggg_b10 = Bmmm_iz \
            * bias[isample1, iz, self._ik1] \
            * bias[isample2, iz, self._ik2] \
            * bias[isample3, iz, self._ik3]
        return Bggg_b10

    def _get_Bggg_b20_at_iz(self, iz, isample1, isample2, isample3):

        (pk12, pk23, pk13) = self._get_pk12_23_13(iz)
        bias = self._grs_ingredients.get('galaxy_bias_without_AP') 
        bias_20 = self._grs_ingredients.get('galaxy_bias_20') 
        #TODO double check this should really be of Gaussian biass
        
        Bggg_b20 = bias[isample1, iz, self._ik1] \
                        * bias[isample2, iz, self._ik2] \
                        * bias_20[isample3, iz] \
                        * pk12 \
                + bias[isample1, iz, self._ik1] \
                        * bias_20[isample2, iz] \
                        * bias[isample3, iz, self._ik3] \
                        * pk13 \
                + bias_20[isample1, iz] \
                        * bias[isample2, iz, self._ik2] \
                        * bias[isample3, iz, self._ik3] \
                        * pk23 
        
        return Bggg_b20
    
    def _calc_Bmmm(self):
        """Matter bispectrum: 2d numpy array of shape (nz, ntri)."""

        Bmmm_prim = np.zeros((self._data_spec.nz, self._triangle_spec.ntri))
        Bmmm_grav = np.zeros((self._data_spec.nz, self._triangle_spec.ntri))

        for iz in range(self._data_spec.nz):

            (pk12, pk23, pk13) = self._get_pk12_23_13(iz)
            (alpha1, alpha2, alpha3) = self._get_alpha1_alpha2_alpha3(iz)
            (k1_array, k2_array, k3_array) = self._get_k1_k2_k3_array()

            fnl = self._cosmo_par.fnl

            t1_prim = 2.0 * fnl * alpha3 / (alpha1 * alpha2) 
            t1_grav = 2.0 * self._get_F2(k1_array, k2_array, k3_array)

            t2_prim = 2.0 * fnl * alpha2 / (alpha1 * alpha3) 
            t2_grav = 2.0 * self._get_F2(k1_array, k3_array, k2_array)
            
            t3_prim = 2.0 * fnl * alpha1 / (alpha2 * alpha3) 
            t3_grav = 2.0 * self._get_F2(k2_array, k3_array, k1_array)
            
            Bmmm_prim[iz,:] = t1_prim * pk12 + t2_prim * pk13 + t3_prim * pk23
            Bmmm_grav[iz,:] = t1_grav * pk12 + t2_grav * pk13 + t3_grav * pk23

        self._internal_state['Bmmm_prim'] = Bmmm_prim
        self._internal_state['Bmmm_grav'] = Bmmm_grav

    def _get_pk12_23_13(self, iz):

        matter_power = self._grs_ingredients.get('matter_power_without_AP') 
        (ik1, ik2, ik3) = self._triangle_spec.get_ik1_ik2_ik3()

        pk12, pk23, pk13 = self._get_matter_power_quadratic_permutations(\
            matter_power, iz, ik1, ik2, ik3) 

        return pk12, pk23, pk13

    def _get_alpha1_alpha2_alpha3(self, iz):

        alpha = self._grs_ingredients.get('alpha_without_AP') # shape = (nz, nk)

        alpha1 = alpha[iz, self._ik1]
        alpha2 = alpha[iz, self._ik2]
        alpha3 = alpha[iz, self._ik3]

        return (alpha1, alpha2, alpha3)

    def _get_k1_k2_k3_array(self):

        k1_array = self._data_spec.k[self._ik1]
        k2_array = self._data_spec.k[self._ik2]
        k3_array = self._data_spec.k[self._ik3]

        return (k1_array, k2_array, k3_array)

    @staticmethod
    def _get_F2(k1, k2, k3):  
        dot_k1k2 = 0.5 * (-k1**2 - k2**2 + k3**2)
        cos = dot_k1k2 / k1 / k2
        ans = 5.0 / 7.0 + 0.5 * (k1 / k2 + k2 / k1) * cos + 2.0 / 7.0 * cos**2
        return ans

    @staticmethod
    def _get_matter_power_quadratic_permutations(matter_power_z_k, iz, ik1, ik2, ik3):
        
        mp = matter_power_z_k[iz,:]
        pk12 = mp[ik1] * mp[ik2]
        pk23 = mp[ik2] * mp[ik3]
        pk13 = mp[ik1] * mp[ik3]

        return pk12, pk23, pk13

    def get_expected_Bggg_b10_equilateral_triangles_single_tracer(self, isample=0, iz=0, imu=0):
        """Returns a 1D numpy array for expected value of Bggg b10 terms 
        for equilateral triangles in single tracer specified by isample."""

        matter_power = self._grs_ingredients.get('matter_power_with_AP')
        Pm = matter_power[iz, :]
        
        bias = self._grs_ingredients.get('galaxy_bias_without_AP') 
        b = bias[isample, iz, :]

        alpha = self._grs_ingredients.get('alpha_without_AP') 
        alpha1 = alpha[iz, np.arange(self._data_spec.nk)]

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

        bias = self._grs_ingredients.get('galaxy_bias_without_AP') 
        b_g1 = bias[isample1, iz, ik1] 
        b_g2 = bias[isample2, iz, ik2] 
        b_g3 = bias[isample3, iz, ik3]

        matter_power = self._grs_ingredients.get('matter_power_without_AP')
        Pm = matter_power[iz, :]
        pk12 = Pm[ik1] * Pm[ik2] 
        pk23 = Pm[ik2] * Pm[ik3]
        pk13 = Pm[ik1] * Pm[ik3]

        alpha = self._grs_ingredients.get('alpha_without_AP') # shape = (nz, nk, nmu)
        alpha1 = alpha[iz, ik1]
        alpha2 = alpha[iz, ik2]
        alpha3 = alpha[iz, ik3]

        k1_array = self._data_spec.k[ik1]
        k2_array = self._data_spec.k[ik2]
        k3_array = self._data_spec.k[ik3]

        fnl = self._cosmo_par.fnl
        t1 = 2.0 * fnl * alpha3 / (alpha1 * alpha2) + \
                2.0 * self._get_F2(k1_array, k2_array, k3_array)
        t2 = 2.0 * fnl * alpha2 / (alpha1 * alpha3) + \
                2.0 * self._get_F2(k1_array, k3_array, k2_array)
        t3 = 2.0 * fnl * alpha1 / (alpha2 * alpha3) + \
                2.0 * self._get_F2(k2_array, k3_array, k1_array)
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

        bias = self._grs_ingredients.get('galaxy_bias_without_AP') 
        bias_20 = self._grs_ingredients.get('galaxy_bias_20') 

        matter_power = self._grs_ingredients.get('matter_power_without_AP')
        Pm = matter_power[iz, :]
        pk12 = Pm[ik1] * Pm[ik2] 
        pk23 = Pm[ik2] * Pm[ik3]
        pk13 = Pm[ik1] * Pm[ik3]
        
        Bggg_b20 = bias[isample1, iz, ik1] \
                        * bias[isample2, iz, ik2] \
                        * bias_20[isample3, iz] \
                        * pk12 \
                + bias[isample1, iz, ik1] \
                        * bias_20[isample2, iz] \
                        * bias[isample3, iz, ik3] \
                        * pk13 \
                + bias_20[isample1, iz] \
                        * bias[isample2, iz, ik2] \
                        * bias[isample3, iz, ik3] \
                        * pk23 

        return Bggg_b20
        

class B3D_RSD(B3D):

    """No AP"""

    def __init__(self, cosmo_par, cosmo_par_fid, survey_par, b3d_spec):
        
        assert isinstance(b3d_spec, DataSpecBispectrumOriented)

        super().__init__(cosmo_par, cosmo_par_fid, survey_par, b3d_spec)

        assert isinstance(self._data_spec, DataSpecBispectrumOriented)

        self._triangle_spec = self._data_spec.triangle_spec
        print('self.triangle_spec.nori', self._triangle_spec.nori)
        print('self.triangle_spec.theta1', self._triangle_spec.theta1)

        self._f_array = self._get_f_array()
        self._fog_all = self._get_fog_all()
        (self._Z1_k1, self._Z1_k2, self._Z1_k3) = self._get_Z1_for_k1_k2_k3()

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
                            Bggg_tmp =  getattr(self, '_get_Bggg_' + term_name + '_at_iz')(iz, isample1, isample2, isample3)
                            Bggg[ib, iz, :, :] = Bggg_tmp
                            ib = ib + 1
            return Bggg
        else:
            raise NotImplementedError

    # NEW
    def _get_Bggg_optimized(self, term_name='b10'):
        """3d numpy array of shape (nb, nz, ntri)"""

        allowed_term_name = ['b10_prim', 'b10_grav', 'b20']

        if term_name in allowed_term_name:

            Bggg = getattr(self, '_get_Bggg_' + term_name)()
            return Bggg
        else:
            raise NotImplementedError

    #NEW
    def _get_Bggg_b10_prim(self):
        
        nb = self._data_spec.nsample**3
        nz = self._data_spec.nz
        ntri = self._triangle_spec.ntri
        nori = self._triangle_spec.nori

        Bggg_b10 = np.zeros((nb, nz, ntri, nori))

        for ib in range(nb):

            isamples = self._data_spec.dict_ib_to_isamples['%s'%ib]
            isample1 = isamples[0]
            isample2 = isamples[1]
            isample3 = isamples[2]

            Bggg_b10[ib, :, :, :] = self._get('Bmmm_prim')[np.newaxis, :, :, np.newaxis] \
                * self._Z1_k1[isample1, :, :, :] \
                * self._Z1_k2[isample2, :, :, :] \
                * self._Z1_k3[isample3, :, :, :] \
                * self._fog_all[ib, :, :, :]

        return Bggg_b10

    #NEW
    def _get_Bggg_b10_grav(self):
        
        nb = self._data_spec.nsample**3
        nz = self._data_spec.nz
        ntri = self._triangle_spec.ntri
        nori = self._triangle_spec.nori
        
        Bggg_b10 = np.zeros((nb, nz, ntri, nori))

        for ib in range(nb):

            isamples = self._data_spec.dict_ib_to_isamples['%s'%ib]
            isample1 = isamples[0]
            isample2 = isamples[1]
            isample3 = isamples[2]

            Bggg_b10[ib, :, :, :] = self._get('Bmmm_grav')[np.newaxis, :, :, np.newaxis] \
                * self._Z1_k1[isample1, :, :, :] \
                * self._Z1_k2[isample2, :, :, :] \
                * self._Z1_k3[isample3, :, :, :] \
                * self._fog_all[ib, :, :, :]

        return Bggg_b10

    #NEW
    def _get_Bggg_b20(self):

        bias = self._grs_ingredients.get('galaxy_bias_without_AP') 
        bias_20 = self._grs_ingredients.get('galaxy_bias_20') 

        nb = self._data_spec.nsample**3
        nz = self._data_spec.nz
        ntri = self._triangle_spec.ntri
        nori = self._triangle_spec.nori

        pks = np.array([np.transpose(np.array(self._get_pk12_23_13(iz))) for iz in range(nz)])
        pk12 = np.array(pks)[:,:,0]
        pk23 = np.array(pks)[:,:,1]
        pk13 = np.array(pks)[:,:,2]
        assert pks.shape == (nz, ntri, 3), (pks.shape, (nz, ntri, 3))
        print('pks.shape', pks.shape)

        Bggg_b20 = np.zeros((nb, nz, ntri, nori))

        for ib in range(nb):

            isamples = self._data_spec.dict_ib_to_isamples['%s'%ib]
            isample1 = isamples[0]
            isample2 = isamples[1]
            isample3 = isamples[2]

            ib = self._data_spec.dict_isamples_to_ib['%s_%s_%s'%(isample1, isample2, isample3)]
   
            Bggg_b20[ib, :, :, :] = \
                self._fog_all[ib, :, :, :] * (\
                    self._Z1_k1[isample1, :, :, :] \
                        * self._Z1_k2[isample2, :, :, :] \
                        * bias_20[isample3, :, np.newaxis, np.newaxis] \
                        * pk12[np.newaxis,:,:,np.newaxis] \
                    + self._Z1_k1[isample1, :, :, :] \
                        * bias_20[isample2, :, np.newaxis, np.newaxis] \
                        * self._Z1_k3[isample3, :, :, :] \
                        * pk13[np.newaxis,:,:,np.newaxis] \
                    + bias_20[isample1, :, np.newaxis, np.newaxis] \
                        * self._Z1_k2[isample2, :, :, :] \
                        * self._Z1_k3[isample3, :, :, :] \
                        * pk23[np.newaxis,:,:,np.newaxis]
                    )
                            
        return Bggg_b20

    def _get_Z1_for_k1_k2_k3(self):

        """Returns a tuple of 3 elements, each a 4d numpy array of shape (nb, nz, ntri, nori).
        for Z1 = (b(k)+f(z)*mu^2), where mu is a function of triangle shape and orientation."""

        bias = self._grs_ingredients.get('galaxy_bias_without_AP') 

        Z1_k1 = bias[:, :, self._ik1, np.newaxis] \
            + self._f_array[np.newaxis, :, np.newaxis, np.newaxis]\
            * self._triangle_spec.mu_array[np.newaxis, np.newaxis, :, :, 0] ** 2
        
        Z1_k2 = bias[:, :, self._ik2, np.newaxis] \
            + self._f_array[np.newaxis, :, np.newaxis, np.newaxis]\
            * self._triangle_spec.mu_array[np.newaxis, np.newaxis, :, :, 1] ** 2
        
        Z1_k3 = bias[:, :, self._ik3, np.newaxis] \
            + self._f_array[np.newaxis, :, np.newaxis, np.newaxis]\
            * self._triangle_spec.mu_array[np.newaxis, np.newaxis, :, :, 2] ** 2

        assert Z1_k1.shape == self._data_spec.shape_bis_transfer

        return (Z1_k1, Z1_k2, Z1_k3)

    def _get_fog_all(self):

        k1, k2, k3 = self._get_k1_k2_k3_array()
        sigp = self._grs_ingredients.get('sigp') #

        ib = 0
        fog = np.zeros((self._data_spec.nb, self._data_spec.nz, self._data_spec.ntri, self._data_spec.nori))
        
        for isample1 in range(self._data_spec.nsample):
            for isample2 in range(self._data_spec.nsample):
                for isample3 in range(self._data_spec.nsample):

                    #shape (ntri, nori)
                    k1mu1 = k1[:,np.newaxis] * self._triangle_spec.mu_array[:,:,0]
                    k2mu2 = k2[:,np.newaxis] * self._triangle_spec.mu_array[:,:,1]
                    k3mu3 = k3[:,np.newaxis] * self._triangle_spec.mu_array[:,:,2]         

                    #shape (nz, ntri, nori)
                    sigp_kmu_squared = (k1mu1[np.newaxis, :, :] * sigp[isample1, :, np.newaxis, np.newaxis])**2 \
                        + (k2mu2[np.newaxis, :, :] * sigp[isample2, :, np.newaxis, np.newaxis]) ** 2 \
                        + (k3mu3[np.newaxis, :, :] * sigp[isample3, :, np.newaxis, np.newaxis]) ** 2

                    fog[ib,:,:,:] = np.exp( -0.5 * (sigp_kmu_squared))
                    ib = ib + 1

            return fog

    #OLD
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

    #OLD
    def _get_Bggg_b10_prim_at_iz(self, iz, isample1, isample2, isample3):
       
        return self._get_Bggg_b10_with_Bmmm_at_iz(\
            iz, isample1, isample2, isample3, self._get('Bmmm_prim')[iz,:])

    #OLD
    def _get_Bggg_b10_grav_at_iz(self, iz, isample1, isample2, isample3):

        return self._get_Bggg_b10_with_Bmmm_at_iz(\
            iz, isample1, isample2, isample3, self._get('Bmmm_grav')[iz,:])

    #OLD
    def _get_Bggg_b10_with_Bmmm_at_iz(self, iz, isample1, isample2, isample3, Bmmm_iz):

        bias = self._grs_ingredients.get('galaxy_bias_without_AP') 
        
        ntri = self._triangle_spec.ntri
        nori = self._triangle_spec.nori

        Bggg_b10 = np.zeros((ntri, nori))

        f_of_z = self._f_array[iz]

        for iori in range(self._triangle_spec.nori):

            mu1 = self._triangle_spec.mu_array[:, iori, 0]
            mu2 = self._triangle_spec.mu_array[:, iori, 1]
            mu3 = self._triangle_spec.mu_array[:, iori, 2]
            assert mu1.size == self._ik1.size

            Z1_k1 = bias[isample1, iz, self._ik1] + f_of_z * mu1 ** 2
            Z1_k2 = bias[isample2, iz, self._ik2] + f_of_z * mu2 ** 2
            Z1_k3 = bias[isample3, iz, self._ik3] + f_of_z * mu3 ** 2

            #TODO could factor Z1, Z2, Z3 into a separate function 
            # if profiling determs it to be bottle neck.

            #fog = self._get_fog(iz, isample1, isample2, isample3, k1*mu1, k2*mu2, k3*mu3)

            ib = self._data_spec.dict_isamples_to_ib['%s_%s_%s'%(isample1, isample2, isample3)]
            fog = self._fog_all[ib, iz, :, iori]

            Bggg_b10[:, iori] = Bmmm_iz * Z1_k1 * Z1_k2 * Z1_k3 * fog \
                * bias[isample1, iz, self._ik1] \
                * bias[isample2, iz, self._ik2] \
                * bias[isample3, iz, self._ik3] \

        return Bggg_b10

    def _get_Bggg_b20_at_iz(self, iz, isample1, isample2, isample3):
        (pk12, pk23, pk13) = self._get_pk12_23_13(iz)
        bias = self._grs_ingredients.get('galaxy_bias_without_AP') 
        bias_20 = self._grs_ingredients.get('galaxy_bias_20') 
        
        ntri = self._triangle_spec.ntri
        nori = self._triangle_spec.nori

        k1, k2, k3 = self._get_k1_k2_k3_array()

        Bggg_b20 = np.zeros((ntri, nori))

        f_of_z = self._f_array[iz]

        for iori in range(self._triangle_spec.nori):

            mu1 = self._triangle_spec.mu_array[:, iori, 0]
            mu2 = self._triangle_spec.mu_array[:, iori, 1]
            mu3 = self._triangle_spec.mu_array[:, iori, 2]
            assert mu1.size == self._ik1.size

            Z1_k1 = bias[isample1, iz, self._ik1] + f_of_z * mu1 ** 2
            Z1_k2 = bias[isample2, iz, self._ik2] + f_of_z * mu2 ** 2
            Z1_k3 = bias[isample3, iz, self._ik3] + f_of_z * mu3 ** 2

            #TODO could factor Z1, Z2, Z3 into a separate function 
            # if profiling determs it to be bottle neck.

            #fog = self._get_fog(iz, isample1, isample2, isample3, k1*mu1, k2*mu2, k3*mu3)

            ib = self._data_spec.dict_isamples_to_ib['%s_%s_%s'%(isample1, isample2, isample3)]
            fog = self._fog_all[ib, iz, :, iori]

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
        
    def _get_f_array(self):

        z_array = self._data_spec.z
        debug_f = self._data_spec._debug_f_of_z

        if debug_f is not None:
            f_array = debug_f * np.ones_like(z_array)
        else:
            f_array = self._grs_ingredients._get_f()

        self.logger.debug('f_array = {}'.format(f_array))

        return f_array