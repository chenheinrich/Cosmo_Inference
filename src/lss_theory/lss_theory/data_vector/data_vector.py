import numpy as np

from lss_theory.utils.errors import NameNotAllowedError
from lss_theory.utils.logging import class_logger
from lss_theory.data_vector.data_spec import Bispectrum3DRSDSpec_MurMuphi

from lss_theory.utils.profiler import profiler

class DataVector():

    def __init__(self, grs_ingredients, survey_par, data_spec):

        self.logger = class_logger(self)

        self._survey_par = survey_par
        self._data_spec = data_spec

        self._grs_ingredients = grs_ingredients

        self._state = {}
        self._setup_allowed_names()

    def _setup_allowed_names(self):
        self._allowed_names = []

    def get(self, name):
        """Call self._calc_<name> to calculate and store quantity <name>
        in self._state[name], if it hasn't been calculated before; 
        otherwise, return directly from what is stored in self._state.
        """
        if name in self._allowed_names:
            if name not in self._state.keys():
                getattr(self, '_calc_'+name)()
            return self._state[name]
        else:
            raise NameNotAllowedError(name, self._allowed_names)

class PowerSpectrum3D(DataVector):

    def __init__(self, grs_ingredients, survey_par, p3d_spec):

        #TODO to debug why this doesn't work
        #if not isinstance(p3d_spec, PowerSpectrum3DSpec):
        #    message = '3rd input to PowerSpectrum3DSpec \
        #        must be an instance of PowerSpectrum3DSpec'
        #    raise ValueError(message) 

        super().__init__(grs_ingredients, survey_par, p3d_spec)

    def _setup_allowed_names(self):
        self._allowed_names = [\
            'galaxy_ps', \
            'galaxy_ps_without_AP',\
            'galaxy_ps_without_AP_no_fog_no_kaiser', \
            'galaxy_transfer', \
            'galaxy_transfer_without_AP', \
            'galaxy_transfer_without_AP_no_fog_no_kaiser'\
            ]

    def get_ips(self, isample1, isample2):
        """Returns index of power spectrum given indices of 2 galaxy samples"""
        return self._data_spec.get_ips(isample1, isample2)

    def get_isamples(self, ips):
        """Returns a tuple of 2 galaxy sample indices given index of power spectrum"""
        return self._data_spec.get_isamples(ips)

    def _calc_galaxy_ps(self):
        galaxy_ps = np.zeros(self._data_spec.shape)

        AP = self._grs_ingredients.get('AP')
        matter_power = self._grs_ingredients.get('matter_power_with_AP')
        galaxy_transfer = self.get('galaxy_transfer')

        jj = 0
        for j1 in range(self._data_spec.nsample):
            for j2 in range(j1, self._data_spec.nsample):
                galaxy_ps[jj] = AP[:, np.newaxis, np.newaxis] \
                    * matter_power \
                    * galaxy_transfer[j1, :, :, :] \
                    * galaxy_transfer[j2, :, :, :]
                jj = jj + 1
        assert jj == self._data_spec.nps, (jj, self._data_spec.nps)

        self._state['galaxy_ps'] = galaxy_ps
    
    def _calc_galaxy_transfer(self):

        bias = self._grs_ingredients.get('galaxy_bias')
        kaiser = self._grs_ingredients.get('kaiser')
        fog = self._grs_ingredients.get('fog')

        galaxy_transfer = bias * kaiser * fog

        self._state['galaxy_transfer'] = galaxy_transfer

    def _calc_galaxy_ps_without_AP(self):
        galaxy_ps_without_AP = np.zeros(self._data_spec.shape)

        #(nz, nk)
        matter_power_without_AP = self._grs_ingredients.get('matter_power_without_AP')
        galaxy_transfer_without_AP = self.get('galaxy_transfer_without_AP')

        jj = 0
        for j1 in range(self._data_spec.nsample):
            for j2 in range(j1, self._data_spec.nsample):
                galaxy_ps_without_AP[jj] = \
                    matter_power_without_AP[np.newaxis, :, :, np.newaxis] \
                    * galaxy_transfer_without_AP[j1, :, :, :] \
                    * galaxy_transfer_without_AP[j2, :, :, :]
                jj = jj + 1
        assert jj == self._data_spec.nps, (jj, self._data_spec.nps)

        self._state['galaxy_ps_without_AP'] = galaxy_ps_without_AP

    def _calc_galaxy_transfer_without_AP(self):

        bias_without_AP = self._grs_ingredients.get('galaxy_bias_without_AP') #(nsample, nz, nk)
        kaiser_without_AP = self._grs_ingredients.get('kaiser_without_AP') #(nsample, nz, nk, nmu)
        fog = self._grs_ingredients.get('fog') #not affected by AP

        galaxy_transfer_without_AP = bias_without_AP[:, :, :, np.newaxis] \
            * kaiser_without_AP * fog

        self._state['galaxy_transfer_without_AP'] = galaxy_transfer_without_AP


    def _calc_galaxy_ps_without_AP_no_fog_no_kaiser(self):
        """Calculates and stores 3d numpy array of shape (nsample, nz, nk)
        for the galaxy power spectrum without AP effects and
        without fog and Kaiser terms, so there is no mu dependence, 
        suitatble for e.g. bispectrum covariance calculations where mu
        values needed may be different than that specified in here."""
        galaxy_ps_without_AP = np.zeros(self._data_spec.shape[:3])

        #(nz, nk)
        matter_power_without_AP = self._grs_ingredients.get('matter_power_without_AP')
        galaxy_transfer_without_AP = self.get('galaxy_transfer_without_AP_no_fog_no_kaiser')

        jj = 0
        for j1 in range(self._data_spec.nsample):
            for j2 in range(j1, self._data_spec.nsample):
                galaxy_ps_without_AP[jj] = \
                    matter_power_without_AP[np.newaxis, :, :] \
                    * galaxy_transfer_without_AP[j1, :, :] \
                    * galaxy_transfer_without_AP[j2, :, :]
                jj = jj + 1
        assert jj == self._data_spec.nps, (jj, self._data_spec.nps)

        self._state['galaxy_ps_without_AP_no_fog_no_kaiser'] = galaxy_ps_without_AP

    def _calc_galaxy_transfer_without_AP_no_fog_no_kaiser(self):

        bias_without_AP = self._grs_ingredients.get('galaxy_bias_without_AP') #(nsample, nz, nk)
        self._state['galaxy_transfer_without_AP_no_fog_no_kaiser'] = bias_without_AP

class Bispectrum3DBase(DataVector):

    """ 
    .. module:: Bispectrum3DBase

    :Synopsis: :class:`Bispectrum3DBase is a base class for 3D galaxy bispectrum calculation, 
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

    def __init__(self, grs_ingredients, survey_par, b3d_base_spec):
        
        """Inits Bispectrum3DBase with survey_par, b3d_base_spec.
        
        Checks:
        - We check that b3d_base_spec is an instance of Bispectrum3DBaseSpec.

        Setups:
        - We setup self._triangle_spec, ik1, ik2, ik3
        """ 

        #TODO decide what to do
        #assert isinstance(b3d_base_spec, Bispectrum3DBaseSpec)

        super().__init__(grs_ingredients, survey_par, b3d_base_spec)

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

    def get_ips(self, isample1, isample2, isample3):
        """Returns index of bispectrum given indices of 3 galaxy samples"""
        return self._data_spec.get_ips(isample1, isample2, isample3)

    def get_isamples(self, ib):
        """Returns a tuple of 3 galaxy sample indices given index of bispectrum"""
        return self._data_spec.get_isamples(ib)

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
        ntri = self._data_spec.ntri

        Bggg = np.zeros(self._data_spec.shape)
        
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

            fnl = self._grs_ingredients.get('fnl')

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


#changes:
# isamples = dict[...] --> get_isamples(ib)
# nb = nsamples**3 --> dataspec.nb
# 
class Bispectrum3DRSD(Bispectrum3DBase):

    """No AP"""

    def __init__(self, grs_ingredients, survey_par, b3d_rsd_spec):
        
        #assert isinstance(b3d_rsd_spec, Bispectrum3DRSDSpec)

        super().__init__(grs_ingredients, survey_par, b3d_rsd_spec)

        self._triangle_spec = self._data_spec.triangle_spec
        self.logger.debug('Initiating Bispectrum3DRSD class with\
            nori = {}'.format(self._data_spec.nori))

        self._f_array = self._get_f_array()
        self._fog_all = self._get_fog_all()

        (self._Z1_k1, self._Z1_k2, self._Z1_k3) = self._get_Z1_for_k1_k2_k3()

        self._data_spec.print_isamples()

    def _get_Bggg(self, term_name='b10_prim'):
        """3d numpy array of shape (nb, nz, ntri)"""

        allowed_term_name = ['b10_prim', 'b10_grav', 'b20']

        if term_name in allowed_term_name:

            Bggg = getattr(self, '_get_Bggg_' + term_name)()
            return Bggg
        else:
            raise NotImplementedError

    def _get_Bggg_b10_prim(self):
        
        nb = self._data_spec.nb
        nz = self._data_spec.nz
        ntri = self._triangle_spec.ntri
        nori = self._triangle_spec.nori

        Bggg_b10 = np.zeros((nb, nz, ntri, nori))

        for ib in range(nb):

            list_isamples = self._data_spec.get_list_of_isamples_for_ib(ib)

            for isamples in list_isamples:
                (isample1, isample2, isample3) = isamples
                Bggg_b10[ib, :, :, :] += self._get('Bmmm_prim')[np.newaxis, :, :, np.newaxis] \
                    * self._Z1_k1[isample1, :, :, :] \
                    * self._Z1_k2[isample2, :, :, :] \
                    * self._Z1_k3[isample3, :, :, :] \
                    * self._fog_all[ib, :, :, :]

        return Bggg_b10

    def _get_Bggg_b10_grav(self):
        
        nb = self._data_spec.nb
        nz = self._data_spec.nz
        ntri = self._triangle_spec.ntri
        nori = self._triangle_spec.nori
        
        Bggg_b10 = np.zeros((nb, nz, ntri, nori))

        for ib in range(nb):

            list_isamples = self._data_spec.get_list_of_isamples_for_ib(ib)
            
            for isamples in list_isamples:
                (isample1, isample2, isample3) = isamples
                Bggg_b10[ib, :, :, :] += self._get('Bmmm_grav')[np.newaxis, :, :, np.newaxis] \
                    * self._Z1_k1[isample1, :, :, :] \
                    * self._Z1_k2[isample2, :, :, :] \
                    * self._Z1_k3[isample3, :, :, :] \
                    * self._fog_all[ib, :, :, :]

        return Bggg_b10

    def _get_Bggg_b20(self):

        bias_20 = self._grs_ingredients.get('galaxy_bias_20') 

        nb = self._data_spec.nb
        nz = self._data_spec.nz
        ntri = self._triangle_spec.ntri
        nori = self._triangle_spec.nori

        #TODO should write access function like this 
        pks = np.array([np.transpose(np.array(self._get_pk12_23_13(iz)))\
             for iz in range(nz)])
        pk12 = np.array(pks)[:,:,0]
        pk23 = np.array(pks)[:,:,1]
        pk13 = np.array(pks)[:,:,2]
        
        if pks.shape != (nz, ntri, 3):
            self.logger.error('Assertion error {} vs {}'.format(pks.shape, (nz, ntri, 3)))

        Bggg_b20 = np.zeros((nb, nz, ntri, nori))

        for ib in range(nb):
            
            list_isamples = self._data_spec.get_list_of_isamples_for_ib(ib)
            
            for isamples in list_isamples:
                (isample1, isample2, isample3) = isamples

                Bggg_b20[ib, :, :, :] += \
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

        """Returns a tuple of 3 elements, each a 4d numpy array of shape (nsample, nz, nk, nori).
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

        if Z1_k1.shape != self._data_spec.transfer_shape:
            self.logger.error('Assertion error {} vs {}'.format(
                Z1_k1.shape, self._data_spec.transfer_shape))

        return (Z1_k1, Z1_k2, Z1_k3)

    def _get_sigp_kmu_squared_for_isamples(self, isample1, isample2, isample3, k1, k2, k3, sigp):
        #shape (ntri, nori)
        k1mu1 = k1[:,np.newaxis] * self._triangle_spec.mu_array[:,:,0]
        k2mu2 = k2[:,np.newaxis] * self._triangle_spec.mu_array[:,:,1]
        k3mu3 = k3[:,np.newaxis] * self._triangle_spec.mu_array[:,:,2]         

        #shape (nz, ntri, nori)
        sigp_kmu_squared = (k1mu1[np.newaxis, :, :] \
                * sigp[isample1, :, np.newaxis, np.newaxis])**2 \
            + (k2mu2[np.newaxis, :, :] \
                * sigp[isample2, :, np.newaxis, np.newaxis]) ** 2 \
            + (k3mu3[np.newaxis, :, :] \
                * sigp[isample3, :, np.newaxis, np.newaxis]) ** 2

        return sigp_kmu_squared


    def _get_fog_all(self):

        k1, k2, k3 = self._get_k1_k2_k3_array()
        sigp = self._grs_ingredients.get('sigp') #

        ib = 0
        fog = np.zeros((self._data_spec.nb, self._data_spec.nz, \
            self._data_spec.ntri, self._data_spec.nori))
        
        # TODO could write more succintly by creating a list of 
        # (isample1, isample2, isample3) tuples to iterate over instead.
        if self._data_spec._do_unique_multitracer == True:

            for isample1 in range(self._data_spec.nsample):
                for isample2 in range(isample1, self._data_spec.nsample):
                    for isample3 in range(isample2, self._data_spec.nsample):
                        #shape (nz, ntri, nori)
                        sigp_kmu_squared = self._get_sigp_kmu_squared_for_isamples(
                            isample1, isample2, isample3, k1, k2, k3, sigp
                        )
                        fog[ib,:,:,:] = np.exp( -0.5 * (sigp_kmu_squared))

                        ib = ib + 1

        else:

            for isample1 in range(self._data_spec.nsample):
                for isample2 in range(self._data_spec.nsample):
                    for isample3 in range(self._data_spec.nsample):
                        #shape (nz, ntri, nori)
                        sigp_kmu_squared = self._get_sigp_kmu_squared_for_isamples(
                            isample1, isample2, isample3, k1, k2, k3, sigp
                        )
                        fog[ib,:,:,:] = np.exp( -0.5 * (sigp_kmu_squared))

                        ib = ib + 1

        return fog
        
    def _get_f_array(self):

        z_array = self._data_spec.z
        debug_f = self._data_spec._debug_f_of_z

        if debug_f is not None:
            f_array = debug_f * np.ones_like(z_array)
        else:
            f_array = self._grs_ingredients.get('growth_rate_f')

        self.logger.debug('f_array = {}'.format(f_array))

        return f_array

# TODO NEXT do this next
# write this and refactor
class Bispectrum3DRSD_MurMuphi(Bispectrum3DRSD):

    """No AP"""

    def __init__(self, grs_ingredients, survey_par, b3d_rsd_spec):
        
        assert isinstance(b3d_rsd_spec, Bispectrum3DRSDSpec_MurMuphi)

        super().__init__(grs_ingredients, survey_par, b3d_rsd_spec)
