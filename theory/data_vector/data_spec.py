import numpy as np
class DataSpec():

    """
    Sample Usage:
        d = DataSpec(survey_par, data_spec_dict)

    You can direclty access variables: z, sigz, k, dk, mu, dmu, nps, 
        nsample, nz, nk, nmu, shape = (nps, nz, nk, nmu).
    These cannot be set from outside the class, 
        but can be accessed as e.g. k = d.k, because we 
        used the @property decorator around a getter.
    """

    def __init__(self, survey_par, data_spec_dict):
        self._survey_par = survey_par
        self._dict = data_spec_dict
        self._setup_specs()

    def _setup_specs(self):

        self._setup_z()
        self._setup_sigz()
        self._setup_k()
        self._setup_mu()

        self._setup_n()
        self._setup_shape()
        self._setup_triangle_specs()

    def _setup_z(self):
        self._z = self._survey_par.get_zmid_array()
    
    def _setup_sigz(self):
        self._sigz = self._survey_par.get_sigz_array()

    def _setup_k(self):
        self._k = np.logspace(np.log10(self._dict['kmin']), np.log10(self._dict['kmax']), self._dict['nk'])
        self._dk = self._get_dk(self._k)

    def _get_dk(self, k):
        """Return dk array.

        Note: This assumes that k is uniformly spaced in log space. 
        It computes dk by taking the difference in linear space between the
        log bin-centers, assuming that the first and last bin are only half 
        the usual log bin size."""

        #TODO change this prescription AFTER test against old code!!
        logk = np.log(k)
        logk_mid = (logk[:-1] + logk[1:]) / 2
        dk = np.zeros(k.size)
        dk[1:-1] = np.exp(logk_mid)[1:] - np.exp(logk_mid)[0:-1]
        dk[0] = np.exp(logk_mid[0]) - np.exp(logk[0])
        dk[-1] = np.exp(logk[-1]) - np.exp(logk_mid[-1])
        return dk

    def _setup_mu(self):
        self._mu_edges = np.linspace(0, 1, self._dict['nmu'] + 1)
        self._mu = (self._mu_edges[:-1] + self._mu_edges[1:]) / 2.0
        self._dmu = self._mu_edges[1:] - self._mu_edges[:-1]

        assert self._mu.size == self._dmu.size, ('mu and dmu do not have the same size: {}, {}'.format(
            self._mu.size, self._dmu.size))
    
    def _setup_n(self):
        self._nsample = self._sigz.shape[0]
        self._nps = int(self._nsample * (self._nsample + 1) / 2)
        self._nz = self._z.size
        self._nk = self._k.size
        self._nmu = self._mu.size

        assert self._nmu == self._dict['nmu']
        assert self._nk == self._dict['nk'] #- 1 #TODO to be changed so input nk means nkbin
    
    def _setup_shape(self):
        self._shape = (self._nps, self._nz, self._nk, self._nmu)
        self._transfer_shape = (self._nsample, self._nz, self._nk, self._nmu)

    def _setup_triangle_specs(self):
        self._triangle_specs = TriangleSpecs(self._k)

    @property
    def z(self):
        return self._z
    
    @property
    def sigz(self):
        return self._sigz
    
    @property
    def k(self):
        return self._k

    @property
    def mu(self):
        return self._mu

    @property
    def dk(self):
        return self._dk

    @property
    def dmu(self):
        return self._dmu

    @property
    def shape(self):
        return self._shape
    
    @property
    def transfer_shape(self):
        return self._transfer_shape

    @property
    def nps(self):
        return self._nps

    @property
    def nz(self):
        return self._nz

    @property
    def nk(self):
        return self._nk

    @property
    def nmu(self):
        return self._nmu

    @property
    def nsample(self):
        return self._nsample

    @property
    def triangle_specs(self):
        return self._triangle_specs
    
    def get_k_actual_perp_and_para(self, ap_perp, ap_para, z=None):
        """Return two 3-d numpy arrays of shape (nz, nk, mu) 
        for the actual values of k_perp and k_para to line-of-sight,
        given the two AP factors in directions perpendicular to 
        and parallel to the line-of-sigh, ap_perp and ap_para, 
        each specified as a 1-d numpy array of size self._d.size:
            k_perp = k_perp|ref * D_A(z)|ref / D_A(z),
            k_para = k_para|ref * (1/H(z))|ref / (1/H(z))
        """
        
        if z is not None:
            assert ap_perp.shape == z.shape, (ap_perp.shape, z.shape)
            assert ap_para.shape == z.shape, (ap_para.shape, z.shape)

        k_perp_ref = self.k[:, np.newaxis] * \
            np.sqrt(1. - (self.mu**2)[np.newaxis, :])
        k_para_ref = self.k[:, np.newaxis] * self.mu[np.newaxis, :]

        k_actual_perp = k_perp_ref[np.newaxis, :, :] * \
            ap_perp[:, np.newaxis, np.newaxis]

        k_actual_para = k_para_ref[np.newaxis, :, :] * \
            ap_para[:, np.newaxis, np.newaxis]

        return (k_actual_perp, k_actual_para)

    def get_k_and_mu_actual(self, ap_perp, ap_para, z=None):
        """Return two 3-d numpy arrays of shape (nz, nk, mu) 
        for the actual values of k and mu,
        given the two AP factors in directions perpendicular to 
        and parallel to the line-of-sigh, ap_perp and ap_para, 
        each specified as a 1-d numpy array of size self._d.size:
            k_perp = k_perp|ref * D_A(z)|ref / D_A(z),
            k_para = k_para|ref * (1/H(z))|ref / (1/H(z)),
        where
            k// = mu * k,
            kperp = sqrt(k^2  - k//^2) = k sqrt(1 - mu^2).
        """
        k_actual_perp, k_actual_para = self.get_k_actual_perp_and_para(ap_perp, ap_para, z=None)
        
        k_actual = np.sqrt(k_actual_perp**2 + k_actual_para**2)
        mu_actual = k_actual_para/k_actual

        return (k_actual, mu_actual)


        
class TriangleSpecs():

    def __init__(self, k):
        self._k = k
        self._nk = k.size
        self._tri_dict_tuple2index, self._tri_index_array, self._tri_array, self._ntri, \
            self._indices_equilateral, self._indices_k2_equal_k3 \
            = self._get_tri_info()
    
    @property
    def ntri(self):
        return self._ntri

    @property
    def tri_dict_tuple2index(self):
        return self._tri_dict_tuple2index

    @property
    def tri_index_array(self):
        return self._tri_index_array

    @property
    def tri_array(self):
        return self._tri_array

    @property
    def indices_equilateral(self):
        assert np.all(self._indices_equilateral == self.indices_equilateral2)
        return self._indices_equilateral

    @property
    def indices_equilateral2(self):
        (ik1, ik2, ik3) = self.get_ik1_ik2_ik3()
        ind12 = np.where(ik1 == ik2)[0]
        ind23 = np.where(ik2 == ik3)[0]
        indices = [ind for ind in ind12 if ind in ind23] 
        assert np.all(ik1[indices] == ik2[indices])
        assert np.all(ik2[indices] == ik3[indices])
        assert len(indices) == self._nk
        return indices

    @property
    def indices_k2_equal_k3(self):
        return self._indices_k2_equal_k3

    def get_ik1_ik2_ik3(self):
        tri_index_array = self.tri_index_array
        ik1 = tri_index_array[:,0].astype(int)
        ik2 = tri_index_array[:,1].astype(int)
        ik3 = tri_index_array[:,2].astype(int)
        return (ik1, ik2, ik3)

    def _get_tri_info(self):
        itri = 0
        nk = self._nk
        k = self._k

        indices_equilateral = []
        indices_k2_equal_k3 = []
        
        tri_dict_tuple2index = {}
        tri_index_array = np.zeros((nk**3, 3))
        tri_array = np.zeros((nk**3, 3))

        for ik1 in np.arange(nk):

            indices_equilateral.append(itri)

            for ik2 in np.arange(ik1, nk):   
                
                indices_k2_equal_k3.append(itri)
                
                k1 = k[ik1]
                k2 = k[ik2]
                k3_array = k
                ik3_range = self._get_ik3_range_satisfying_triangle_inequality(k1, k2, k3_array)

                for ik3 in ik3_range:

                    k3 = k3_array[ik3]
                    tri_dict_tuple2index['%i, %i, %i'%(ik1, ik2, ik3)] = itri
                    tri_index_array[itri] = [ik1, ik2, ik3]
                    tri_array[itri] = [k1, k2, k3]
                    itri = itri + 1

        ntri = itri
        assert np.all(tri_index_array[ntri:-1, :] == 0)
        tri_index_array = tri_index_array[:ntri, :]
        tri_index_array = tri_index_array[:ntri, :]

        print('indices_steppint_in_k2', indices_k2_equal_k3)
        print('tri_array[indices_k2_equal_k3,:]', tri_array[indices_k2_equal_k3,:])
        print(len(indices_k2_equal_k3))
        
        assert len(indices_k2_equal_k3) == self._nk * (self._nk + 1)/2, \
            (len(indices_k2_equal_k3), self._nk * (self._nk + 1)/2)
        
        return tri_dict_tuple2index, tri_index_array, tri_array, ntri, indices_equilateral, indices_k2_equal_k3

    @staticmethod
    def _get_ik3_range_satisfying_triangle_inequality(k1, k2, k3_array):
        ik3_min = np.min(np.where(k3_array >= k2))
        ik3_max = np.max(np.where(k3_array <= (k1 + k2)))
        ik3_array = np.arange(ik3_min, ik3_max+1)
        k3min = np.min(k3_array[ik3_array])
        k3max = np.max(k3_array[ik3_array])
        assert k3max <= (k1+k2)  
        assert k3min >= k2  
        return ik3_array




