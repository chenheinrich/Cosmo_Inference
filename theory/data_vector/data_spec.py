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
        self._setup_triangle_spec()

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

    def _setup_triangle_spec(self):
        self._triangle_spec = TriangleSpec(self._k)

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
    def triangle_spec(self):
        return self._triangle_spec
    
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

class DataSpecBispectrum(DataSpec):

    def __init__(self, survey_par, data_spec_dict):
        super().__init__(survey_par, data_spec_dict)
        self._dict_isamples_to_ib, self._dict_ib_to_isamples, self._nb = \
            self._get_multi_tracer_config_all(self.nsample) 
    
    @property
    def nb(self):
        return self._nb
        
    @property
    def dict_isamples_to_ib(self):
        """Returns an integer for ib (the bispectrum index) given a string 
        of the form '%i_%i_%i'%(isample1, isample2, isample3)
        """
        return self._dict_isamples_to_ib

    @property
    def dict_ib_to_isamples(self):
        """Returns a tuple (isample1, isample2, isample3) given ib, 
        the bispectrum index
        """
        return self._dict_ib_to_isamples

    @staticmethod
    def _get_multi_tracer_config_all(nsample):
        dict_isamples_to_ib = {}
        dict_ib_to_isamples = {}
        ib = 0
        for isample1 in range(nsample):
            for isample2 in range(nsample):
                for isample3 in range(nsample):
                    dict_isamples_to_ib['%i_%i_%i'%(isample1, isample2, isample3)] = ib
                    dict_ib_to_isamples['%i'%ib] = (isample1, isample2, isample3)
                    ib = ib + 1
        nb = nsample**3 #TODO need to treat self._nps in power spectrum version
        assert ib == nb
        
        return dict_isamples_to_ib, dict_ib_to_isamples, nb

class DataSpecBispectrumOriented(DataSpecBispectrum):

    def __init__(self, survey_par, data_spec_dict):
        super().__init__(survey_par, data_spec_dict)
        
        self._theta1, self._phi12 = self._setup_angles(data_spec_dict)
    
    def _setup_angles(self, data_spec_dict):
        
        if data_spec_dict['triangle_orientation'] == 'theta1_phi12':
            
            min_theta1 = 0 if 'min_theta1' not in data_spec_dict.keys() \
                else data_spec_dict['min_theta1']
            max_theta1 = np.pi if 'max_theta1' not in data_spec_dict.keys() \
                else data_spec_dict['max_theta1']
            min_phi12 = 0 if 'min_phi12' not in data_spec_dict.keys() \
                else data_spec_dict['min_phi12']
            max_phi12 = 2. * np.pi if 'max_phi12' not in data_spec_dict.keys() \
                else data_spec_dict['max_phi12']
            
            theta1 = self._get_bin_centers_from_nbin(min_theta1, max_theta1, data_spec_dict['nbin_theta1'])
            phi12 = self._get_bin_centers_from_nbin(min_phi12, max_phi12, data_spec_dict['nbin_phi12'])

            print('theta1, phi12', theta1, phi12)

            return theta1, phi12
            
    @property
    def theta1(self):
        return self._theta1
    
    @property
    def phi12(self):
        return self._phi12

    @staticmethod
    def _get_bin_centers_from_nbin(min_value, max_value, nbin):
        edges = np.linspace(min_value, max_value, nbin+1)
        center = (edges[1:] + edges[:-1])/2.0
        return center
    

class TriangleSpec():

    """Class managing a list of k1, k2, k3 satisfying triangle inequalities given discretized k list."""

    def __init__(self, k):
        self._k = k
        self._nk = k.size
        self._tri_dict_tuple2index, self._tri_index_array, self._tri_array, self._ntri, \
            self._indices_equilateral, self._indices_k2_equal_k3 \
            = self._get_tri_info()
    
    @property
    def k(self):
        return self._k

    @property
    def ntri(self):
        return self._ntri

    @property
    def tri_dict_tuple2index(self):
        return self._tri_dict_tuple2index

    @property
    def tri_index_array(self):
        """Returns a 2d numpy array of shape (ntri, 3) for indices [ik1, ik2, ik3]
        that satisfies the triangle inequality.
        e.g. [[0, 0, 0],
              [0, 0, 1],
              ...
              [21, 21, 21]]
        """
        return self._tri_index_array

    @property
    def tri_array(self):
        """Returns a 2d numpy array of shape (ntri, 3) for k values [k1, k2, k3]
        in that satisfies the triangle inequality (same units as input k to class).
        e.g. [[0.0007, 0.0007, 0.0007],
              [0.0007, 0.0007, 0.01],
              ...
              [0.1, 0.1, 0.1]]
        """
        return self._tri_array

    @property
    def indices_equilateral(self):
        """Indices of equilateral triangles where k1 = k2 = k3."""
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
        """Indices of isoceles triangles where k2 = k3."""
        return self._indices_k2_equal_k3

    def get_ik1_ik2_ik3(self):
        """Returns a tuple of 3 elements, each being a 1d numpy array for ik1, ik2, ik3 respectively."""
        tri_index_array = self.tri_index_array
        ik1 = tri_index_array[:,0].astype(int)
        ik2 = tri_index_array[:,1].astype(int)
        ik3 = tri_index_array[:,2].astype(int)
        return (ik1, ik2, ik3)

    def _get_tri_info(self):
        """Find eligible triangles and create the dictionaries and arrays used in this class."""
        itri = 0
        nk = self._nk
        k = self._k

        indices_equilateral = []
        indices_k2_equal_k3 = []
        
        tri_dict_tuple2index = {}
        tri_index_array = np.zeros((nk**3, 3), dtype=int)
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


class AnglesNotInRangeError(Exception):

    def __init__(self, message='.'):
        self.message = 'Input angles are not in allowed range' + message
        super().__init__(self.message)

class TriangleSpecTheta1Phi12(TriangleSpec):

    """Class managing a list of triangles parametrized by (k1, k2, k3, theta1, phi12) given a discretized k list.
    We follow Scoccimarro 2015 page 4 for the definition of theta1, theta12 and phi12.
    
    mu1 = cos(theta1) = los dot k1;
    mu2 = cos(theta2) = los dot k2;
    
    theta12 and phi12 are respectively the polar and azimuthal angle in the frame formed by
    z' // k1, 
    x' in the same plane as los and k1, and k1 cross x' in the same direction as los x k1. 
    y' \perp z, y \perp x, such that x, y, z form a right-handed coordinates.
    """

    def __init__(self, k, theta1, phi12):
        super().__init__(k)
        
        try:
            self._check_input_angle_range(theta1, phi12)
        except AnglesNotInRangeError as e:
            print(e.message)

        self._theta1 = theta1
        self._phi12 = phi12

        self._ntheta1 = self._theta1.size
        self._nphi12 = self._phi12.size

        self._nori = self._ntheta1 * self._nphi12

        self._orientation_array = self._get_orientation_array()

        self._oriented_triangle_info = self._get_oriented_triangle_info()
        
    @property
    def nori(self):
        """An integer for the number of orientations per triangle shape."""
        return self._nori
        
    @property
    def theta1(self):
        """1d numpy array for the input theta1."""
        return self._theta1 
    
    @property
    def phi12(self):
        """1d numpy array for the input phi12."""
        return self._phi12
    
    @property
    def orientation_array(self):
        """2d numpy array where orientation_array[iori, :] gives [itheta1, iphi12] for iori-th orientation."""
        return self._orientation_array
        
    @property
    def oriented_tri_array(self):
        """3d numpy array where oriented_tri_array[itri, iori, :] gives [k1, k2, k3, theta1, phi12] 
        for the itri-th triangle and iori-th orientation."""
        return self._oriented_triangle_info['oriented_triangle']

    @property
    def oriented_tri_index_array(self):
        """3d numpy array where oriented_tri_array[itri, iori, :] gives [ik1, ik2, ik3, itheta1, iphi12] 
        for the itri-th triangle and iori-th orientation."""
        return self._oriented_triangle_info['index']

    @property
    def mu_array(self):
        """3d numpy array where mu_array[itri, iori, :] gives [mu1, mu2, mu3] 
        for the itri-th triangle and iori-th orientation."""
        return self._oriented_triangle_info['mu']

    @staticmethod
    def _check_input_angle_range(theta1, phi12):
        min_theta1 = 0.0
        max_theta1 = np.pi
        min_phi12 = 0.0
        max_phi12 = 2*np.pi
        
        if np.any(theta1 > max_theta1):
            raise AnglesNotInRangeError(message='found theta1 larger than {}'.format(max_theta1))
        elif not np.any(theta1 < min_theta1):
            raise AnglesNotInRangeError(message='found theta1 smaller than {}'.format(min_theta1))
        elif np.any(phi12 > max_phi12):
            raise AnglesNotInRangeError(message='found phi12 larger than {}'.format(max_phi12))
        elif np.any(phi12 < min_phi12):
            raise AnglesNotInRangeError(message='found phi12 smaller than {}'.format(min_phi12))

    def _get_orientation_array(self):

        orientation_array = np.zeros((self._nori, 2), dtype=int)

        iori = 0
        for itheta1, theta1 in enumerate(self._theta1):
            for iphi12, phi12 in enumerate(self._phi12):
                orientation_array[iori, :] = np.array([theta1, phi12])
                iori = iori + 1

        return orientation_array

    def _get_oriented_triangle_info(self):
        
        oriented_tri_mu_array = np.zeros((self._ntri, self._nori, 3))
        oriented_tri_index_array = np.zeros((self._ntri, self._nori, 5), dtype=int)
        oriented_tri_array = np.zeros((self._ntri, self._nori, 5))

        for itri in np.arange(self.ntri):

            ik1 = self.tri_index_array[itri][0]
            ik2 = self.tri_index_array[itri][1]
            ik3 = self.tri_index_array[itri][2]
            
            iori = 0
            for itheta1, theta1 in enumerate(self._theta1):
                for iphi12, phi12 in enumerate(self._phi12):
                    
                    k1 = self._tri_array[itri, 0]
                    k2 = self._tri_array[itri, 1]
                    k3 = self._tri_array[itri, 2]
                    mu1 = self._get_mu1(k1, k2, k3, theta1, phi12)
                    mu2 = self._get_mu1(k1, k2, k3, theta1, phi12)
                    mu3 = self._get_mu1(k1, k2, k3, theta1, phi12)

                    
                    oriented_tri_mu_array[itri, iori, :] = np.array([mu1, mu2, mu3])
                    oriented_tri_index_array[itri, iori, :] = np.array([ik1, ik2, ik3, itheta1, iphi12])
                    oriented_tri_array[itri, iori, :] = np.array([k1, k2, k3, theta1, phi12])

                    iori = iori + 1

        oriented_triangle_info = {}
        oriented_triangle_info['mu'] = oriented_tri_mu_array
        oriented_triangle_info['index'] = oriented_tri_index_array
        oriented_triangle_info['oriented_triangle'] = oriented_tri_array

        return oriented_triangle_info
                    
    @staticmethod
    def _get_mu1(k1, k2, k3, theta1, phi12):
        return np.cos(theta1)

    @staticmethod
    def _get_mu2(k1, k2, k3, theta1, phi12):
        theta12 = self._get_theta12(k1, k2, k3)
        mu2 = np.cos(theta1) * np.cos(theta12) - np.sin(theta1) * np.sin(theta12) * np.cos(phi12)
        return mu2

    @staticmethod
    def _get_mu3(k1, k2, k3, theta1, phi12):
        mu3 = k2 * np.sin(theta12) * np.cos(phi12) * np.sin(theta1) - (k1 + k2*np.cos(theta12)) * np.cos(theta1)
        mu3 = mu3/k3
        return mu3

    @staticmethod
    def _get_theta12(k1, k2, k3):
        """arccos always returns angle between [0, pi], 
        so theta12 is the same for two triangles of opposite handedness."""
        theta12 = np.arccos(0.5 * (-k1*k1 - k2*k2 + k3*k3) / (k1 * k2))
        return theta12
        