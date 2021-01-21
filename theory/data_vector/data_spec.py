import numpy as np

from theory.utils.misc import evaluate_string_to_float
from theory.data_vector.triangle_spec import TriangleSpec, TriangleSpecTheta1Phi12

class AnglesNotInRangeError(Exception):

    def __init__(self, angle, message='.'):
        self.angle = angle
        self.message = 'Error: Input angles %s are not in allowed range: '%self.angle + message
        super().__init__(self.message)

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
        nmu = self._dict['nmu'] #TODO HACK have to see how this works

        if nmu == -1:
            self._mu = np.array([0.0])
        else:
            self._mu_edges = np.linspace(0, 1, nmu + 1)
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

        #TODO think about how to treat this nmu = -1 case: #assert self._nmu == self._dict['nmu'] 
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
        self._triangle_spec = TriangleSpecTheta1Phi12(self._k, self._theta1, self._phi12)
    
    @property
    def theta1(self):
        return self._theta1
    
    @property
    def phi12(self):
        return self._phi12

    def _setup_angles(self, data_spec_dict):

        """Returns theta1 and phi12 given min max and number of bins of cos(theta1) and phi12. """
        
        if data_spec_dict['triangle_orientation'] == 'theta1_phi12':
            
            min_cos_theta1 = -1.0 if 'min_cos_theta1' not in data_spec_dict.keys() \
                else evaluate_string_to_float(data_spec_dict['min_cos_theta1'])
            max_cos_theta1 = 1.0 if 'max_cos_theta1' not in data_spec_dict.keys() \
                else evaluate_string_to_float(data_spec_dict['max_cos_theta1'])
            min_phi12 = 0 if 'min_phi12' not in data_spec_dict.keys() \
                else evaluate_string_to_float(data_spec_dict['min_phi12'])
            max_phi12 = 2. * np.pi if 'max_phi12' not in data_spec_dict.keys() \
                else evaluate_string_to_float(data_spec_dict['max_phi12'])
            
            print('min_cos_theta1, max_cos_theta1', min_cos_theta1, max_cos_theta1)
            print('min_phi12, max_phi12', min_phi12, max_phi12)
            
            cos_theta1 = self._get_bin_centers_from_nbin(min_cos_theta1, max_cos_theta1, data_spec_dict['nbin_cos_theta1'])
            theta1 = np.arccos(cos_theta1) 
            phi12 = self._get_bin_centers_from_nbin(min_phi12, max_phi12, data_spec_dict['nbin_phi12'])

            print('cos_theta1, theta1, phi12', cos_theta1, theta1, phi12)
            print('theta1_in_deg', theta1 / np.pi * 180.0)
            print('phi12 in deg', phi12 / np.pi * 180.0)
            
            return theta1, phi12
            
    @property
    def cos_theta1(self):
        return self._cos_theta1
    
    @property
    def phi12(self):
        return self._phi12

    @staticmethod
    def _get_bin_centers_from_nbin(min_value, max_value, nbin):
        edges = np.linspace(min_value, max_value, nbin+1)
        center = (edges[1:] + edges[:-1])/2.0
        return center
    
