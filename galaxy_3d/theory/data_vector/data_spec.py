import numpy as np

from theory.utils.misc import evaluate_string_to_float
from theory.utils.logging import class_logger
from theory.data_vector.triangle_spec import TriangleSpec, TriangleSpecTheta1Phi12

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
        self._logger = class_logger(self)
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
        nmu = self._dict['nmu'] 

        if nmu == -1: 
            #TODO setting mu = 0 still leaves k_actual = k_ref (H_ref/ H_actual)
            # this could be useful for not including FoG effects and Kaiser
            # perhaps could that a debug setting instead.
            #Changed all bispectrum to use no AP effects
            # Might need to change this depending on what we decide about AP.
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

        if self._dict['nmu'] == -1:
            assert self._nmu == 1, (self._nmu, 'expect 1')
        else:
            assert self._nmu == self._dict['nmu'], (self._nmu, 'expect input value', self._dict['nmu'])

        assert self._nk == self._dict['nk'] #TODO to be changed so input nk means nkbin
    
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

class PowerSpectrum3DSpec(DataSpec):

    def __init__(self, survey_par, data_spec_dict):
        super().__init__(survey_par, data_spec_dict)
        self._dict_isamples_to_ips, self._dict_ips_to_isamples, _nps = \
            self._get_multi_tracer_config_all(self.nsample) 
        assert _nps == self._nps
    
    @staticmethod
    def _get_multi_tracer_config_all(nsample):
        dict_isamples_to_ips = {}
        dict_ips_to_isamples = {}
        ips = 0
        for isample1 in range(nsample):
            for isample2 in range(isample1, nsample):
                    dict_isamples_to_ips['%i_%i'%(isample1, isample2)] = ips
                    dict_ips_to_isamples['%i'%ips] = (isample1, isample2)
                    ips = ips + 1
        nps = int(nsample * (nsample + 1)/2)
        assert ips == nps
        
        return dict_isamples_to_ips, dict_ips_to_isamples, nps

class Bispectrum3DBaseSpec(DataSpec):

    def __init__(self, survey_par, data_spec_dict):
        super().__init__(survey_par, data_spec_dict) 
        
        self._dict_isamples_to_ib, self._dict_ib_to_isamples, self._nb = \
            self._get_multi_tracer_config_all(self.nsample)

        self._setup_ntri()
        self._overwrite_shape_for_b3d_base()
        
    @property
    def nb(self):
        return self._nb
    
    @property
    def ntri(self):
        return self._ntri

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
        nb = nsample**3 
        assert ib == nb
        
        return dict_isamples_to_ib, dict_ib_to_isamples, nb
    
    def _setup_ntri(self):
        self._ntri = self.triangle_spec.ntri

    def _setup_nb(self):
        self._nb = self.nsample**3

    def _overwrite_shape_for_b3d_base(self):
        self._shape = (self.nb, self.nz, self.ntri)
        self._transfer_shape = (self.nsample, self.nz, self.ntri)

    def get_dk1_dk2_dk3(self):
        (ik1, ik2, ik3) = self.triangle_spec.get_ik1_ik2_ik3() 
        return (self._dk[ik1], self._dk[ik2], self._dk[ik3])

class Bispectrum3DRSDSpec(Bispectrum3DBaseSpec):

    def __init__(self, survey_par, data_spec_dict):
        super().__init__(survey_par, data_spec_dict)
        
        (self._min_cos_theta1, self._max_cos_theta1, self._min_phi12, self._max_phi12) \
            = self._get_min_max_angles(data_spec_dict)
        self._theta1, self._phi12, self._cos_theta1 = self._setup_angles(data_spec_dict)
        
        self._triangle_spec = TriangleSpecTheta1Phi12(self._k, self._theta1, self._phi12, \
            set_mu_to_zero = data_spec_dict['debug_settings']['set_mu_to_zero']) 
    
        self._debug_sigp = data_spec_dict['debug_settings']['sigp']
        self._debug_f_of_z = data_spec_dict['debug_settings']['f_of_z']

        self.do_folded_signal = data_spec_dict['do_folded_signal']

        self._setup_nori()
        self._overwrite_shape_for_b3d_rsd()

    @property
    def nori(self):
        return self._nori

    @property
    def theta1(self):
        return self._theta1
    
    @property
    def phi12(self):
        return self._phi12

    @property
    def cos_theta1(self):
        return self._cos_theta1

    @property
    def Sigma_scaled_to_4pi(self):
        total_omega_over_4pi = (self._max_cos_theta1 - self._min_cos_theta1) \
            * (self._max_phi12 - self._min_phi12)/(4.*np.pi)
        return (self.Sigma/total_omega_over_4pi)

    @property
    def Sigma(self):
        dmu1 = self.triangle_spec.dmu1
        dphi12 = self.triangle_spec.dphi12
        Sigma = dmu1 * dphi12 / (4.0*np.pi) 
        return Sigma

    def _setup_angles(self, data_spec_dict):

        """Returns theta1 and phi12 given min max and number of bins of cos(theta1) and phi12. """
        
        if data_spec_dict['triangle_orientation'] == 'theta1_phi12':
            
            (min_cos_theta1, max_cos_theta1, min_phi12, max_phi12) = self._get_min_max_angles(data_spec_dict)
            
            self._logger.info('Using cos_theta1 in [{}, {}]'.format(min_cos_theta1, max_cos_theta1))
            self._logger.info('Using phi12 in [{}, {}]'.format(min_phi12, max_phi12))
            
            cos_theta1 = self._get_bin_centers_from_nbin(min_cos_theta1, max_cos_theta1, data_spec_dict['nbin_cos_theta1'])
            theta1 = np.arccos(cos_theta1) 
            phi12 = self._get_bin_centers_from_nbin(min_phi12, max_phi12, data_spec_dict['nbin_phi12'])

            self._logger.info('theta1_in_deg = {}'.format(theta1 / np.pi * 180.0))
            self._logger.info('phi12 in deg = {}'.format(phi12 / np.pi * 180.0))
            
            return theta1, phi12, cos_theta1
    
    def _get_min_max_angles(self, data_spec_dict):

        if data_spec_dict['triangle_orientation'] == 'theta1_phi12':

            min_cos_theta1 = -1.0 if 'min_cos_theta1' not in data_spec_dict.keys() \
                else evaluate_string_to_float(data_spec_dict['min_cos_theta1'])

            max_cos_theta1 = 1.0 if 'max_cos_theta1' not in data_spec_dict.keys() \
                else evaluate_string_to_float(data_spec_dict['max_cos_theta1'])

            min_phi12 = 0 if 'min_phi12' not in data_spec_dict.keys() \
                else evaluate_string_to_float(data_spec_dict['min_phi12'])

            max_phi12 = 2.*np.pi if 'max_phi12' not in data_spec_dict.keys() \
                else evaluate_string_to_float(data_spec_dict['max_phi12'])

            return (min_cos_theta1, max_cos_theta1, min_phi12, max_phi12)

        else:
            raise NotImplementedError
            
    @staticmethod
    def _get_bin_centers_from_nbin(min_value, max_value, nbin):
        edges = np.linspace(min_value, max_value, nbin+1)
        center = (edges[1:] + edges[:-1])/2.0
        return center

    def _setup_nori(self):
        self._nori = self._triangle_spec.nori

    def _overwrite_shape_for_b3d_rsd(self):
        self._shape = (self.nb, self.nz, self.ntri, self.nori)
        self._transfer_shape = (self.nsample, self.nz, self.ntri, self.nori)
    
