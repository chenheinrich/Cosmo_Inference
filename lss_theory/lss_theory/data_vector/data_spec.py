import numpy as np

from lss_theory.utils.misc import evaluate_string_to_float
from lss_theory.utils.logging import class_logger
from lss_theory.data_vector.triangle_spec import TriangleSpec
from lss_theory.data_vector.triangle_spec import TriangleOrientationSpec_Theta1Phi12
from lss_theory.data_vector.triangle_spec import TriangleOrientationSpec_MurMuphi

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

    @property
    def data_spec_dict(self):
        return self._dict

    def _setup_specs(self):

        self._setup_z()
        self._setup_sigz()
        self._setup_k()
        self._setup_mu()
        self._setup_k_perp_and_para()

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
        
    def _setup_k_perp_and_para(self):
        """Assigns 2d numpy array of shape (nk, nmu) to self._k_perp and self._k_para
        for k perpendicular and k parallel given k and mu array."""
        self._k_perp = self.k[:, np.newaxis] * \
            np.sqrt(1. - (self.mu**2)[np.newaxis, :])
        self._k_para = self.k[:, np.newaxis] * self.mu[np.newaxis, :]

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
    def k_perp(self):
        """2d numpy array of shape (nk, nmu)."""
        return self._k_perp

    @property
    def k_para(self):
        """2d numpy array of shape (nk, nmu)."""
        return self._k_para

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

        k_perp_ref = self.k_perp
        k_para_ref = self.k_para

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
        """Assuming galaxy power spectrum is symmetric in the two galaxy samples, 
        i.e. P^{ab} = P^{ba}, so that ab = 12 and ab = 21 give the same power
        spectrum index ips, but a given ips only gives increasing galaxy sample 
        indices, ab = 12.
        """
        dict_isamples_to_ips = {}
        dict_ips_to_isamples = {}
        ips = 0
        for isample1 in range(nsample):
            for isample2 in range(isample1, nsample):
                dict_isamples_to_ips['%i_%i'%(isample1, isample2)] = ips
                dict_isamples_to_ips['%i_%i'%(isample2, isample1)] = ips
                dict_ips_to_isamples['%i'%ips] = (isample1, isample2)
                ips = ips + 1
        nps = int(nsample * (nsample + 1)/2)
        assert ips == nps
        
        return dict_isamples_to_ips, dict_ips_to_isamples, nps

    def get_ips(self, isample1, isample2):
        """Returns index of power spectrum given indices of galaxy samples"""
        return self._dict_isamples_to_ips['%i_%i'%(isample1, isample2)]
        # TODO NEXT need to store in dict the opposite 1_0 too.
    
    def get_isamples(self, ips):
        """Returns a tuple of 2 galaxy sample indices given the index of power spectrum"""
        return self._dict_ips_to_isamples['%i'%(ips)]

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

    #TODO abort and replace by get_ib()
    @property
    def dict_isamples_to_ib(self):
        """Returns an integer for ib (the bispectrum index) given a string 
        of the form '%i_%i_%i'%(isample1, isample2, isample3)
        """
        return self._dict_isamples_to_ib

    #TODO abort and replace by get isamples
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

    def get_ib(self, isample1, isample2, isample3):
        """Returns the index of bispectrum given indices of 3 galaxy samples"""
        return self._dict_isamples_to_ib['%i_%i_%i'%(isample1, isample2, isample3)]

    def get_isamples(self, ib):
        """Returns a tuple of 3 galaxy sample indices given index of bispectrum"""
        return self._dict_ib_to_isamples['%i'%(ib)]

class Bispectrum3DRSDSpec(Bispectrum3DBaseSpec):
    """Bispectrum3DRSDSpec should not be used directly, use its subclasses"""

    def __init__(self, survey_par, data_spec_dict):
        super().__init__(survey_par, data_spec_dict)

        self._debug_sigp = data_spec_dict['debug_settings']['sigp']
        self._debug_f_of_z = data_spec_dict['debug_settings']['f_of_z']
        self._set_mu_to_zero = data_spec_dict['debug_settings']['set_mu_to_zero']

        triangle_orientation_dict = data_spec_dict['triangle_orientation_info']
        self._triangle_spec = self._get_triangle_spec(triangle_orientation_dict)

        self.do_folded_signal = triangle_orientation_dict['do_folded_signal']

        self._overwrite_shape_for_b3d_rsd()

    @property
    def nori(self):
        return self._triangle_spec.nori
    
    @property
    def angle_array(self):
        """angle_array[iori, :] returns the two orientation parameters
        a and b (in radians if it's an angle) given index iori. """
        return self._triangle_spec.angle_array

    @staticmethod
    def _get_bin_centers_from_nbin(min_value, max_value, nbin):
        edges = np.linspace(min_value, max_value, nbin+1)
        center = (edges[1:] + edges[:-1])/2.0
        return center

    def _overwrite_shape_for_b3d_rsd(self):
        self._shape = (self.nb, self.nz, self.ntri, self.nori)
        self._transfer_shape = (self.nsample, self.nz, self.ntri, self.nori)

    def _setup_angles(self, tri_ori_dict):
        raise NotImplementedError

    def _get_min_max_parameters(self, tri_ori_dict):
        raise NotImplementedError

    def _get_triangle_spec(self, triangle_orientation_dict):
        raise NotImplementedError

class Bispectrum3DRSDSpec_Theta1Phi12(Bispectrum3DRSDSpec):

    def __init__(self, survey_par, data_spec_dict):
        super().__init__(survey_par, data_spec_dict)
    
    @property
    def ntheta1(self):
        return self._triangle_spec._ntheta1
    
    @property
    def nphi12(self):
        return self._triangle_spec._nphi12

    @property
    def theta1(self):
        return self._theta1
    
    @property
    def phi12(self):
        return self._phi12

    @property
    def cos_theta1(self):
        return np.cos(self._theta1)

    @property 
    def Sigma_to_use(self):
        """Returns the right Sigma to use depending on whether the input
        triangle_orientation_info had do_folded_signal = True or False."""
        if self.do_folded_signal is True:
            return self.Sigma_scaled_to_4pi
        else:
            return self.Sigma

    @property
    def dOmega(self):
        """The solid angle element dOmega = dmu1 * dphi12. """
        dmu1 = self.triangle_spec.da
        dphi12 = self.triangle_spec.db
        dOmega = dmu1 * dphi12
        return dOmega

    @property
    def Sigma(self):
        """The fraction of discretized solid angle element to total 4pi. 
        Use if do_folded_signal = False."""
        Sigma = self.dOmega / (4.0 * np.pi)
        return Sigma

    @property
    def Sigma_scaled_to_4pi(self):
        """If do_folded_signal = True, one is choosing to calculate the signal 
        in only part of the whole 4pi solid angle, because of symmetries. 
        Then we assume that the covariance is reduced by a factor corresponding
        to that allowed by the symmetry, and that the data vector is already
        averaged over these modes with the same expected signal. In this case,
        Nmodes propto Sigma is multiplied by (4pi)/total_omega, we put this 
        factor here in Sigma_scaled_to_4pi, dividing by total_omega_over_4pi."""
        total_omega_over_4pi = (self._max_cos_theta1 - self._min_cos_theta1) \
            * (self._max_phi12 - self._min_phi12)/(4.*np.pi)
        return (self.Sigma/total_omega_over_4pi)

    def _get_orientation_parameters(self, tri_ori_dict):

        """Returns theta1 and phi12 given min max and number of bins of cos(theta1) and phi12. """
        
        if tri_ori_dict['parametrization_name'] == 'theta1_phi12':
            
            (min_cos_theta1, max_cos_theta1, min_phi12, max_phi12) = self._get_min_max_parameters(tri_ori_dict)
            
            self._logger.info('Using cos_theta1 in [{}, {}]'.format(min_cos_theta1, max_cos_theta1))
            self._logger.info('Using phi12 in [{}, {}]'.format(min_phi12, max_phi12))
            
            cos_theta1 = self._get_bin_centers_from_nbin(min_cos_theta1, max_cos_theta1, tri_ori_dict['nbin_cos_theta1'])
            theta1 = np.arccos(cos_theta1) 
            phi12 = self._get_bin_centers_from_nbin(min_phi12, max_phi12, tri_ori_dict['nbin_phi12'])

            self._logger.info('theta1_in_deg = {}'.format(theta1 / np.pi * 180.0))
            self._logger.info('phi12 in deg = {}'.format(phi12 / np.pi * 180.0))
            
            return theta1, phi12
    
    def _get_min_max_parameters(self, tri_ori_dict):

        if tri_ori_dict['parametrization_name'] == 'theta1_phi12':

            min_cos_theta1 = -1.0 if 'min_cos_theta1' not in tri_ori_dict.keys() \
                else evaluate_string_to_float(tri_ori_dict['min_cos_theta1'])

            max_cos_theta1 = 1.0 if 'max_cos_theta1' not in tri_ori_dict.keys() \
                else evaluate_string_to_float(tri_ori_dict['max_cos_theta1'])

            min_phi12 = 0 if 'min_phi12' not in tri_ori_dict.keys() \
                else evaluate_string_to_float(tri_ori_dict['min_phi12'])

            max_phi12 = 2.*np.pi if 'max_phi12' not in tri_ori_dict.keys() \
                else evaluate_string_to_float(tri_ori_dict['max_phi12'])

            return (min_cos_theta1, max_cos_theta1, min_phi12, max_phi12)

        else:
            raise NotImplementedError

    def _get_triangle_spec(self, triangle_orientation_dict):

        (self._min_cos_theta1, self._max_cos_theta1, self._min_phi12, self._max_phi12) \
            = self._get_min_max_parameters(triangle_orientation_dict)

        # should reflect back cos_theta1 up there
        self._theta1, self._phi12 = \
            self._get_orientation_parameters(triangle_orientation_dict)

        return TriangleOrientationSpec_Theta1Phi12(self._k, self._theta1, self._phi12, \
            self._set_mu_to_zero) 

# Note: The advantage of making murmuphi a whole new observable class
# is that you get to define things like Sigma and other fraction of 
# triangles within a bin so that when we are doing integration
# over orientation parameters we can directly access them here.
class Bispectrum3DRSDSpec_MurMuphi(Bispectrum3DRSDSpec):

    def __init__(self, survey_par, data_spec_dict):
        super().__init__(survey_par, data_spec_dict)

    @property
    def nmur(self):
        return self._triangle_spec._na
    
    @property
    def nmuphi(self):
        return self._triangle_spec._nb

    @property
    def mu_r(self):
        return self._triangle_spec._a
    
    @property
    def mu_phi(self):
        return self._triangle_spec._b

    @property 
    def fraction_of_triangles_per_ori_bin(self): 
        #TODO call this from cov code
        #TODO adjust the one in Bispectrum3DRSD_Theta1Phi12_Spec to follow same format
        """Returns Sigma_mu1_mu2 * dOmega"""
        return self.Sigma_mu1_mu2 * self.dOmega

    @property
    def dOmega(self):
        """
        Returns 2d numpy array such that dOmega[itri, iori] gives the 
        solid angle element dOmega = mur * dmur * dmuphi = dmu1 dmu2.
        """
        return self.triangle_spec.dOmega

    @property
    def Sigma_mu1_mu2(self):
        """
        A 2d numpy array for Sigma(mu1, mu2) such that Sigma[itri, iori] returns 
        Sigma(mu1, mu2) which is defined as:
            Sigma(mu1, mu2) dmu1 dmu2 = Sigma(mu1, mu2) mur dmur dmuphi 
            is the fraction of triangles with fixed shape in a dmu1-dmu2 bin, 
        and so integrating Sigma(mu1, mu2) dmu1 dmu2 over the whole sphere = 4pi.
        """
        return self.triangle_spec.Sigma_mu1_mu2

    def _get_orientation_parameters(self, tri_ori_dict):

        """Returns theta1 and phi12 given min max and number of bins of cos(theta1) and phi12. """
        
        if tri_ori_dict['parametrization_name'] == 'mur_muphi':
            
            (min_a, max_a, min_b, max_b) = self._get_min_max_parameters(tri_ori_dict)
            
            self._logger.info('Using mu_r in [{}, {}]'.format(min_a, max_a))
            self._logger.info('Using mu_phi in [{}, {}]'.format(min_b, max_b))
            
            mu_r = self._get_bin_centers_from_nbin(min_a, max_a, tri_ori_dict['nbin_mur'])
            mu_phi = self._get_bin_centers_from_nbin(min_b, max_b, tri_ori_dict['nbin_muphi'])

            self._logger.info('mu_r = {}'.format(mu_r))
            self._logger.info('mu_phi in deg = {}'.format(mu_phi / np.pi * 180.0))
            
            return mu_r, mu_phi
    
    def _get_min_max_parameters(self, tri_ori_dict):

        if tri_ori_dict['parametrization_name'] == 'mur_muphi':

            min_mur = 0 if 'min_mur' not in tri_ori_dict.keys() \
                else evaluate_string_to_float(tri_ori_dict['min_mur'])

            max_mur = np.sqrt(2) if 'max_mur' not in tri_ori_dict.keys() \
                else evaluate_string_to_float(tri_ori_dict['max_mur'])

            min_muphi = 0 if 'min_muphi' not in tri_ori_dict.keys() \
                else evaluate_string_to_float(tri_ori_dict['min_muphi'])

            max_muphi = 2.*np.pi if 'max_muphi' not in tri_ori_dict.keys() \
                else evaluate_string_to_float(tri_ori_dict['max_muphi'])

            return (min_mur, max_mur, min_muphi, max_muphi)

        else:
            raise NotImplementedError

    def _get_triangle_spec(self, triangle_orientation_dict):

        (self._min_mu_r, self._max_mu_phi, self._min_mu_r, self._max_mu_phi) \
            = self._get_min_max_parameters(triangle_orientation_dict)
            
        self._mu_r, self._mu_phi = \
            self._get_orientation_parameters(triangle_orientation_dict)

        triangle_spec = TriangleOrientationSpec_MurMuphi(self._k, self._mu_r, self._mu_phi, \
            self._set_mu_to_zero) 

        return triangle_spec
            