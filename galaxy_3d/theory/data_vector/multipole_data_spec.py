import numpy as np

from theory.utils.misc import evaluate_string_to_float
from theory.utils.logging import class_logger
from theory.data_vector.triangle_spec import TriangleSpec, TriangleSpecTheta1Phi12

from theory.data_vector.data_vector import Bispectrum3DBaseSpec

class BispectrumMultipoleSpec(Bispectrum3DBaseSpec):

    """
    Sample Usage: 
        d = BispectrumMultipoleSpec(survey_par, multipole_data_spec_dict)

    You can direclty access variables: z, sigz, k, dk, mu, dmu, nps, 
        nsample, nz, nk, nmu, shape = (nps, nz, nk, nmu).
    """
    #TODO update documentation above

    #TODO add a different dictionary and treatment bloew
    def __init__(self, survey_par, data_spec_dict):
        super().__init__(survey_par, data_spec_dict)
        
        triangle_orientation_dict = data_spec_dict['triangle_orientation_info']

        (self._min_cos_theta1, self._max_cos_theta1, self._min_phi12, self._max_phi12) \
            = self._get_min_max_angles(triangle_orientation_dict)
        self._theta1, self._phi12, self._cos_theta1 = \
            self._setup_angles(triangle_orientation_dict)
        
        self._triangle_spec = TriangleSpecTheta1Phi12(self._k, self._theta1, self._phi12, \
            set_mu_to_zero = data_spec_dict['debug_settings']['set_mu_to_zero']) 
    
        self._debug_sigp = data_spec_dict['debug_settings']['sigp']
        self._debug_f_of_z = data_spec_dict['debug_settings']['f_of_z']

        self.do_folded_signal = triangle_orientation_dict['do_folded_signal']

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
    def Sigma(self):
        dmu1 = self.triangle_spec.dmu1
        dphi12 = self.triangle_spec.dphi12
        Sigma = dmu1 * dphi12 / (4.0*np.pi) 
        return Sigma

    @property
    def Sigma_scaled_to_4pi(self):
        """If do_folded_signal = True, one is choosing to calculate the signal 
        in only part of the whole 4pi solid angle, because of symmetries. 
        Then we assume that the covariance is reduced by a factor corresponding
        to that allowed by the symmetry, and that the data vector is already
        averaged over these modes with the same expected signal. In this case,
        Nmodes propto Sigma is multiplied by (4pi)/total_omega, we put this 
        facotr here in Sigma_scaled_to_4pi, dividing by total_omega_over_4pi."""
        total_omega_over_4pi = (self._max_cos_theta1 - self._min_cos_theta1) \
            * (self._max_phi12 - self._min_phi12)/(4.*np.pi)
        return (self.Sigma/total_omega_over_4pi)

    def _setup_angles(self, tri_ori_dict):

        """Returns theta1 and phi12 given min max and number of bins of cos(theta1) and phi12. """
        
        if tri_ori_dict['parametrization_name'] == 'theta1_phi12':
            
            (min_cos_theta1, max_cos_theta1, min_phi12, max_phi12) = self._get_min_max_angles(tri_ori_dict)
            
            self._logger.info('Using cos_theta1 in [{}, {}]'.format(min_cos_theta1, max_cos_theta1))
            self._logger.info('Using phi12 in [{}, {}]'.format(min_phi12, max_phi12))
            
            cos_theta1 = self._get_bin_centers_from_nbin(min_cos_theta1, max_cos_theta1, tri_ori_dict['nbin_cos_theta1'])
            theta1 = np.arccos(cos_theta1) 
            phi12 = self._get_bin_centers_from_nbin(min_phi12, max_phi12, tri_ori_dict['nbin_phi12'])

            self._logger.info('theta1_in_deg = {}'.format(theta1 / np.pi * 180.0))
            self._logger.info('phi12 in deg = {}'.format(phi12 / np.pi * 180.0))
            
            return theta1, phi12, cos_theta1
    
    def _get_min_max_angles(self, tri_ori_dict):

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
    


