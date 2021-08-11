import os
import numpy as np
import scipy
import sys

from lss_theory.data_vector import Bispectrum3DRSD
from lss_theory.utils.logging import class_logger
from lss_theory.utils import file_tools
from lss_theory.utils import profiler

#TODO need to refactor together with b3d_rsd_covariance.py
# many similarities

def check_matrix_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

class Bispectrum3DBaseCovarianceCalculator():

    def __init__(self, info):

        self.logger = class_logger(self)
        self._info = info

        # TODO test consistency p and b:
        #[PowerSpectrum3D]
        #nk: 11 # number of k points (to be changed into bins)
        #nmu: -1 # number of mu bins
        #kmin: 0.0007 # equivalent to 0.001 h/Mpc
        #kmax: 0.14 # equivalent to 0.2 h/Mpc

        self._p3d = self._get_p3d()
        self._galaxy_ps = self._p3d.get('galaxy_ps_without_AP_no_fog_no_kaiser') # (nsample, nz, nk)
        
        #changed
        self._fsky = self._info['Bispectrum3DBaseCovariance']['fsky']
        self._do_cvl_noise = self._info['Bispectrum3DBaseCovariance']['do_cvl_noise']
        self._plot_dir = self._info['plot_dir']
        self._result_dir = self._info['result_dir']
        
        file_tools.mkdir_p(self._plot_dir)
        file_tools.mkdir_p(self._result_dir)
        self._run_name = self._info['run_name']
        
        self._fn_cov = self._get_fn_cov()
        self._fn_invcov = self._get_fn_invcov()

        self._survey_volume_array = self._get_survey_volume_array()
        self._ps_noise = self._get_noise()

        self._b3d = self._get_b3d()
        self._b3d_spec = self._b3d._data_spec

        self._setup_cov_ingredients()

    #changed
    def _setup_cov_ingredients(self):
        self._Nmodes = self._get_Nmodes()
        self._cov_rescale = self._get_cov_rescale()
        self._ps_noise_all = self._get_ps_noise_all()

    #changed
    @profiler
    def get_invcov(self):
        nz = self._b3d_spec.nz
        ntri = self._b3d_spec.ntri

        invcov = np.zeros_like(self.cov)

        if len(self.cov.shape) == 5:
            for iz in range(nz):
                for itri in range(ntri):
                    invcov[:, :, iz, itri] = scipy.linalg.inv(self.cov[:, :, iz, itri])

        elif len(self.cov.shape) == 4:
            for iz in range(nz):
                for itri in range(ntri):
                    try:
                        invcov[:, :, iz, itri] = scipy.linalg.inv(self.cov[:, :, iz, itri])
                    except np.linalg.LinAlgError as e:
                        self.logger.info(e)
                        self.logger.info('cov[:,:,iz,itri] = {}'.format(self.cov[:,:,iz,itri]))

        return invcov

    def get_and_save_invcov(self, fn=None):
        self.invcov = self.get_invcov()
        fn = fn or self._fn_invcov
        self.save_invcov(fn)
        return self.invcov

    #changed
    def get_and_save_cov(self, fn=None, do_invcov=True):
        self.cov = self.get_cov(do_invcov)
        fn = fn or self._fn_cov
        self.save_cov(fn)
        return self.cov

    def save_cov(self, fn):
        self.save_file(fn, self.cov, name='covariance')

    def save_invcov(self, fn=None):
        fn = fn or self._fn_invcov
        self.save_file(fn, self.invcov, name='inverse covariance')

    def save_file(self, fn, data, name = ''):
        np.save(fn, data)
        self.logger.info('Saved file {}: {}'.format(name, fn))

    def load_cov_from_fn(self, fn):
        if hasattr(self, 'cov'):
            self.logger.error('load_cov_from_fn: self.cov already exists.')
            sys.exit()
        else:
            self.cov = np.load(fn)

    def load_invcov_from_fn(self, fn):
        if hasattr(self, 'invcov'):
            self.logger.error('load_invcov_from_fn: self.invcov already exists.')
            sys.exit()
        else:
            self.invcov = np.load(fn)

    #changed
    @profiler
    def get_cov(self, do_invcov):
        """Returns the non-diagonal blocks of the covariance for the b3d_base
        data vector in the shape (nb, nb, nz, ntri), where 
        cov[:,:, iz, itri] corresponds to the smallest non-diagonal block.
        """ 
        
        block_size = self._b3d_spec.nb 
        nz = self._b3d_spec.nz
        ntri = self._b3d_spec.ntri

        cov = np.zeros((block_size, block_size, nz, ntri))

        if do_invcov is True:
            self.invcov = np.zeros((block_size, block_size, nz, ntri))

        for iz in range(nz):
            for itri in range(ntri):

                print('iz = {}, itri = {}'.format(iz, itri))

                cov_tmp = self.get_cov_smallest_nondiagonal_block(iz, itri)\
                    * self._cov_rescale[iz, itri]
                cov[:, :, iz, itri] = cov_tmp

                print('self.cov[-1, :, iz, itri] = {}'.format(cov_tmp))

                if do_invcov is True:
                    try:
                        self.invcov[:, :, iz, itri] = scipy.linalg.inv(cov_tmp)
                    except np.linalg.LinAlgError as e:
                        #self.logger.info('{}'.format(e))
                        self.logger.info('cov[:,:,iz,itri] = {}'.format(cov_tmp))
                    
                    assert check_matrix_symmetric(self.invcov[:, :, iz, itri])
                #self.logger.debug('iz={}, itri={}, ib=0: {}'.format(iz, itri, cov[0, 0, iz, itri]))
                #self.logger.debug('cov_tmp = {}'.format(cov_tmp))
                #self.logger.debug('invcov[:, :, iz, itri] = {}'.format(invcov[:, :, iz, itri]))
        return cov

    #TODO add test for cov:
    # no FOG, no Kaiser, no AP
    # iz = 0, itri = 0: assert cov[0, 0, iz, itri] == 7.695621720409703e+28
    # iz = 0, itri = 1: assert cov[0, 0, iz, itri] == 3.073008067677029e+28
    # with FOG, no Kaiser, no AP
    # iz = 0, itri = 0: assert cov[0, 0, iz, itri] == 7.694486671443942e+28
    # iz = 0, itri = 1: assert cov[0, 0, iz, itri] == 3.0725297000383867e+28

    def _get_cov_rescale(self):
        """Returns 2d numpy array of shape (nz, ntri) for scaling each non-diagonal block of cov."""
        cov_rescale = self._survey_volume_array[:, np.newaxis] / self._Nmodes
        return cov_rescale
    
    def _get_p3d(self):
        from lss_theory.scripts.get_ps import get_data_vec_p3d
        info_p3d = self._info.copy()
        #changed
        info_p3d['PowerSpectrum3D'] = self._info['Bispectrum3DBaseCovariance']['PowerSpectrum3D'] 
        self.logger.debug('info_p3d = {}'.format(info_p3d))
        p3d = get_data_vec_p3d(info_p3d)
        return p3d

    #changed
    def _get_b3d(self):
        from lss_theory.scripts. import get_b3d_base
        info_b3d = self._info.copy()
        info_b3d['Bispectrum3DBase'] = self._info['Bispectrum3DBaseCovariance']['Bispectrum3DBase'] 
        b3d = get_b3d_base(info_b3d)
        return b3d
    
    def _get_fn_cov(self):
        fn = os.path.join(self._result_dir, 'cov.npy')
        return fn
    
    def _get_fn_invcov(self):
        fn = os.path.join(self._result_dir, 'invcov.npy')
        return fn

    def _get_noise(self):
        """"Returns 2d numpy array of shape (nsample, nz) for the power spectrum
        noise, where noise = 1/number density, and is in unit of (Mpc)^3."""

        number_density = self._p3d._grs_ingredients._get_number_density_array()
        noise = 1./number_density

        return noise

    #changed
    #@profiler
    def get_cov_smallest_nondiagonal_block(self, iz, itri):
        cov = self.get_cov_nb_x_nb_block(iz, itri)   
        assert check_matrix_symmetric(cov)     
        return cov

    #changed
    #@profiler
    def get_cov_nb_x_nb_block(self, iz, itri):

        (ik1, ik2, ik3) = self._b3d_spec.triangle_spec.get_ik1_ik2_ik3_for_itri(itri) #TODO make part of b3d_rsd_spec

        nb = self._b3d_spec.nb

        cov = np.zeros((nb, nb))

        for ib in range(nb):

            (a, b, c) = self._b3d_spec.get_isamples(ib)

            for jb in range(nb):

                (d, e, f) = self._b3d_spec.get_isamples(jb)

                ips1 = self._p3d._data_spec.get_ips(a, d)
                ips2 = self._p3d._data_spec.get_ips(b, e)
                ips3 = self._p3d._data_spec.get_ips(c, f)

                cov[ib, jb] = self._get_observed_ps(ips1, iz, ik1) \
                    * self._get_observed_ps(ips2, iz, ik2) \
                    * self._get_observed_ps(ips3, iz, ik3)

                # TODO not accounting for equilateral and isoceles triangles yet 
                # TODO for these other terms in the cov, will need to change fog definition

        return cov

    #leave for now (not used)
    #@profiler too slow, not sure why
    def get_cov_nb_x_nb_block2(self, iz, itri, iori, jori):

        (ik1, ik2, ik3) = self._b3d_rsd_spec.triangle_spec.get_ik1_ik2_ik3_for_itri(itri) #TODO make part of b3d_rsd_spec

        nb = self._b3d_rsd_spec.nb

        cov = np.zeros((nb, nb))

        for ib in range(nb):

            (a, b, c) = self._b3d_rsd_spec.get_isamples(ib)

            for jb in range(nb):

                (d, e, f) = self._b3d_rsd_spec.get_isamples(jb)

                ips1 = self._p3d._data_spec.get_ips(a, d)
                ips2 = self._p3d._data_spec.get_ips(b, e)
                ips3 = self._p3d._data_spec.get_ips(c, f)

                fog_a = self._get_fog(a, iz, ik1, itri, iori, 0)
                fog_b = self._get_fog(b, iz, ik2, itri, iori, 1)
                fog_c = self._get_fog(c, iz, ik3, itri, iori, 2)

                fog_d = self._get_fog(d, iz, ik1, itri, jori, 0)
                fog_e = self._get_fog(e, iz, ik2, itri, jori, 1)
                fog_f = self._get_fog(f, iz, ik3, itri, jori, 2)

                cov = self._theory_ps_nb_x_nb_block[:, :, 0, iz, itri] * fog_a * fog_d \
                    + self._theory_ps_nb_x_nb_block[:, :, 1, iz, itri] * fog_b * fog_e \
                    + self._theory_ps_nb_x_nb_block[:, :, 2, iz, itri] * fog_c * fog_f
                # TODO not accounting for equilateral and isoceles triangles yet 
                # TODO for these other terms in the cov, will need to change fog definition

        return cov

    #leave for now (not used, too slow)
    def _get_theory_ps_nb_x_nb_block(self):

        """Returns 4-d numpy array of shape (nb, nb, nchannel, nz, ntri) where
        nchannel = 3 right now is for the three power spectra to be multiplied
        together after taking into fog and noise in the first term of the bispectrum
        covariance (more could be added if more terms in the bispectrum are needed.)"""

        nb = self._b3d_spec.nb
        nz = self._b3d_spec.nz
        ntri = self._b3d_spec.ntri
        nchannel = 3

        cov = np.zeros((nb, nb, nchannel, nz, ntri))

        #ik1, ik2, ik3 has size ntri
        (ik1, ik2, ik3) = self._b3d_spec.triangle_spec.get_ik1_ik2_ik3() 

        for ib in range(nb):
            (a, b, c) = self._b3d_spec.get_isamples(ib)

            for jb in range(nb):
                (d, e, f) = self._b3d_spec.get_isamples(jb)

                ips1 = self._p3d._data_spec.get_ips(a, d)
                ips2 = self._p3d._data_spec.get_ips(b, e)
                ips3 = self._p3d._data_spec.get_ips(c, f)

                # galaxy_ps has shape (nps, nz, nk, nmu)
                cov[ib, jb, 0, :, :] = np.transpose(self._galaxy_ps[ips1, :, ik1]) 
                cov[ib, jb, 1, :, :] = np.transpose(self._galaxy_ps[ips2, :, ik2])
                cov[ib, jb, 2, :, :] = np.transpose(self._galaxy_ps[ips3, :, ik3])

        return cov

    #changed in rsd version only
    def _get_sigp(self, isample, iz):
        return self._b3d._grs_ingredients.get('sigp')[isample, iz]
    
    def _get_observed_ps(self, ips, iz, ik, fog=1, kaiser=1):
        """Returns a float for the observed galaxy power spectrum with shot noise."""
        ps = self._galaxy_ps[ips, iz, ik] * kaiser * fog 
        ps += self._ps_noise_all[ips, iz]
        return ps

    def _get_ps_noise_all(self):
        """Returns d numpy array of shape (nps, nz) for galaxy noise spectrum. """

        nps = self._p3d._data_spec.nps
        nz = self._p3d._data_spec.nz

        ps_noise_all = np.zeros((nps, nz))

        for ips in range(nps):
            (isample1, isample2) = self._p3d._data_spec.get_isamples(ips)
            if isample1 == isample2:
                if self._do_cvl_noise is False:
                    ps_noise_all[ips,:] = self._ps_noise[isample1, :]

        return ps_noise_all

    #changed (in rsd version only)
    def _get_fog(self, isample, iz, ik, itri, iori, itri_side):
        return self._fog_all[isample, iz, ik, itri, iori, itri_side]

    #changed (in rsd version only)
    def _get_fog_all(self):

        """Returns a 6d numpy array of shape (nsample, nz, nk, ntri, nori, ntri_side)"""

        fog = np.zeros((self._b3d_rsd_spec.nsample, self._b3d_rsd_spec.nz, \
            self._b3d_rsd_spec.nk, self._b3d_rsd_spec.ntri, self._b3d_rsd_spec.nori))
        
        sigp = self._b3d_rsd._grs_ingredients.get('sigp') 

        k = self._b3d_rsd_spec.k

        #TODO Need to optimize and not compute for all k and tri.
        #shape (nk, ntri, nori, ntri_side)
        k1mu1 = k[:, np.newaxis, np.newaxis, np.newaxis] * self._b3d_rsd_spec.triangle_spec.mu_array[:,:,:]
        #shape (nsample, nz, nk, ntri, nori, ntri_side)
        sigp_kmu_squared = (k1mu1[np.newaxis, np.newaxis, :, :, :, :] \
                * sigp[:, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis])**2 

        fog = np.exp( -0.5 * (sigp_kmu_squared))

        return fog

    #changed (in rsd version only)
    def _get_kaiser_all(self):

        """Returns a 6d numpy array of shape (nsample, nz, nk, ntri, nori, ntri_side)"""

        expected_shape = (self._b3d_rsd_spec.nsample, self._b3d_rsd_spec.nz, \
            self._b3d_rsd_spec.nk, self._b3d_rsd_spec.ntri, self._b3d_rsd_spec.nori, 3)
        
        f = self._b3d_rsd._grs_ingredients.get('growth_rate_f')
        bias = self._b3d_rsd._grs_ingredients.get('galaxy_bias_without_AP') #(nsample, nz, nk)

        #TODO Need to optimize and not compute for all k and tri.
        #shape (nk, ntri, nori, ntri_side)
        mu_squared = self._b3d_rsd_spec.triangle_spec.mu_array[:,:,:] ** 2
        f_over_b = f[np.newaxis, :, np.newaxis] / bias
        kaiser = 1.0 + f_over_b[:, :, :, np.newaxis, np.newaxis, np.newaxis] \
            * mu_squared[np.newaxis, np.newaxis, np.newaxis, :, :, :]

        assert kaiser.shape == expected_shape, (kaiser.shape, expected_shape)

        return kaiser
    
    #changed
    def _get_Nmodes(self): #TODO double check this
        """1d numpy array of shape (nz, ntri) for the number of modes in each triangle shape bin."""

        (k1, k2, k3) = self._b3d_spec.triangle_spec.get_k1_k2_k3()
        (dk1, dk2, dk3) = self._b3d_spec.get_dk1_dk2_dk3()

        K_triangle = 8.0 * (np.pi * np.pi) * (k1 * k2 * k3) * (dk1 * dk2 * dk3) 
        V = self._survey_volume_array

        Nmodes = (V**2)[:, np.newaxis] / (2*np.pi)**6 * K_triangle

        return Nmodes

    def _get_survey_volume_array(self):
        """Returns 1d numpy array of shape (nz,) for the volume of the 
        redshift bins in units of (Mpc)^3 with fsky included."""
        return self._fsky * self._p3d._grs_ingredients._get_survey_volume_array() 

