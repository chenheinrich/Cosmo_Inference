import os
import numpy as np
import scipy

from theory.scripts.get_ps import get_data_vec_p3d
from theory.scripts.get_bis_rsd import get_b3d_rsd
from theory.data_vector import Bispectrum3DRSD
from theory.utils.logging import class_logger
from theory.utils import file_tools
from theory.utils import profiler


class Bispectrum3DRSDCovarianceCalculator():

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
        
        self._fsky = self._info['Bispectrum3DRSDCovariance']['fsky']
        self._do_cvl_noise = self._info['Bispectrum3DRSDCovariance']['do_cvl_noise']
        self._do_folded_signal = self._info['Bispectrum3DRSDCovariance']\
            ['Bispectrum3DRSD']['triangle_orientation_info']['do_folded_signal']
        self._plot_dir = self._info['plot_dir']
        
        file_tools.mkdir_p(self._plot_dir)
        self._fn_cov = self._get_fn_cov()
        self._fn_invcov = self._get_fn_invcov()

        self._survey_volume_array = self._get_survey_volume_array()
        self._ps_noise = self._get_noise()

        self._b3d_rsd = self._get_b3d()
        self._b3d_rsd_spec = self._b3d_rsd._data_spec

        self._setup_cov_ingredients()

    def _setup_cov_ingredients(self):
        self._Nmodes = self._get_Nmodes()
        self._cov_rescale = self._get_cov_rescale()
        self._fog_all = self._get_fog_all()

        self._ps_noise_all = self._get_ps_noise_all()
        self._theory_ps_nb_x_nb_block = self._get_theory_ps_nb_x_nb_block()

    #@profiler
    def get_invcov(self):
        nz = self._b3d_rsd_spec.nz
        ntri = self._b3d_rsd_spec.ntri

        invcov = np.zeros_like(self.cov)

        for iz in range(nz):
            for itri in range(ntri):
                invcov[:, :, iz, itri] = scipy.linalg.inv(self.cov[:, :, iz, itri])
        return invcov

    def get_and_save_invcov(self, fn):
        self.invcov = self.get_invcov()
        self.save_invcov(fn)
        return self.invcov

    def get_and_save_cov(self, fn):
        self.cov = self.get_cov()
        self.save_cov(fn)
        return self.cov

    def save_cov(self, fn):
        self.save_file(fn, self.cov, name='covariance')

    def save_invcov(self, fn):
        self.save_file(fn, self.invcov, name='inverse covariance')

    def save_file(self, fn, data, name = ''):
        np.save(fn, data)
        print('Saved file {}: {}'.format(name, fn))

    @profiler
    def get_cov(self):
        """Returns the non-diagonal blocks of the covariance for the b3d_rsd
        data vector in the shape (nb*nori, nb*nori, nzi, ntri), where 
        cov[:,:, iz, itri] corresponds to the smallest non-diagonal block.
        """ 
        
        block_size = self._b3d_rsd_spec.nb * self._b3d_rsd_spec.nori
        
        nz = self._b3d_rsd_spec.nz
        ntri = self._b3d_rsd_spec.ntri
        cov = np.zeros((block_size, block_size, nz, ntri))

        iblock = 0
        for iz in range(nz):
            for itri in range(ntri):
                cov[:, :, iz, itri] = self.get_cov_smallest_nondiagonal_block(iz, itri)
                cov = cov * self._cov_rescale[iblock]
                print('iz=%s, itri=%i, ib=0, imu=0'%(iz, itri), cov[0, 0, iz, itri])
                iblock = iblock + 1
        return cov

    #TODO add test for cov:
    # no FOG, no Kaiser, no AP
    # iz = 0, itri = 0: assert cov[0, 0, iz, itri] == 7.695621720409703e+28
    # iz = 0, itri = 1: assert cov[0, 0, iz, itri] == 3.073008067677029e+28
    # with FOG, no Kaiser, no AP
    # iz = 0, itri = 0: assert cov[0, 0, iz, itri] == 7.694486671443942e+28
    # iz = 0, itri = 1: assert cov[0, 0, iz, itri] == 3.0725297000383867e+28

    def _get_cov_rescale(self):
        """Returns 1d numpy array of size nz*ntri for scaling each non-diagonal block of cov."""
        cov_rescale = np.ravel(self._survey_volume_array[:, np.newaxis] / self._Nmodes)
        return cov_rescale
    
    def _get_p3d(self):
        info_p3d = self._info.copy()
        info_p3d['PowerSpectrum3D'] = self._info['Bispectrum3DRSDCovariance']['PowerSpectrum3D'] 
        print('info_p3d', info_p3d)
        p3d = get_data_vec_p3d(info_p3d)
        return p3d

    def _get_b3d(self):
        info_b3d = self._info.copy()
        info_b3d['Bispectrum3DRSD'] = self._info['Bispectrum3DRSDCovariance']['Bispectrum3DRSD'] 
        b3d = get_b3d_rsd(info_b3d)
        return b3d
    
    def _get_fn_cov(self):
        fn = os.path.join(self._info['plot_dir'], self._info['run_name'] + 'cov.npy')
        return fn
    
    def _get_fn_invcov(self):
        fn = os.path.join(self._info['plot_dir'], self._info['run_name'] + 'invcov.npy')
        return fn

    def _get_noise(self):
        """"Returns 2d numpy array of shape (nsample, nz) for the power spectrum
        noise, where noise = 1/number density, and is in unit of (Mpc)^3."""

        number_density = self._p3d._grs_ingredients._get_number_density_array()
        noise = 1./number_density

        return noise

    @profiler
    def get_cov_smallest_nondiagonal_block(self, iz, itri):

        nori = self._b3d_rsd_spec.nori
        nb = self._b3d_rsd_spec.nb

        cov = np.zeros((nori * nb, nori * nb))
        
        for iori in range(nori):            

            for jori in range(nori):

                Mstart = nb * iori
                Mend = nb * (iori + 1)
                Nstart = nb * jori
                Nend = nb * (jori + 1)

                cov[Mstart:Mend, Nstart:Nend] = self.get_cov_nb_x_nb_block(iz, itri, iori, jori)
                #TODO check if symmetric later
        return cov

    #@profiler
    def get_cov_nb_x_nb_block(self, iz, itri, iori, jori):

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

                cov[ib, jb] = self._get_observed_ps(ips1, iz, ik1, fog=fog_a*fog_d) \
                    * self._get_observed_ps(ips2, iz, ik2, fog=fog_b*fog_e) \
                    * self._get_observed_ps(ips3, iz, ik3, fog=fog_c*fog_f)

                # TODO not accounting for equilateral and isoceles triangles yet 
                # TODO for these other terms in the cov, will need to change fog definition

        return cov

    @profiler #too slow,not sure why
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

    def _get_theory_ps_nb_x_nb_block(self):

        """Returns 4-d numpy array of shape (nb, nb, nchannel, nz, ntri) where
        nchannel = 3 right now is for the three power spectra to be multiplied
        together after taking into fog and noise in the first term of the bispectrum
        covariance (more could be added if more terms in the bispectrum are needed.)"""

        nb = self._b3d_rsd_spec.nb
        nz = self._b3d_rsd_spec.nz
        ntri = self._b3d_rsd_spec.ntri
        nchannel = 3

        cov = np.zeros((nb, nb, nchannel, nz, ntri))

        #ik1, ik2, ik3 has size ntri
        (ik1, ik2, ik3) = self._b3d_rsd_spec.triangle_spec.get_ik1_ik2_ik3() 

        for ib in range(nb):
            (a, b, c) = self._b3d_rsd_spec.get_isamples(ib)

            for jb in range(nb):
                (d, e, f) = self._b3d_rsd_spec.get_isamples(jb)

                ips1 = self._p3d._data_spec.get_ips(a, d)
                ips2 = self._p3d._data_spec.get_ips(b, e)
                ips3 = self._p3d._data_spec.get_ips(c, f)

                # galaxy_ps has shape (nps, nz, nk, nmu)
                cov[ib, jb, 0, :, :] = np.transpose(self._galaxy_ps[ips1, :, ik1]) 
                cov[ib, jb, 1, :, :] = np.transpose(self._galaxy_ps[ips2, :, ik2])
                cov[ib, jb, 2, :, :] = np.transpose(self._galaxy_ps[ips3, :, ik3])

        return cov

    def _get_sigp(self, isample, iz):
        return self._b3d_rsd._grs_ingredients.get('sigp')[isample, iz]
    
    def _get_observed_ps(self, ips, iz, ik, fog=1):
        """Returns a float for the observed galaxy power spectrum with shot noise."""
        ps = self._galaxy_ps[ips, iz, ik] * fog #TODO add Kaiser too 
        ps += self._ps_noise_all[ips, iz]
        return ps

        # TODO might want to check that old self._galaxy_ps[ips, iz, ik, 0] gave the same as this self._galaxy_ps[ips, iz, ik] ?

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


    def _get_fog(self, isample, iz, ik, itri, iori, itri_side):
        return self._fog_all[isample, iz, ik, itri, iori, itri_side]

    def _get_fog_all(self):

        """Returns a 3d numpy array of shape (nsample, nz, nk, ntri, nori, ntri_side)"""

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
    
    def _get_Nmodes(self): #TODO double check this
        """1d numpy array of shape (nz, ntri) for the number of modes in each triangle shape bin."""

        if self._do_folded_signal is True:
            nori = self._b3d_rsd_spec.triangle_spec.nori 
            Sigma = self._b3d_rsd_spec.Sigma_scaled_to_4pi
            assert np.allclose(nori * Sigma, 1)
        else:
            Sigma = self._b3d_rsd_spec.Sigma

        (k1, k2, k3) = self._b3d_rsd_spec.triangle_spec.get_k1_k2_k3()
        (dk1, dk2, dk3) = self._b3d_rsd_spec.get_dk1_dk2_dk3()

        K_triangle = 8.0 * (np.pi * np.pi) * (k1 * k2 * k3) * (dk1 * dk2 * dk3) * Sigma
        V = self._survey_volume_array

        Nmodes = (V**2)[:, np.newaxis] / (2*np.pi)**6 * K_triangle

        return Nmodes

    def _get_survey_volume_array(self):
        """Returns 1d numpy array of shape (nz,) for the volume of the 
        redshift bins in units of (Mpc)^3 with fsky included."""
        return self._fsky * self._p3d._grs_ingredients._get_survey_volume_array() 

