import os
import numpy as np
import scipy
import sys

from lss_theory.scripts.get_ps import get_data_vec_p3d
from lss_theory.scripts.get_b3d_rsd import get_b3d_rsd
from lss_theory.data_vector import Bispectrum3DRSD
from lss_theory.utils.logging import class_logger
from lss_theory.utils import file_tools
from lss_theory.utils import profiler
from lss_theory.math_utils import matrix_utils


def check_matrix_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

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
            ['Bispectrum3DRSD']
        self._do_unique_multitracer = self._info['Bispectrum3DRSDCovariance']\
            ['Bispectrum3DRSD']['do_unique_multitracer']
        self._plot_dir = self._info['plot_dir']
        self._output_dir = self._info['output_dir']

        file_tools.mkdir_p(self._plot_dir)
        file_tools.mkdir_p(self._output_dir)
        self._run_name = self._info['run_name']

        self._fn_cov = self._get_fn_cov()
        self._fn_invcov = self._get_fn_invcov()

        self._survey_volume_array = self._get_survey_volume_array()
        self._ps_noise = self._get_noise()

        self._b3d_rsd = self._get_b3d()
        self._b3d_rsd_spec = self._b3d_rsd._data_spec

        self._nb = self._b3d_rsd_spec.nb 
        self._nz = self._b3d_rsd_spec.nz
        self._ntri = self._b3d_rsd_spec.ntri
        self._nori = self._b3d_rsd_spec.nori

        self._setup_cov_ingredients()

    def _setup_cov_ingredients(self):
        self._Nmodes = self._get_Nmodes()
        self._cov_rescale = self._get_cov_rescale()
        self._fog_all = self._get_fog_all()
        self._kaiser_all = self._get_kaiser_all()

        #HACK
        self._fog_all = np.ones_like(self._fog_all)

        self._ps_noise_all = self._get_ps_noise_all()
        self._theory_ps_nb_x_nb_block = self._get_theory_ps_nb_x_nb_block()

    @profiler
    def get_invcov(self):
        nz = self._b3d_rsd_spec.nz
        ntri = self._b3d_rsd_spec.ntri
        nori = self._b3d_rsd_spec.nori

        invcov = np.zeros_like(self.cov)

        if len(self.cov.shape) == 5:
            for iz in range(nz):
                for itri in range(ntri):
                    for iori in range(nori):
                        #self.logger.debug('iz = {}, itri = {}, iori = {}'.format(iz, itri, iori))
                        #self.logger.debug('self.cov[:, :, iz, itri, iori] = {}'.format(self.cov[:, :, iz, itri, iori]))
                        #self.logger.debug('invcov[:, :, iz, itri, iori] = {}'.format(invcov[:, :, iz, itri, iori]))
                        invcov[:, :, iz, itri, iori] = scipy.linalg.inv(self.cov[:, :, iz, itri, iori])

        elif len(self.cov.shape) == 4:
            for iz in range(nz):
                for itri in range(ntri):
                    #self.logger.debug('iz = {}, itri = {}'.format(iz, itri))
                    #self.logger.debug('self.cov[:, :, iz, itri] = {}'.format(self.cov[:, :, iz, itri]))
                    #self.logger.debug('invcov[:, :, iz, itri] = {}'.format(invcov[:, :, iz, itri]))
                    #print('self.cov[-1, :, iz, itri] = {}'.format(self.cov[-1, :, iz, itri]))
                    print('iz = {}, itri = {}'.format(iz, itri))
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

    def get_and_save_cov(self, fn=None, cov_type='diag', do_invcov=True):
        
        fn = fn or self._fn_invcov

        if cov_type == 'full':
            self.cov = self.get_cov(do_invcov)
        elif cov_type == 'diag_in_orientation':
            self.cov = self.get_cov_diagonal_in_triangle_orientation(do_invcov)
        else:
            raise NotImplementedError
        
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

    @profiler
    def get_cov(self, do_invcov):
        """Returns the non-diagonal blocks of the covariance for the b3d_rsd
        data vector in the shape (nb*nori, nb*nori, nzi, ntri), where 
        cov[:,:, iz, itri] corresponds to the smallest non-diagonal block.
        """ 
        
        block_size = self._b3d_rsd_spec.nb * self._b3d_rsd_spec.nori
        nz = self._b3d_rsd_spec.nz
        ntri = self._b3d_rsd_spec.ntri

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

    #@profiler
    def get_cov_diagonal_in_triangle_orientation(self, do_invcov):
        """Returns the non-diagonal blocks of the covariance for the b3d_rsd
        data vector in the shape (nb, nb, nzi, ntri, nori), where 
        cov[:,:, iz, itri, iori] corresponds to the smallest non-diagonal block
        when assuming no covariance between triangles of same shape but 
        different orientations (which is not true in general).
        """ 
        
        nb = self._nb
        nz = self._nz
        ntri = self._ntri
        nori = self._nori

        cov = np.zeros((nb, nb, nz, ntri, nori))
        
        if do_invcov == True:
            self.invcov = np.zeros((nb, nb, nz, ntri, nori))

        for iz in range(nz):
            for itri in range(ntri):
                for iori in range(nori):

                    print('iz = {}, itri = {}, iori = {}'.format(iz, itri, iori))

                    cov_tmp = self.get_cov_nb_x_nb_block(iz, itri, iori, iori)\
                        * self._cov_rescale[iz, itri]
    
                    cov[:, :, iz, itri, iori] = cov_tmp

                    if do_invcov == True:
                        try:
                            #HACK decide if want this? cuts mode off
                            #self.invcov[:, :, iz, itri, iori] = scipy.linalg.inv(cov_tmp)
                            self.invcov[:, :, iz, itri, iori] = \
                                matrix_utils.invert_with_zero_cols_and_rows(\
                                    cov_tmp, rows_null=None, fill_value=0) 
                    
                        except np.linalg.LinAlgError as e:
                            print('Got error: {}'.format(e))
                            print('cov_tmp = {}'.format(cov_tmp))

                    self.logger.debug('iz={}, itri={}, ib=0, iori={}: cov = {}'.format(iz, itri, iori, cov[0, 0, iz, itri, iori]))
                    #self.logger.debug('cov_tmp = {}'.format(cov_tmp))
                    #self.logger.debug('invcov[:, :, iz, itri, iori] = {}'.format(invcov[:, :, iz, itri, iori]))
        return cov

    def _get_cov_rescale(self):
        """Returns 2d numpy array of shape (nz, ntri) for scaling each non-diagonal block of cov."""
        cov_rescale = self._survey_volume_array[:, np.newaxis] / self._Nmodes
        return cov_rescale
    
    def _get_p3d(self):
        info_p3d = self._info.copy()
        info_p3d['PowerSpectrum3D'] = self._info['Bispectrum3DRSDCovariance']['PowerSpectrum3D'] 
        self.logger.debug('info_p3d = {}'.format(info_p3d))
        p3d = get_data_vec_p3d(info_p3d)
        return p3d

    def _get_b3d(self):
        info_b3d = self._info.copy()
        info_b3d['Bispectrum3DRSD'] = self._info['Bispectrum3DRSDCovariance']['Bispectrum3DRSD'] 
        b3d = get_b3d_rsd(info_b3d)
        return b3d
    
    def _get_fn_cov(self):
        fn = os.path.join(self._output_dir, self._info['run_name'] + 'cov.npy')
        return fn
    
    def _get_fn_invcov(self):
        fn = os.path.join(self._output_dir, self._info['run_name'] + 'invcov.npy')
        return fn

    def _get_noise(self):
        """"Returns 2d numpy array of shape (nsample, nz) for the power spectrum
        noise, where noise = 1/number density, and is in unit of (Mpc)^3."""

        number_density = self._p3d._grs_ingredients._get_number_density_array()
        noise = 1./number_density

        return noise

    #@profiler
    def get_cov_smallest_nondiagonal_block(self, iz, itri):
        """Returns (norixnb, norixnb) block; 
        Do not use though, these matrices are ill-conditioned
        """
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
                
        assert check_matrix_symmetric(cov)
        #TODO clean up later
        #is_symmetric = check_matrix_symmetric(cov)
        #print('get_cov_smallest_nondiagonal_block: is_symmetric = {}'.format(is_symmetric))
        #if is_symmetric is False:
        #    print('cov = {}'.format(cov))
        return cov

    def _get_nmode_rescale_from_kpara_cut(self):

        """Returns the fraction of triangles orientations (of the same triangle shape
        (k1, k2, k3)) that are not cutoff due to photo-z error as a 4d numpy array 
        of shape (nb, nb, nz, ntri); use as a rescale (1 means nothing is cutoff).
        """

        nb = self._nb
        nz = self._nz
        ntri = self._ntri
        nori = self._nori

        is_cutoff = self._get_is_cutoff_from_kpara_cut()

        axis_triangle_orientation = 4
        fraction_cutoff = np.sum(is_cutoff, axis=axis_triangle_orientation)/nori 
        nmodes_rescale = 1. - fraction_cutoff
        
        expected_shape = (nb, nb, nz, ntri)
        assert nmodes_rescale.shape == expected_shape

        print('nmodes_rescale = ', nmodes_rescale)

        return nmodes_rescale

    def _get_is_cutoff_from_kpara_cut(self):

        """Returns integer array of 1 and 0 to indicate if mode is cutoff or not
        due to photo-z error as a 5d numpy array of shape (nb, nb, nz, ntri, nori); 
        """

        nb = self._nb
        nz = self._nz
        ntri = self._ntri
        nori = self._nori

        is_cutoff = np.zeros((nb, nb, nz, ntri, nori))
        
        tri_array = self._b3d_rsd_spec.triangle_spec.tri_array[:,:] # (ntri, 3), 3 for k1, k2, k3
        mu_array = self._b3d_rsd_spec.triangle_spec.mu_array[:,:,:] # (ntri, nori, 3), 3 for mu1, mu2, mu3
        kpara_1to3 = tri_array[:, np.newaxis, :] * mu_array # (ntri, nori, 3), 3 for k1mu1, k2mu2, k3mu3

        sigp = self._b3d_rsd._grs_ingredients.get('sigp')  # shape (nsample, nz)

        kpara_1to3_cut = np.zeros((nb, nz, 3))
        for ib in range(nb):
            (is1, is2, is3) = self._b3d_rsd_spec.get_isamples(ib)
            for iz in range(nz):
                kpara_1to3_cut[ib, iz, :] = 2.*np.pi/sigp[[is1, is2, is3], iz]

        # shape (nb, nz, ntri, nori, 3) --> (nb, nz, ntri, nori) after sum
        is_cutoff_1to3 = np.any(\
            kpara_1to3[np.newaxis, np.newaxis, :, :, :] \
            > kpara_1to3_cut[:, :, np.newaxis, np.newaxis, :], \
            axis = -1)
        for ib in range(nb):
            for jb in range(nb):
                is_cutoff_ib = is_cutoff_1to3[ib, :, :, :]
                is_cutoff_jb = is_cutoff_1to3[jb, :, :, :]
                is_cutoff[ib, jb, :, :, :] = np.logical_or(is_cutoff_ib, is_cutoff_jb)

        return is_cutoff 

    def get_cov_nb_x_nb_block(self, iz, itri, iori, jori, do_signal_noise_split=False):
        
        nb = self._b3d_rsd_spec.nb
        if do_signal_noise_split == True:
            cov = np.zeros((nb, nb, 2))
        else:
            cov = np.zeros((nb, nb))

        if iori != jori: 
            return cov

        else:
            for ib in range(nb):
                for jb in range(nb):
                    if do_signal_noise_split == True:
                        (cov[ib, jb, 0], cov[ib, jb, 1]) = self._get_cov_ib_jb(ib, jb, iz, itri, iori, jori, do_signal_noise_split=True)
                    else:
                        cov[ib, jb] = self._get_cov_ib_jb(ib, jb, iz, itri, iori, jori, do_signal_noise_split)
            return cov

    def _get_cov_ib_jb(self, ib, jb, iz, itri, iori, jori, do_signal_noise_split):

        if self._do_unique_multitracer == True:
            return self._get_cov_ib_jb_unique_multitracer(ib, jb, iz, itri, iori, jori, do_signal_noise_split)
        else:
            return self._get_cov_ib_jb_all_multitracer(ib, jb, iz, itri, iori, jori, do_signal_noise_split)
        
    def _get_cov_ib_jb_unique_multitracer(self, ib, jb, iz, itri, iori, jori, do_signal_noise_split=False):

        list_of_isamples = self._get_list_of_isamples_for_ib_unique_multitracer(ib)
        list_of_jsamples = self._get_list_of_isamples_for_ib_unique_multitracer(jb)

        cov_ib_jb = 0.0
        cov_ib_jb_no_noise = 0.0
        cov_ib_jb_noise = 0.0
        
        #print('ib = {}, jb = {}'.format(ib, jb))
        #print('isamples = {}, jsamples = {}'.format(list_of_isamples, list_of_jsamples))
        for isamples in list_of_isamples:
            for jsamples in list_of_jsamples:

                if do_signal_noise_split == True:
                    cov_tmp_no_noise, cov_tmp_noise = self._get_cov_isamples_jsamples(isamples, jsamples, iz, itri, iori, jori, do_signal_noise_split = True)
                    cov_ib_jb_no_noise += cov_tmp_no_noise
                    cov_ib_jb_noise += cov_tmp_noise

                else:
                    cov_tmp = self._get_cov_isamples_jsamples(isamples, jsamples, iz, itri, iori, jori)
                    cov_ib_jb += cov_tmp
                    # TODO not accounting for equilateral and isoceles triangles yet 
        
        if do_signal_noise_split == True:
            return cov_ib_jb_no_noise, cov_ib_jb_noise
        else:
            return cov_ib_jb

    def _get_cov_ib_jb_all_multitracer(self, ib, jb, iz, itri, iori, jori, do_signal_noise_split=False):

        isamples = self._b3d_rsd_spec.get_isamples(ib)
        jsamples = self._b3d_rsd_spec.get_isamples(jb)

        if do_signal_noise_split == True:
            cov_ib_jb_no_noise, cov_ib_jb_noise = \
                self._get_cov_isamples_jsamples(isamples, jsamples, iz, itri, iori, jori,\
                     do_signal_noise_split=True)
            return cov_ib_jb_no_noise, cov_ib_jb_noise

        else:
            cov_ib_jb = self._get_cov_isamples_jsamples(isamples, jsamples, iz, itri, iori, jori)
            # TODO not accounting for equilateral and isoceles triangles yet 
            return cov_ib_jb

    def _get_list_of_isamples_for_ib_unique_multitracer_debug_one_term(self, ib):
        triplet = self._b3d_rsd_spec.get_isamples(ib)
        list_of_isamples = self._get_one_perm_for_a_b_c(*triplet)
        return list_of_isamples

    def _get_list_of_isamples_for_ib_unique_multitracer(self, ib):

        triplet = self._b3d_rsd_spec.get_isamples(ib)

        if self._is_equilateral(*triplet):
            list_of_isamples = self._get_one_perm_for_a_b_c(*triplet)

        elif self._is_isoceles(*triplet):
            list_of_isamples = self._get_cyclic_perm_for_a_b_c(*triplet)

        elif self._is_scalene(*triplet):
            list_of_isamples = self._get_all_perm_for_a_b_c(*triplet)

        return list_of_isamples

    @staticmethod
    def _get_one_perm_for_a_b_c(a, b, c):
        return [(a, b, c)]
    
    @staticmethod
    def _get_cyclic_perm_for_a_b_c(a, b, c):
        return [(a, b, c), (b, c, a), (c, a, b)]
    
    @staticmethod
    def _get_all_perm_for_a_b_c(a, b, c):
        return [(a, b, c), (b, c, a), (c, a, b), \
                (c, b, a), (b, a, c), (a, c, b)] 

    @staticmethod
    def _is_equilateral(a, b, c):
        return ((a == b) and (b == c))
    
    @staticmethod
    def _is_isoceles_a_b(a, b, c):
        return ((a == b) and (a != c))
    
    @staticmethod
    def _is_isoceles_b_c(a, b, c):
        return ((b == c) and (b != a))

    @staticmethod
    def _is_isoceles_a_c(a, b, c):
        return ((a == c) and (a != b))

    def _is_isoceles(self, a, b, c):
        t = (a, b, c)
        iso1 = self._is_isoceles_a_b(*t)
        iso2 = self._is_isoceles_b_c(*t)
        iso3 = self._is_isoceles_a_c(*t)
        return ((iso1 or iso2) or iso3)

    @staticmethod
    def _is_scalene(a, b, c):
        return (((a != b) and (b != c)) and (a != b))

    def _get_cov_isamples_jsamples(self, isamples, jsamples, iz, itri, iori, jori, do_signal_noise_split=False):
        """Can be used for both do_unique_multitracer = True/False."""

        (ik1, ik2, ik3) = self._b3d_rsd_spec.triangle_spec.get_ik1_ik2_ik3_for_itri(itri) #TODO make part of b3d_rsd_spec

        debug_sigp = self._b3d_rsd_spec._debug_sigp

        (a, b, c) = isamples
        (d, e, f) = jsamples

        ips1 = self._p3d._data_spec.get_ips(a, d)
        ips2 = self._p3d._data_spec.get_ips(b, e)
        ips3 = self._p3d._data_spec.get_ips(c, f)

        if (debug_sigp is None) or (debug_sigp > 0):
            fog_a = self._get_fog(a, iz, ik1, itri, iori, 0)
            fog_b = self._get_fog(b, iz, ik2, itri, iori, 1)
            fog_c = self._get_fog(c, iz, ik3, itri, iori, 2)

            fog_d = self._get_fog(d, iz, ik1, itri, jori, 0)
            fog_e = self._get_fog(e, iz, ik2, itri, jori, 1)
            fog_f = self._get_fog(f, iz, ik3, itri, jori, 2)

            fog1 = fog_a * fog_d
            fog2 = fog_b * fog_e
            fog3 = fog_c * fog_f

        elif debug_sigp == 0:

            fog1 = fog2 = fog3 = 1.0

        kaiser_a = self._kaiser_all[a, iz, ik1, itri, iori, 0]
        kaiser_b = self._kaiser_all[b, iz, ik2, itri, iori, 1]
        kaiser_c = self._kaiser_all[c, iz, ik3, itri, iori, 2]

        kaiser_d = self._kaiser_all[d, iz, ik1, itri, jori, 0]
        kaiser_e = self._kaiser_all[e, iz, ik2, itri, jori, 1]
        kaiser_f = self._kaiser_all[f, iz, ik3, itri, jori, 2]

        kaiser_1 = kaiser_a * kaiser_d
        kaiser_2 = kaiser_b * kaiser_e
        kaiser_3 = kaiser_c * kaiser_f

        cov_isamples_jsamples = \
              self._get_observed_ps(ips1, iz, ik1, fog=fog1, kaiser=kaiser_1) \
            * self._get_observed_ps(ips2, iz, ik2, fog=fog2, kaiser=kaiser_2) \
            * self._get_observed_ps(ips3, iz, ik3, fog=fog3, kaiser=kaiser_3)

        if do_signal_noise_split == True:
            #P^3 term only
            cov_isamples_jsamples_no_noise = \
                  self._get_observed_ps(ips1, iz, ik1, fog=fog1, kaiser=kaiser_1, no_noise=True) \
                * self._get_observed_ps(ips2, iz, ik2, fog=fog2, kaiser=kaiser_2, no_noise=True) \
                * self._get_observed_ps(ips3, iz, ik3, fog=fog3, kaiser=kaiser_3, no_noise=True)
            cov_isamples_jsamples_noise = cov_isamples_jsamples - cov_isamples_jsamples_no_noise
            return (cov_isamples_jsamples_no_noise, cov_isamples_jsamples_noise)
        else:
            return cov_isamples_jsamples



    #@profiler #too slow,not sure why
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
    
    def _get_observed_ps(self, ips, iz, ik, fog=1, kaiser=1, no_noise=False):
        """Returns a float for the observed galaxy power spectrum with shot noise."""
        ps = self._galaxy_ps[ips, iz, ik] * kaiser * fog 
        if no_noise == False:
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


    def _get_fog(self, isample, iz, ik, itri, iori, itri_side):
        return self._fog_all[isample, iz, ik, itri, iori, itri_side]

    def _get_fog_all(self):

        """Returns a 6d numpy array of shape (nsample, nz, nk, ntri, nori, ntri_side)"""

        debug_sigp = self._b3d_rsd_spec._debug_sigp

        if (debug_sigp is None) or (debug_sigp > 0):

            sigp = self._b3d_rsd._grs_ingredients.get('sigp') 

            k = self._b3d_rsd_spec.k

            #TODO Need to optimize and not compute for all k and tri.
            #TODO Could make a variant of mu_array but for k1mu1, k2mu2, k3mu3 
            # instead of mu1, mu2, mu3 for a given itri and iori.
            # then the return results can have smaller dimension.

            #shape (nk, ntri, nori, ntri_side)
            k1mu1 = k[:, np.newaxis, np.newaxis, np.newaxis] * self._b3d_rsd_spec.triangle_spec.mu_array[:,:,:]
            #shape (nsample, nz, nk, ntri, nori, ntri_side)
            sigp_kmu_squared = (k1mu1[np.newaxis, np.newaxis, :, :, :, :] \
                    * sigp[:, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis])**2 

            fog = np.exp( -0.5 * (sigp_kmu_squared))
    
        elif debug_sigp == 0:

            fog = np.ones((self._b3d_rsd_spec.nsample, self._b3d_rsd_spec.nz, \
                self._b3d_rsd_spec.nk, self._b3d_rsd_spec.ntri, \
                self._b3d_rsd_spec.nori, 3))

        return fog

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
        
        #HACK
        print('survey_volume = ', self._survey_volume_array)

        Nmodes = (V**2)[:, np.newaxis] / (2*np.pi)**6 * K_triangle

        print('K_triangle = ', K_triangle)

        return Nmodes

    def _get_survey_volume_array(self):
        """Returns 1d numpy array of shape (nz,) for the volume of the 
        redshift bins in units of (Mpc)^3 with fsky included."""
        return self._fsky * self._p3d._grs_ingredients._get_survey_volume_array() 

