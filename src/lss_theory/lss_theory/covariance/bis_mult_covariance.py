from operator import itruediv
import yaml
import numpy as np
import os
import copy
import scipy.linalg as linalg

from lss_theory.data_vector.multipole_data_spec import BispectrumMultipoleSpec
from lss_theory.math_utils.spherical_harmonics import SphericalHarmonicsTable
import lss_theory.math_utils.matrix_utils as matrix_utils
from lss_theory.scripts.get_bis_mult import get_subdir_name_for_bis_mult
from lss_theory.utils.file_tools import mkdir_p
from lss_theory.params.survey_par import SurveyPar
from lss_theory.utils.logging import class_logger

def get_conditional_number(cov):
    w, v = linalg.eig(cov)
    #print('w = {}, v = {}'.format(w,v)) # conditional number 1e10
    conditional_number = np.abs(w[0])/np.abs(w[-1])
    print('conditional number = {:e}'.format(conditional_number))
    return conditional_number
class BispectrumMultipoleCovarianceBase():
    """This class takes in the 3D Fourier galaxy bispectrum covariance
    and performs two integrals over the spherical harmonics to
    return the bispectrum multipole covariance"""

    def __init__(self, info):
        """Args:
        """
        self._logger = class_logger(self)
        self._info = copy.deepcopy(info)
        self._setup_dir()
        self._setup_paths()
        #HACK delete later if not needed
        #self._load_b3d_rsd_cov()
        
        survey_par = SurveyPar(self._info['survey_par_file'])

        self._bis_mult_spec = BispectrumMultipoleSpec(survey_par, self._info['BispectrumMultipole'])
        self._setup_dims()

        self._theta1, self._phi12, self._lmax = self._get_theta1_phi12_lmax()
        self._do_negative_m = self._bis_mult_spec.do_negative_m
        self._ylms_conj = self._get_ylms_conj()
        self._ylms_conj_transpose = np.transpose(self._ylms_conj)

    def _setup_dir(self):
        self._dir = self._info['covariance']['data_dir']
        subdir_name = get_subdir_name_for_bis_mult(self._info)
        self._data_dir = os.path.join(self._dir, subdir_name)
        mkdir_p(self._dir)
        mkdir_p(self._data_dir)

    def _setup_paths(self):
        self._b3d_rsd_cov_path = self._info['covariance']['b3d_rsd_cov_path']
        self._bis_mult_cov_path = os.path.join(self._data_dir, 'cov.npy')
        self._bis_mult_invcov_path = os.path.join(self._data_dir, 'invcov.npy')

    def _load_b3d_rsd_cov(self):
        self._b3d_rsd_cov = np.load(self._b3d_rsd_cov_path)
        #HACK delete later
        #is_cov_type_full = len(self._b3d_rsd_cov.shape) == 4
        #assert is_cov_type_full == True

    def _setup_dims(self):

        self._nb = self._bis_mult_spec.nb
        self._nlm = self._bis_mult_spec.nlm
        self._ntri = self._bis_mult_spec.ntri
        self._nz = self._bis_mult_spec.nz

        print('nlm = {}'.format(self._nlm))
        
        #HACK type cov_full
        #assert self._ntri == self._b3d_rsd_cov.shape[-1]
        #assert self._nz == self._b3d_rsd_cov.shape[-2]
        
        #TODO should be checking from metadata of b3d_rsd_cov; hack for now
        self._ntheta = self._info['BispectrumMultipole']['triangle_orientation_info']['nbin_cos_theta1']
        self._nphi = self._info['BispectrumMultipole']['triangle_orientation_info']['nbin_phi12']
        self._nori = self._ntheta * self._nphi
        
        nbxnori = self._nb * self._nori
        #HACK type cov_full
        #assert nbxnori == self._b3d_rsd_cov.shape[0]
        #HACK restore after test on ylms
        #assert self._b3d_rsd_cov.shape == (self._nb, self._nb, self._nz, self._ntri, self._nori)

    def _get_ylms_conj(self):
        """Returns the 2d numpy array of shape (nori, nlms) for the 
        precomputed spherical harmonics on the theta-phi grid and 
        lmax specified in data_spec.
        """
        spherical_harmonics_table = SphericalHarmonicsTable(self._theta1, self._phi12, self._lmax, \
            self._do_negative_m)
        ylms = spherical_harmonics_table.data

        return np.conj(ylms)

    def _get_theta1_phi12_lmax(self):
        theta1 = self._bis_mult_spec.b3d_rsd_spec.theta1
        phi12 = self._bis_mult_spec.b3d_rsd_spec.phi12
        lmax = self._bis_mult_spec.lmax
        return theta1, phi12, lmax

    def _save(self):
        np.save(self._bis_mult_cov_path, self._cov)
        np.save(self._bis_mult_invcov_path, self._invcov)
        print('Saved cov at {}'.format(self._bis_mult_cov_path))
        print('Saved invcov at {}'.format(self._bis_mult_invcov_path))

class BispectrumMultipoleCovariance(BispectrumMultipoleCovarianceBase):
    """This class takes in the 3D Fourier galaxy bispectrum covariance
    and performs two integrals over the spherical harmonics to
    return the bispectrum multipole covariance"""

    def __init__(self, info):
        
        self._do_unique_multitracer = info['BispectrumMultipole']['do_unique_multitracer']
        self._b3d_cov_calc = self._setup_cov_b3d_rsd(self._do_unique_multitracer)
        super().__init__(info)
        
        self._cov, self._invcov = self._get_cov_and_invcov()
        self._save()
    
    def _get_theta1_phi12_lmax(self):
        theta1 = self._b3d_cov_calc._b3d_rsd_spec.theta1
        phi12 = self._b3d_cov_calc._b3d_rsd_spec.phi12
        lmax = self._bis_mult_spec.lmax
        return theta1, phi12, lmax

    #TODO to delete when new is tested to work
    def _get_cov_and_invcov_old(self):

        nbxnlm = self._nb * self._nlm
        nlm = self._nlm
        shape = (nbxnlm, nbxnlm, self._nz, self._ntri)

        cov = np.zeros(shape, dtype=complex)
        invcov = np.zeros(shape, dtype=complex)
        
        for iz in range(self._nz):
            for itri in range(self._ntri):

                print('iz = {}, itri = {}'.format(iz, itri))
            
                #HACK
                for ib in range(self._nb):
                    for jb in range(ib, self._nb): 
                        
                        print('ib = {}, jb = {}'.format(ib, jb))
                        #HACK
                        #array_nori = self._b3d_rsd_cov[ib, jb, iz, itri, :]
                        #one_block_nori_x_nori = np.diag(array_nori)
                        #dOmega = (4.*np.pi)/(self._nori)
                        #one_block_nlm_x_nlm = self._apply_ylm_integral_on_one_block_nori_x_nori(\
                        #    one_block_nori_x_nori) * dOmega 

                        one_block_nlm_x_nlm = self._get_one_block_nlm_x_nlm(ib, jb, iz, itri)
                        
                        #HACK debugging now:
                        #inv_one_block = linalg.inv(one_block_nlm_x_nlm)
                        #is_passed = matrix_utils.check_matrix_inverse(one_block_nlm_x_nlm, inv_one_block,
                        #    atol=1e-3, feedback_level=1)
                        #assert is_passed

                        #HACK restore later, debug a smaller block first
                        cov[ib*(nlm):(ib+1)*nlm, jb*nlm:(jb+1)*nlm, iz, itri] = one_block_nlm_x_nlm
                        cov[jb*(nlm):(jb+1)*nlm, ib*nlm:(ib+1)*nlm, iz, itri] = one_block_nlm_x_nlm  
                        #matrix_utils.check_matrix_symmetric(one_block_nlm_x_nlm)
                
                print('about to get inverse')
                invcov[:, :, iz, itri] = linalg.inv(cov[:, :, iz, itri])

                #matrix_utils.check_matrix_symmetric(cov[:, :, iz, itri])
                #matrix_utils.check_matrix_symmetric(invcov[:, :, iz, itri])
                matrix_utils.check_matrix_inverse(cov[:,:,iz,itri], \
                    invcov[:,:,iz,itri], atol=1e-6, feedback_level=1)

                #HACK
                sys.exit()

        return cov, invcov

    def _get_cov_and_invcov(self):

        #HACK debug, make input later if useful
        do_signal_noise_split = True 

        nb = self._b3d_cov_calc._nb

        #TODO delete later
        for ib in range(nb):
            isamples = self._b3d_cov_calc._b3d_rsd_spec.get_isamples(ib)
            #print('ib = {}, isamples = {}'.format(ib, isamples))

        nbxnlm = self._nb * self._nlm
        nlm = self._nlm

        shape = (nbxnlm, nbxnlm, self._nz, self._ntri)
        cov = np.zeros(shape, dtype=complex)
        invcov = np.zeros(shape, dtype=complex)

        nori = self._b3d_cov_calc._nori
        blocks_of_nori_x_nori = np.zeros((nb, nb, nori, nori), dtype=complex)
        one_block_nori_x_nori = np.zeros((nori, nori), dtype=complex)

        blocks_of_nori_x_nori_signal_noise_split = np.zeros((nb, nb, 2, nori, nori), dtype=complex)
        one_block_nori_x_nori_signal_noise_split = np.zeros((nori, nori, 2), dtype=complex)
        cov_signal = np.zeros(shape, dtype=complex)
        cov_noise = np.zeros(shape, dtype=complex)

        #HACK
        print('self._nb', self._nb)
        nbxnori = self._nb * nori
        shape2 = (nbxnori, nbxnori, self._nz, self._ntri)
        cov_ori = np.zeros(shape2, dtype=complex)

        #HACK

        itris_equilateral = []
        itris_isoceles = []
        for itri in range(self._ntri):
            ik_triplet = self._b3d_cov_calc._b3d_rsd_spec.triangle_spec.get_ik1_ik2_ik3_for_itri(itri) #TODO make part of b3d_rsd_spec
            print('itri = {}, (ik1, ik2, ik3) = {}'.format(itri, ik_triplet))
            is_equilateral = self._b3d_cov_calc._is_equilateral(*ik_triplet)
            is_isoceles = self._b3d_cov_calc._is_equilateral(*ik_triplet)
            if is_equilateral:
                itris_equilateral.append(itri)
            if is_isoceles:
                itris_isoceles.append(itri)
        
        print('Equilateral itris: {}'.format(itris_equilateral))

        for iz in range(self._nz):
            for itri in range(self._ntri):
            #for itri in itris_isoceles:
                print('iz = {}, itri = {}'.format(iz, itri))

                #HACK remove later test
                #print('testing ylm orthogonality')
                #self._test_ylm_orthogonality(rtol=1e-2, atol=1e-2)

                for iori in range(nori):
                    print('iori=', iori)
                    for jori in range(nori):
                        blocks_of_nori_x_nori_signal_noise_split[:, :, :, iori, jori] = \
                            self._b3d_cov_calc.get_cov_nb_x_nb_block(iz, itri, iori, jori, do_signal_noise_split=True)
                        
                        blocks_of_nori_x_nori[:, :, iori, jori] = \
                            self._b3d_cov_calc.get_cov_nb_x_nb_block(iz, itri, iori, jori, do_signal_noise_split=False)
                
                #assert np.allclose(blocks_of_nori_x_nori, blocks_of_nori_x_nori_signal_noise_split[:,:,0,:,:] + blocks_of_nori_x_nori_signal_noise_split[:,:,1,:,:])

                rescale = np.ones(nb*nb)

                iib = 0
                for ib in range(nb):
                    for jb in range(nb):

                        print('\nib, jb = ({}, {})'.format(ib, jb))
                    
                        one_block_nori_x_nori = blocks_of_nori_x_nori[ib, jb, :, :]
                        cov_ori[ib*(nori):(ib+1)*nori, jb*(nori):(jb+1)*nori, iz, itri] = one_block_nori_x_nori
                        
                        one_block_nlm_x_nlm = self._apply_ylm_integral_on_one_block_nori_x_nori(one_block_nori_x_nori)
                        
                        rank = np.linalg.matrix_rank(one_block_nori_x_nori)
                        print('matrix rank (one_block_nori_x_nori)= {:e}'.format(rank))
                        assert int(rank) == 100
                        rank = np.linalg.matrix_rank(one_block_nlm_x_nlm)
                        print('matrix rank (one_block_nlm_x_nlm) = {:e}'.format(rank))
                        assert int(rank) == 6

                        #HACK debug
                        one_block_nori_x_nori_signal_noise_split = blocks_of_nori_x_nori_signal_noise_split[ib, jb, :, :, :]
                        one_block_nlm_x_nlm_signal = self._apply_ylm_integral_on_one_block_nori_x_nori(one_block_nori_x_nori_signal_noise_split[0,:,:])
                        one_block_nlm_x_nlm_noise = self._apply_ylm_integral_on_one_block_nori_x_nori(one_block_nori_x_nori_signal_noise_split[1,:,:])
                        
                        inv_one_block = linalg.inv(one_block_nlm_x_nlm)

                        print('testing for one_block_nlm_x_nlm')
                        rescale[iib] = np.mean(np.diag(one_block_nori_x_nori))

                        self._test_matrix_and_inverse(one_block_nlm_x_nlm/rescale[iib], inv_one_block*rescale[iib], \
                            atol_inv=1e-5, feedback_level_inv=0, assert_tests=['symmetric_mat', 'symmetric_inv', 'inverse'])

                        #HACK
                        cov[ib*(nlm):(ib+1)*nlm, jb*nlm:(jb+1)*nlm, iz, itri] = one_block_nlm_x_nlm
                        cov_signal[ib*(nlm):(ib+1)*nlm, jb*nlm:(jb+1)*nlm, iz, itri] = one_block_nlm_x_nlm_signal
                        cov_noise[ib*(nlm):(ib+1)*nlm, jb*nlm:(jb+1)*nlm, iz, itri] = one_block_nlm_x_nlm_noise

                        iib = iib + 1

                #HACK debug
                rank = np.linalg.matrix_rank(cov[:,:,iz,itri])
                print('matrix rank (cov[:,:,iz,itri])= {:e}'.format(rank))

                #rank = np.linalg.matrix_rank(cov_ori[:,:,iz,itri])
                #print('matrix rank (cov_ori[:,:,iz,itri])= {:e}'.format(rank))

                #HACK debug
                output_dir = './results/bis_mult/covariance/cosmo_planck2018_fiducial/nk_11/lmax_2/do_folded_signal_False/theta_phi_2_4/35x35_with_triangle_cases_debug_iz_%s_itri_%s/'%(iz, itri)
                mkdir_p(output_dir)

                fn_cov_ori = os.path.join(output_dir, 'cov_ori.npy')
                np.save(fn_cov_ori, cov_ori[:,:,iz,itri])
            
                fn_cov = os.path.join(output_dir, 'cov.npy')
                np.save(fn_cov, cov[:,:,iz,itri])

                fn_cov_noise = os.path.join(output_dir, 'cov_noise.npy')
                np.save(fn_cov_noise, cov_noise[:,:,iz,itri])

                fn_cov_signal = os.path.join(output_dir, 'cov_signal.npy')
                np.save(fn_cov_signal, cov_signal[:,:,iz,itri])

                print('Saved files: {}'.format(fn_cov))
                print('Saved files: {}'.format(fn_cov_ori))
                print('Saved files: {}'.format(fn_cov_signal))
                print('Saved files: {}'.format(fn_cov_noise))

                print('testing for entire nbxnlm, nbxnlm block')

                self._test_matrix_and_inverse(cov[:, :, iz, itri], invcov[:, :, iz, itri], 
                    atol_inv=1e-3, feedback_level_inv=1, assert_tests=[]) #assert_tests=['inverse']
        
        return cov, invcov

    def _test_matrix_and_inverse(self, mat, inv, atol_inv=1e-3, feedback_level_inv=0,\
            assert_tests=['symmetric_mat', 'symmetric_inv', 'inverse']):
        """Checks through assertion if a matrix (mat) and its inverse (inv) 
        pass the symmetric matrix test individual and the inverse tests.
        To select a subset of tests, use the flag assert_tests, e.g.
        assert_tests = ['inverse']"""

        is_symmetric_cov_passed = matrix_utils.check_matrix_symmetric(mat)
        is_symmetric_invcov_passed = matrix_utils.check_matrix_symmetric(inv)
        is_inverse_test_passed = matrix_utils.check_matrix_inverse(mat, \
            inv, atol=atol_inv, feedback_level=feedback_level_inv)

        self._logger.info('is_symmetric_cov_passed = {}'.format(is_symmetric_cov_passed))
        self._logger.info('is_symmetric_invcov_passed = {}'.format(is_symmetric_invcov_passed))
        self._logger.info('is_inverse_test_passed = {}'.format(is_inverse_test_passed))

        if 'symmetric_mat' in assert_tests:
             assert is_symmetric_cov_passed
        if 'symmetric_inv' in assert_tests:
            assert is_symmetric_invcov_passed
        if 'inverse' in assert_tests:
            assert is_inverse_test_passed
        

    def _apply_ylm_integral_on_one_block_nori_x_nori(self, one_block_nori_x_nori):
        """Performs integral \int dOmega Ylm* Cov Ylm"""

        mean_cov = np.mean(np.abs(one_block_nori_x_nori))
        mean_ylms = np.mean(np.abs(self._ylms_conj))
        min_ylms = np.min(np.abs(self._ylms_conj))
        max_ylms = np.max(np.abs(self._ylms_conj))
        print('mean_cov, mean_ylms, min_ylms, max_ylms: ', mean_cov, mean_ylms, min_ylms, max_ylms)

        cov = np.matmul(one_block_nori_x_nori, self._ylms_conj) 
        cov = np.matmul(self._ylms_conj_transpose, cov)
        dOmega = self._b3d_cov_calc._b3d_rsd_spec.dOmega
        cov = cov * dOmega
        
        #TODO need to add some conversion factor here:
        # 2) Calculate the number of modes here with Volume

        return cov

    def _test_ylm_orthogonality(self, rtol=1e-2, atol=1e-2):
        nori = self._b3d_cov_calc._nori
        one_block_nori_x_nori = np.identity(nori)
        cov = np.matmul(one_block_nori_x_nori, self._ylms_conj) 
        cov = np.matmul(np.conj(self._ylms_conj_transpose), cov)
        dOmega = self._b3d_cov_calc._b3d_rsd_spec.dOmega
        cov = cov * dOmega
        nlm = self._ylms_conj.shape[1]
        
        is_symmetric = matrix_utils.check_matrix_symmetric(cov)

        identity = np.identity(nlm, dtype=complex)
        is_identity = np.allclose(cov, identity, rtol=rtol, atol=atol)
        
        max_diff = np.max(cov-identity)

        print('_test_ylm_orthogonality: is_symmetric = {}'.format(is_symmetric))
        assert is_symmetric

        print('_test_ylm_orthogonality: is_identity = {} (rtol={}, atol={})'.format(is_identity, rtol, atol))
        print('_test_ylm_orthogonality: orthogonality test = {}'.format(cov))
        print('_test_ylm_orthogonality: max_diff = {}'.format(max_diff))
        assert is_identity
        

    def _setup_cov_b3d_rsd(self, do_unique_multitracer):

        from lss_theory.covariance import Bispectrum3DRSDCovarianceCalculator

        #TODO buried file path, need to change this
        config_file = './src/lss_theory/sample_inputs/get_covariance_b3d_rsd.yaml'
        with open(config_file) as file:
            info = yaml.load(file, Loader=yaml.FullLoader)
        
        nbin_cos_theta1 = 10
        nbin_phi12 = 10

        do_folded_signal = False#TODO not input from yaml yet

        info_b3d_rsd = info['Bispectrum3DRSDCovariance']['Bispectrum3DRSD']['triangle_orientation_info']
        info_b3d_rsd['min_cos_theta1'] = -1
        info_b3d_rsd['max_cos_theta1'] = 1.0
        info_b3d_rsd['nbin_cos_theta1'] = nbin_cos_theta1
        info_b3d_rsd['min_phi12'] = 0.0
        info_b3d_rsd['max_phi12'] = 2.*np.pi
        info_b3d_rsd['nbin_phi12'] = nbin_phi12
        info_b3d_rsd['do_folded_signal'] = do_folded_signal #TODO not input from yaml

        info['Bispectrum3DRSDCovariance']['Bispectrum3DRSD']['do_unique_multitracer'] = do_unique_multitracer

        info['output_dir'] = './results/b3d_rsd/covariance/cosmo_planck2018_fiducial/nk_11/do_folded_signal_%s/theta_phi_%s_%s/debug/'%(do_folded_signal, nbin_cos_theta1, nbin_phi12)
        info['plot_dir'] = './plots/theory/covariance/b3d_rsd_theta1_phi12_%s_%s/fnl_0/nk_11/'%(nbin_cos_theta1, nbin_phi12)

        b3d_cov_calc = Bispectrum3DRSDCovarianceCalculator(info)
        
        cov_type = info['Bispectrum3DRSDCovariance']['cov_type'] 

        fn_cov = os.path.join(info['output_dir'], 'cov_%s.npy'%cov_type)
        fn_invcov = os.path.join(info['output_dir'], 'invcov_%s.npy'%cov_type)

        return b3d_cov_calc


    def _get_cov_nb_x_nb_block(self, iz, itri, iori, jori):

        (ik1, ik2, ik3) = self._b3d_rsd_spec.triangle_spec.get_ik1_ik2_ik3_for_itri(itri) #TODO make part of b3d_rsd_spec

        nb = self._b3d_rsd_spec.nb

        cov = np.zeros((nb, nb))

        debug_sigp = self._b3d_rsd_spec._debug_sigp

        for ib in range(nb):

            (a, b, c) = self._b3d_rsd_spec.get_isamples(ib)

            for jb in range(nb):

                (d, e, f) = self._b3d_rsd_spec.get_isamples(jb)

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

                cov[ib, jb] = self._get_observed_ps(ips1, iz, ik1, fog=fog1, kaiser=kaiser_1) \
                    * self._get_observed_ps(ips2, iz, ik2, fog=fog2, kaiser=kaiser_2) \
                    * self._get_observed_ps(ips3, iz, ik3, fog=fog3, kaiser=kaiser_3)

                # TODO not accounting for equilateral and isoceles triangles yet 
                # TODO for these other terms in the cov, will need to change fog definition

        return cov

#TODO to delete when new (BispectrumMultipoleCovariance) is tested to work        
class BispectrumMultipoleCovarianceOld(BispectrumMultipoleCovarianceBase):
    """This class takes in the 3D Fourier galaxy bispectrum covariance
    and performs two integrals over the spherical harmonics to
    return the bispectrum multipole covariance"""

    def __init__(self, info):
        super().__init__(info)

        self._cov, self._invcov = self._get_cov_and_invcov()
        self._save()

    def _get_cov_and_invcov(self):

        nbxnlm = self._nb * self._nlm
        nlm = self._nlm
        shape = (nbxnlm, nbxnlm, self._nz, self._ntri)
        #shape = (self._nb, self._nb, self._nz, self._ntri, self._nori)
        
        cov = np.zeros(shape, dtype=complex)
        invcov = np.zeros(shape, dtype=complex)
        
        for iz in range(self._nz):
            for itri in range(self._ntri):

                print('iz = {}, itri = {}'.format(iz, itri))
                
                #HACK no need for reshaping anymore
                #cov_nori_blocks_of_nb_x_nb = self._b3d_rsd_cov[:, :, iz, itri] 
                #cov_nb_blocks_of_nori_x_nori = self._reshape(cov_nori_blocks_of_nb_x_nb)
                #cov_nb_blocks_of_nlm_x_nlm = self._apply_ylm_integral(cov_nb_blocks_of_nori_x_nori)
                #matrix_utils.check_matrix_symmetric(cov_nb_blocks_of_nlm_x_nlm)

                #cov[:, :, iz, itri] = cov_nb_blocks_of_nlm_x_nlm
                #invcov[:, :, iz, itri] = linalg.inv(cov_nb_blocks_of_nlm_x_nlm)

                for ib in range(self._nb):
                    for jb in range(ib, self._nb): 
                        
                        print('ib = {}, jb = {}'.format(ib, jb))
                        array_nori = self._b3d_rsd_cov[ib, jb, iz, itri, :]
                        print('array_nori', array_nori)
                        
                        rescale = 1e-18
                        one_block_nori_x_nori = np.diag(array_nori/rescale)
                        
                        dOmega = (4.*np.pi)/(self._nori)
                        one_block_nlm_x_nlm = self._apply_ylm_integral_on_one_block_nori_x_nori(\
                            one_block_nori_x_nori) * dOmega 

                        print('one_block_nori_x_nori', one_block_nori_x_nori)
                        print('one_block_nlm_x_nlm', one_block_nlm_x_nlm)

                        print('eigen norixnori = ', linalg.eig(one_block_nori_x_nori))
                        print('eigen nlmxnlm = ', linalg.eig(one_block_nlm_x_nlm))

                        
                        cov[ib*(nlm):(ib+1)*nlm, jb*nlm:(jb+1)*nlm, iz, itri] = one_block_nlm_x_nlm
                        cov[jb*(nlm):(jb+1)*nlm, ib*nlm:(ib+1)*nlm, iz, itri] = one_block_nlm_x_nlm  
                        #matrix_utils.check_matrix_symmetric(one_block_nlm_x_nlm)
                        
                        #HACK debugging:
                        inv_one_block = linalg.inv(one_block_nlm_x_nlm)
                        is_passed = matrix_utils.check_matrix_inverse(one_block_nlm_x_nlm, inv_one_block,
                            atol=1e-3, feedback_level=1)
                        assert is_passed
                #matrix_utils.check_matrix_symmetric(cov[:, :, iz, itri])
                
                print('about to get inverse')
                invcov[:, :, iz, itri] = linalg.inv(cov[:, :, iz, itri])

                #matrix_utils.check_matrix_symmetric(invcov[:, :, iz, itri])
                
                matrix_utils.check_matrix_inverse(cov[:,:,iz,itri], \
                    invcov[:,:,iz,itri], atol=1e-6, feedback_level=1)

                sys.exit()

        return cov, invcov

    #TODO not used
    def _reshape(self, cov_nori_blocks_of_nb_x_nb):
        nblock_old = self._nori
        block_size_old = self._nb
        cov_nb_blocks_of_nori_x_nori = \
            matrix_utils.reshape_blocks_of_matrix(cov_nori_blocks_of_nb_x_nb, nblock_old, block_size_old)
        return cov_nb_blocks_of_nori_x_nori
        
    #TODO not used
    def _apply_ylm_integral(self, cov_nb_blocks_of_nori_x_nori):
        nori = self._nori
        nlm = self._nlm
        nb = self._nb
        nbxnlm = nb * nlm
        cov_nb_blocks_of_nori_x_nori_3d = matrix_utils.split_matrix_into_blocks(cov_nb_blocks_of_nori_x_nori, nori, nori)
        cov_nb_blocks_of_nlm_x_nlm_3d = np.zeros((nb**2, nlm, nlm), dtype=complex)
        
        iblock = 0

        for ib in range(self._nb):
            for jb in range(self._nb):

                one_block_nori_x_nori = cov_nb_blocks_of_nori_x_nori_3d[iblock, :, :]
                one_block_nlm_x_nlm = self._apply_ylm_integral_on_one_block_nori_x_nori(\
                    one_block_nori_x_nori)
                
                cov_nb_blocks_of_nlm_x_nlm_3d[iblock,:,:] = one_block_nlm_x_nlm

                iblock = iblock + 1

        cov_nb_blocks_of_nlm_x_nlm = matrix_utils.assemble_matrix_from_blocks(\
            cov_nb_blocks_of_nlm_x_nlm_3d, nb)

        assert cov_nb_blocks_of_nlm_x_nlm.shape == (nbxnlm, nbxnlm)

        return cov_nb_blocks_of_nlm_x_nlm

    #@profiler
    def _apply_ylm_integral_on_one_block_nori_x_nori(self, one_block_nori_x_nori):
        #HACK
        #cov = np.matmul(one_block_nori_x_nori, self._ylms_conj) 
        a = np.identity(one_block_nori_x_nori.shape[0])
        cov = np.matmul(a, np.conj(self._ylms_conj))
        #cov = np.matmul(one_block_nori_x_nori, np.conj(self._ylms_conj))

        cov = np.matmul(self._ylms_conj_transpose, cov)
        #TODO need to add some conversion factor here:
        # 1) dcos theta, dxi
        # 2) (P+1/n)(P+1/n)(P+1/n) vs cov 3D that may have number of modes in there.
        return cov 

    def _apply_nmodes(self):
        pass

    