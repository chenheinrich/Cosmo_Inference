import numpy as np
import os
import copy
import scipy.linalg as linalg

from lss_theory.data_vector.multipole_data_spec import BispectrumMultipoleSpec
from lss_theory.math_utils.spherical_harmonics import SphericalHarmonicsTable
import lss_theory.math_utils.matrix as matrix
from lss_theory.scripts.get_bis_mult import get_subdir_name_for_bis_mult
from lss_theory.utils.file_tools import mkdir_p
from lss_theory.params.survey_par import SurveyPar
from lss_theory.utils import profiler

class BispectrumMultipoleCovariance():
    """This class takes in the 3D Fourier galaxy bispectrum covariance
    and performs two integrals over the spherical harmonics to
    return the bispectrum multipole covariance"""

    def __init__(self, info):
        """Args:
        """
        self._info = copy.deepcopy(info)
        self._setup_dir()
        self._setup_paths()
        self._load_b3d_rsd_cov()
        
        survey_par = SurveyPar(self._info['survey_par_file'])

        self._bis_mult_spec = BispectrumMultipoleSpec(survey_par, self._info['BispectrumMultipole'])
        self._setup_dims()

        self._do_negative_m = self._bis_mult_spec._do_negative_m
        self._ylms_conj = self._get_ylms_conj()
        self._ylms_conj_transpose = np.transpose(self._ylms_conj)

        self._cov, self._invcov = self._get_cov_and_invcov()
        self._save()

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
        theta1 = self._bis_mult_spec.b3d_rsd_spec.theta1
        phi12 = self._bis_mult_spec.b3d_rsd_spec.phi12
        lmax = self._bis_mult_spec.lmax

        spherical_harmonics_table = SphericalHarmonicsTable(theta1, phi12, lmax, \
            self._do_negative_m)
        ylms = spherical_harmonics_table.data

        return np.conj(ylms)

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
                #matrix.check_matrix_symmetric(cov_nb_blocks_of_nlm_x_nlm)

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
                        #matrix.check_matrix_symmetric(one_block_nlm_x_nlm)
                        
                        inv_one_block = linalg.inv(one_block_nlm_x_nlm)
                        is_passed = matrix.check_matrix_inverse(one_block_nlm_x_nlm, inv_one_block,
                            atol=1e-3, feedback_level=1)
                        assert is_passed
                #matrix.check_matrix_symmetric(cov[:, :, iz, itri])
                
                print('about to get inverse')
                invcov[:, :, iz, itri] = linalg.inv(cov[:, :, iz, itri])

                #matrix.check_matrix_symmetric(invcov[:, :, iz, itri])
                
                matrix.check_matrix_inverse(cov[:,:,iz,itri], \
                    invcov[:,:,iz,itri], atol=1e-6, feedback_level=1)

                sys.exit()

        return cov, invcov

    #TODO not used
    def _reshape(self, cov_nori_blocks_of_nb_x_nb):
        nblock_old = self._nori
        block_size_old = self._nb
        cov_nb_blocks_of_nori_x_nori = \
            matrix.reshape_blocks_of_matrix(cov_nori_blocks_of_nb_x_nb, nblock_old, block_size_old)
        return cov_nb_blocks_of_nori_x_nori
        
    #TODO not used
    def _apply_ylm_integral(self, cov_nb_blocks_of_nori_x_nori):
        nori = self._nori
        nlm = self._nlm
        nb = self._nb
        nbxnlm = nb * nlm
        cov_nb_blocks_of_nori_x_nori_3d = matrix.split_matrix_into_blocks(cov_nb_blocks_of_nori_x_nori, nori, nori)
        cov_nb_blocks_of_nlm_x_nlm_3d = np.zeros((nb**2, nlm, nlm), dtype=complex)
        
        iblock = 0

        for ib in range(self._nb):
            for jb in range(self._nb):

                one_block_nori_x_nori = cov_nb_blocks_of_nori_x_nori_3d[iblock, :, :]
                one_block_nlm_x_nlm = self._apply_ylm_integral_on_one_block_nori_x_nori(\
                    one_block_nori_x_nori)
                
                cov_nb_blocks_of_nlm_x_nlm_3d[iblock,:,:] = one_block_nlm_x_nlm

                iblock = iblock + 1

        cov_nb_blocks_of_nlm_x_nlm = matrix.assemble_matrix_from_blocks(\
            cov_nb_blocks_of_nlm_x_nlm_3d, nb)

        assert cov_nb_blocks_of_nlm_x_nlm.shape == (nbxnlm, nbxnlm)

        return cov_nb_blocks_of_nlm_x_nlm

    #@profiler
    def _apply_ylm_integral_on_one_block_nori_x_nori(self, one_block_nori_x_nori):
        cov = np.matmul(one_block_nori_x_nori, self._ylms_conj) 
        cov = np.matmul(self._ylms_conj_transpose, cov)
        return cov 

    def _apply_nmodes(self):
        pass

    def _save(self):
        np.save(self._bis_mult_cov_path, self._cov)
        np.save(self._bis_mult_invcov_path, self._invcov)
        print('Saved cov at {}'.format(self._bis_mult_cov_path))
        print('Saved invcov at {}'.format(self._bis_mult_invcov_path))

        