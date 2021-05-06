import numpy as np
import os

from theory.data_vector.data_vector import DataVector, Bispectrum3DRSD
from theory.data_vector.data_spec import Bispectrum3DRSDSpec
from theory.math_utils.spherical_harmonics import SphericalHarmonicsTable

class BispectrumMultipole(DataVector):
    """This class takes in the 3D Fourier galaxy bispectrum results  
    and performs the integral over the spherical harmonics to
    return the bispectrum multipoles"""

    def __init__(self, grs_ingredients, survey_par, bis_mult_spec, \
            spherical_harmonics_table=None):
        """Args:
            grs_ingredients: An instance of the GRSIngredients class.
            survey_par: An instance of the SurveyPar class
            bis_mult_spec: An instance of the BispectrumMultipoleSpec class.
            spherical_harmonics_table (optional): An instance of the
                SphericalHarmonicsTable class.
        """

        super().__init__(grs_ingredients, survey_par, bis_mult_spec)

        self._b3d_rsd = self._get_b3d_rsd(grs_ingredients, survey_par, bis_mult_spec)
        self._ylms = self._get_ylms(spherical_harmonics_table)    

    def _setup_allowed_names(self):
        self._allowed_names = ['galaxy_bis']

    def _get_b3d_rsd(self, grs_ingredients, survey_par, bis_mult_spec):
        b3d_rsd_spec = self._data_spec.b3d_rsd_spec #bis_mult_spec.b3d_rsd_spec
        b3d_rsd = Bispectrum3DRSD(self._grs_ingredients, self._survey_par, b3d_rsd_spec)
        return b3d_rsd

    def _calc_galaxy_bis(self):
        galaxy_bis = self._get_bis_multipoles()
        self._state['galaxy_bis'] = galaxy_bis
    
    def _get_ylms(self, spherical_harmonics_table):
        """Returns the 2d numpy array of shape (nori, nlms) for the 
        precomputed spherical harmonics on the theta-phi grid and 
        lmax specified in data_spec.
        Args:
            spherical_harmonics_table: an instance of the SphericalHarmonicsTable 
                or None to create an instance here.
        """

        if spherical_harmonics_table is None:
            theta1 = self._data_spec.b3d_rsd_spec.theta1
            phi12 = self._data_spec.b3d_rsd_spec.phi12
            lmax = self._data_spec.lmax
            spherical_harmonics_table = SphericalHarmonicsTable(theta1, phi12, lmax)
        
        ylms = spherical_harmonics_table.data
        nori = self._data_spec.b3d_rsd_spec.nori
        nlm = self._data_spec.nlm
        assert ylms.shape == (nori, nlm)

        return ylms

    def _get_bis_multipoles(self):
        """Computes the bispectrum multipoles by integrating over dcostheta1 and dphi12.
            Blm(k1, k2, k3) = Int_{-1}^{1} dcos(theta1) Int_{0}^{2\pi} dphi12
                                x B(k1, k2, k3, theta1, phi12) * Ylm^*(theta1, phi12).
        """
        galaxy_bis = self._b3d_rsd.get('galaxy_bis')
        dOmega = self._b3d_rsd._data_spec.dOmega

        print('dOmega = {}'.format(dOmega))

        print(galaxy_bis.shape)
        print(self._ylms.shape)
        print(np.transpose(self._ylms).shape)
        
        #TEST: test orthogonality relationship of ylm for the given sampling rate
        # for this integral (same lm gives 1, not same lm gives 0)
        one = dOmega * np.matmul(np.transpose(self._ylms), np.conj(self._ylms))
        print('expect identity matrix: ', one)
        passed_test = np.allclose(np.diag(np.ones(9)), one)
        print('Orthogonality of ylms for given sampling in (theta, phi): test passed?', passed_test)
        # TODO need to test for convergence of final integral for bispectrum
        # might want to just interpolate bispectrum if it varies slower than ylms
        # and accuracy matters there.

        galaxy_bis_mult = dOmega * np.matmul(galaxy_bis, np.conj(self._ylms)) 
        assert galaxy_bis_mult.shape[0:3] == galaxy_bis.shape[0:3]
        assert galaxy_bis_mult.shape[-1] == self._data_spec.nlm

        return galaxy_bis_mult





   
   