import numpy as np
import os

from theory.data_vector.data_vector import DataVector, Bispectrum3DRSD
from theory.data_vector.data_spec import Bispectrum3DRSDSpec
from theory.utils.file_tools import mkdir_p

# Options here:
# Store this as a table in a class that's passed around during initialization
# So the bispectrum class doesn't know about it, except that it can be an 
# optional argument if pre-calculated, otherwise it is able to calculate and 
# call it itself if not given.
# So want a class that holds the table and also able to load
# And a second class that just computes it.

class CacheNotFoundError(Exception):
    """Raised when spherical harmonics cache not found.
    Attributes:
        cache_dir: path to cache directory 
        message: explanation of the error
    """

    def __init__(self, cache_dir, message="Cache not found."):
        self.cache_dir = cache_dir
        self.message = message
        super().__init__(self.message)


class SphericalHarmonicsTable():
    """This class loads the saved spherical harmonics
    from disk or computes from scratch and saves them."""
    def __init__(self, theta, phi, lmax):
        self._theta = theta
        self._phi = phi
        self._lmax = lmax
        try:
            self._sh_table = self._load_cache(theta, phi, lmax)
        except CacheNotFoundError as e:
            print('SphericalHarmonicsTable: {} '.format(e.message)
             + '\n ... computing from scratch')
            self._sh_table = self._get_sh(theta, phi, lmax)

    @property
    def data(self):
        return self._data

    @property
    def theta(self):
        return self._theta

    @property
    def phi(self):
        return self._phi

    @property
    def lmax(self):
        return self._lmax

    def _load_cache(self, theta, phi, lmax):
        """Returns spherical harmonics table if cache is found, 
        otherwise returns None."""
        cache_info = self._find_cache(theta, phi, lmax)
        sh_table = self._get_cache(*cache_info)
        return sh_table

    def _find_cache(self, theta, phi, lmax):
        """Returns .npy file path if cache is found, 
        otherwise returns None"""
        cache_dir = './tmp/galaxy_3d/sh_table/'
        mkdir_p(cache_dir)
        # list subdir 
        subdir_list = os.listdir(cache_dir)
        
        path_dict = {}
        #for subdir in subdir_list:
            #load json file
            # compare dictionary for theta and phi
            # if not the same, pass
            # if the same, record lmax and path in a path_dict
        # select entry lmax in path_dict bigger than and closest to desired lmax
        # return lmax and path
       
        #if cache_lmax is not None: 
        #    cache_path = path_dict['cache_lmax']
        #    return cache_lmax, cache_path
        #else:
        message = 'Cache not found for lmax = %s in cache_dir = %s.'%(lmax, cache_dir)
        raise CacheNotFoundError(cache_dir, message)

    def _get_cache(self, cache_path, cache_lmax, desired_lmax):
        sh_table = np.load(cache_path)
        #TODO to implement
        #if cache_lmax > desired_lmax:
            #trim
            #sh_table = ...
        return sh_table

    def _get_sh(self, theta, phi, lmax):
        calc = SphericalHarmonicsCalculator(theta, phi, lmax)
        return calc.get_ylms()

class SphericalHarmonicsCalculator():
    """This class precomputes all the spherical harmonics needed 
    given the theta-phi grid, and lmax."""
    def __init__(self, theta, phi, lmax):
        self._theta = theta
        self._phi = phi
        self._lmax = lmax
        self._ntheta = self._theta.size
        self._nphi = self._phi.size
        self._nlm = np.sum([2*l+1 for l in range(0, lmax+1)])
    
    def get_ylms(self):
        from julia.api import Julia
        jl = Julia(compiled_modules=False)
        from julia import Main
        jl.using("SphericalHarmonics")
        print('Done loading julia')
        #TODO to implement
        #call julia? or anything to get spherical harmonics
        nori = self._ntheta * self._nphi
        ylms = np.zeros((nori, self._nlm), dtype=complex)
        
        iori = 0
        lmax = self._lmax
        for itheta, theta in enumerate(self._theta):
            for iphi, phi in enumerate(self._phi):
                ylms_tmp = Main.eval("computeYlm(%s, %s, %s)"%(theta, phi, lmax))
                ylms[iori, :] = ylms_tmp
                iori = iori + 1
        ylms = np.array(ylms)
        print('ylms.shape = ', ylms.shape)
        print('expected shape = ', (nori, self._nlm))
        return ylms

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
        self._sh_table = self._get_spherical_harmonics_table(spherical_harmonics_table)    

    def _setup_allowed_names(self):
        self._allowed_names = ['galaxy_bis']

    def _get_b3d_rsd(self, grs_ingredients, survey_par, bis_mult_spec):
        b3d_rsd_spec = self._data_spec.b3d_rsd_spec #bis_mult_spec.b3d_rsd_spec
        b3d_rsd = Bispectrum3DRSD(self._grs_ingredients, self._survey_par, b3d_rsd_spec)
        return b3d_rsd

    def _calc_galaxy_bis(self):
        galaxy_bis = self._get_bis_multipoles()
        self._state['galaxy_bis'] = galaxy_bis
    
    def _get_spherical_harmonics_table(self, spherical_harmonics_table):
        if spherical_harmonics_table is None:
            theta1 = self._data_spec.b3d_rsd_spec.theta1
            phi12 = self._data_spec.b3d_rsd_spec.phi12
            lmax = self._data_spec.lmax
            spherical_harmonics_table = SphericalHarmonicsTable(theta1, phi12, lmax)
        return spherical_harmonics_table

    def _get_spherical_harmonics(self):
        """gets the precomputed the spherical harmonics table for the
        right theta-phi grid and lmax."""
        # uses bis_mult_spec for theta phi grid specfications
        # (ntheta, nphi, etc)
        theta1 = self._data_spec.b3d_rsd_spec.theta1
        phi12 = self._data_spec.b3d_rsd_spec.phi12
        nori = self._data_spec.b3d_rsd_spec.nori
        nlm = self._data_spec.nlm
        angle_array = self._data_spec.b3d_rsd_spec.angle_array
        ylm_table = np.zeros((nori, nlm))
        for iori in range(nori):
            theta, phi = tuple(angle_array[iori, :])
            #TODO
            #ylm_table[iori, :] = call Ylm(theta, phi, l, m) 
            #TODO document normalization
        return ylm_table
    
    def _get_bis_multipoles(self):
        """Computes the bispectrum multipoles by integrating over dcostheta1 and dphi12.
            Blm(k1, k2, k3) = Int_{-1}^{1} dcos(theta1) Int_{0}^{2\pi} dphi12
                                x B(k1, k2, k3, theta1, phi12) * Ylm^*(theta1, phi12).
        """
        galaxy_bis = self._b3d_rsd.get('galaxy_bis')
        ylm_table = self._get_spherical_harmonics() #TODO get complex conjugate
        Sigma = self._b3d_rsd._data_spec.Sigma_to_use

        print(galaxy_bis.shape)
        print(ylm_table.shape)
        #TODO do a test with the integration where B(k1, k2, k3, theta1, phi12)
        # is another ylm, to test orthogonality relationship is satisfied
        # for this integral (same lm gives 1, not same lm gives 0)

        galaxy_bis_mult = Sigma * np.matmul(galaxy_bis, ylm_table) 
        assert galaxy_bis_mult.shape[0:3] == galaxy_bis.shape[0:3]
        assert galaxy_bis_mult.shape[-1] == self._data_spec.nlm

        return galaxy_bis_mult





   
   