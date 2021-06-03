import numpy as np
import os
import json
from lss_theory.utils.file_tools import mkdir_p

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


def test_ylms(theta, phi):  
    """Returns 1d numpy array of 4 elements for Ylm values with
    (l, m) = [(0,0), (1,-1), (1,0), (1,1)] given polar angle theta 
    and azimuthal angle phi. Used for testing.
    """
    y00 = 0.5/np.sqrt(np.pi)
    y1m1 = 0.5*np.sqrt(1.5/np.pi) * np.sin(theta) * np.exp(-phi * 1j)
    y1m0 = 0.5*np.sqrt(3.0/np.pi) * np.cos(theta)
    y11 = -0.5*np.sqrt(1.5/np.pi) * np.sin(theta) * np.exp(phi * 1j)
    return np.array([y00, y1m1, y1m0, y11])

class SphericalHarmonicsTable():
    """This class gets the saved spherical harmonics table
    from disk or computes from scratch and saves them.

    Attributes:
        data: 2d numpy array of shape (nori, nlms) for the 
            spherical harmonics table where nori = ntheta * nphi, 
            and nlms = sum of (2l+1) for l = 0 to lmax included.
    """
    def __init__(self, thetas, phis, lmax):
        """
        Args:
            thetas: 1d numpy array for polar angle values.
            phis: 1d numpy array for azimuthal angle values.
            lmax: An integer for the max l value (itself included).
        """
        self._thetas = thetas
        self._phis = phis
        self._lmax = lmax

        self._cache_dir = './tmp/galaxy_3d/sh_table/'
        mkdir_p(self._cache_dir)

        try:
            self._sh_table = self._get_cache()
            message = 'Cache found successfully for ' + \
                'lmax = {} in {}\n'.format(\
                    self._lmax, self._cache_path) + \
                '(see metadata in file {}).'.format(self._cache_metadata_path)
            print(message)
        except CacheNotFoundError as e:
            print('SphericalHarmonicsTable: {} '.format(e.message)
             + '\n ... computing from scratch')
            self._sh_table = self._get_sh()
            self._save_cache(self._sh_table)

    @property
    def data(self):
        return self._sh_table

    @property
    def cache_dir(self):
        return self._cache_dir

    @property
    def thetas(self):
        return self._thetas

    @property
    def phis(self):
        return self._phis

    @property
    def lmax(self):
        return self._lmax

    def _save_cache(self, sh_table):
        subdir = os.path.join(self._cache_dir, '%s/'%(self._get_time_stamp()))
        mkdir_p(subdir)
        
        self._cache_path = os.path.join(subdir, 'sh_table.npy')
        self._cache_metadata_path = os.path.join(subdir, 'metadata.json')
        
        np.save(self._cache_path, sh_table)
        
        cache_metadata = {
            'lmax': self._lmax, 
            'thetas': list(self._thetas), 
            'phis': list(self._phis), 
            }
        with open(self._cache_metadata_path, 'w') as outfile:
            json.dump(cache_metadata, outfile, sort_keys=True, indent=4)
        
        print('Saved cache for spherical harmonics table ' + 
            '(lmax = {}) in directory {} (see metadata at {})'.format(
            self._lmax, subdir, self._cache_metadata_path))

    @staticmethod
    def _get_time_stamp():
        import time
        named_tuple = time.localtime() 
        time_string = time.strftime("%Y%m%d_%H%M%S", named_tuple)
        return time_string

    def _get_cache(self):
        """Returns spherical harmonics table if cache is found, 
        otherwise returns None."""
        self._cache_path, self._cache_metadata_path = self._find_cache()
        sh_table = self._load_cache(self._cache_path)
        return sh_table

    def _find_cache(self):
        """Returns data .npy file and metadata .json file paths 
        if cache is found, otherwise raises error CacheNotFound."""
    
        subdir_list = os.listdir(self._cache_dir)
        
        for subdir in subdir_list:
            cache_metadata_path = os.path.join(self._cache_dir, subdir, 'metadata.json')
            cache_path = os.path.join(self._cache_dir, subdir, 'sh_table.npy')
            
            with open(cache_metadata_path) as json_file:
                metadata = json.load(json_file)
            
            passed = []
            passed.append(metadata['thetas'] == list(self._thetas))
            passed.append(metadata['phis'] == list(self._phis))
            passed.append(metadata['lmax'] == self._lmax)

            if all(passed):
                return cache_path, cache_metadata_path
            else:
                pass

        message = 'Cache not found for lmax = %s in cache_dir = %s.'%(self._lmax, self._cache_dir)
        raise CacheNotFoundError(self._cache_dir, message)

    def _load_cache(self, cache_path):
        sh_table = np.load(cache_path)

        itheta = iphi = 0
        iori = 0
        theta = self._thetas[itheta] 
        phi = self._phis[iphi]
        check_passed = np.allclose(test_ylms(theta, phi), sh_table[iori, :4])
        assert check_passed
        print('Loading ylms: Check passed? %s'%check_passed)

        return sh_table

    def _get_sh(self):
        calc = SphericalHarmonicsCalculator(self._thetas, self._phis, self._lmax)
        return calc.get_ylms()

class SphericalHarmonicsCalculator():
    """This class precomputes all the spherical harmonics 
    needed, given thetas, phis and lmax.
    
    Attributes:
        data: returns 3d numpy array of shape (ntheta, nphi, nlms), where
            nlms is sum (2l+1) from l = 0 to l = lmax."""

    def __init__(self, thetas, phis, lmax):
        """
        Args:
            thetas: 1d numpy array for polar angle theta values.
            phis: 1d numpy array for azimuthal angle phi values.
            lmax: max of l values (0 and lmax included are calculated with -l < m < l.)
        """

        self._thetas = thetas
        self._phis = phis
        self._lmax = lmax
        self._ntheta = self._thetas.size
        self._nphi = self._phis.size
        self._nlm = np.sum([2*l+1 for l in range(0, lmax+1)])

    def get_ylms(self):
        from scipy.special import sph_harm 
        # use as sph_harm(m, l, phi, theta)
        # theta phi reverted, fully normalized convention
        
        nori = self._ntheta * self._nphi
        ylms = np.zeros((nori, self._nlm), dtype=complex)
        lmax = self._lmax

        ilm = 0
        for l in range(0, lmax+1):
            for m in range(-l, l+1):

                iori = 0
                for itheta, theta in enumerate(self._thetas):
                    for iphi, phi in enumerate(self._phis):
                        
                        ylms_tmp = sph_harm(m, l, phi, theta)
                        ylms[iori, ilm] = ylms_tmp
                        iori = iori + 1

                ilm = ilm + 1

        print('ylms.shape = ', ylms.shape)
        print('expected shape = ', (nori, self._nlm))

        return ylms