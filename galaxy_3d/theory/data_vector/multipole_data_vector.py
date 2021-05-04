import numpy as np

from theory.data_vector.data_vector import DataVector
from theory.data_vector.data_spec import Bispectrum3DBaseSpec

class SphericalHarmonicsTable():
    """This class loads the saved spherical harmonics
    from disk or computes from scratch and saves them."""
    def __init__(self):
        pass

class SphericalHarmonicsCalculator():
    """This class precomputes all the spherical harmonics needed 
    given the theta-phi grid, and lmax."""
    def __init__(self):
        pass

    
class BispectrumMultipole(DataVector):
    """This class takes in the 3D Fourier galaxy bispectrum results  
    and performs the integral over the spherical harmonics to
    return the bispectrum multipoles"""

    def __init__(self, grs_ingredients, survey_par, b3d_mult_spec):

        super().__init__(grs_ingredients, survey_par, b3d_mult_spec)

        self.sh_table = self._get_spherical_harmonics()    

    def _setup_allowed_names(self):
        self._allowed_names = ['galaxy_bis']

    def _calc_galaxy_bis(self):
        galaxy_bis = self._get_bis_multipoles()
        self._state['galaxy_bis'] = galaxy_bis
    
    def _get_spherical_harmonics(self):
        """gets the precomputed the spherical harmonics table for the
        right theta-phi grid and lmax."""
        # uses bis_mult_spec for theta phi grid specfications
        # (ntheta, nphi, etc)
        # Calls SphericalHarmonicsTable to get the table as a class variable.
        #TODO to be implemented
        pass

    def _check_format(self):
        """Checks the format of the sh_table and galaxy_bis to make sure 
        that we got everything we need to compute the bispectrum multipoles."""
        #TODO to be implemented
        # pass
    
    def _get_bis_multipoles(self):
        """Computes the bispectrum multipoles"""
        #uses galaxy_bis and self.sh_table to perform integral over dcostheta1 and dphi12.
        #TODO to be implemented
        return np.zeros(1)





   
   