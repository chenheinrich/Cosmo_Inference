import numpy as np
import sys
from theory.utils.logging import class_logger

class AnglesNotInRangeError(Exception):

    def __init__(self, angle, message='.'):
        self.angle = angle
        self.message = 'Error: Input angles %s are not in allowed range: '%self.angle + message
        super().__init__(self.message)


class TriangleSpec():

    """Class managing a list of k1, k2, k3 satisfying triangle inequalities given discretized k list."""

    def __init__(self, k):
        self.logger = class_logger(self)
        self._k = k
        self._nk = k.size
        self._tri_dict_tuple2index, self._tri_index_array, self._tri_array, self._ntri, \
            self._indices_equilateral, self._indices_k2_equal_k3 \
            = self._get_tri_info()
    
    @property
    def k(self):
        return self._k

    @property
    def nk(self):
        return self._nk

    @property
    def ntri(self):
        return self._ntri

    @property
    def tri_dict_tuple2index(self):
        return self._tri_dict_tuple2index

    #TODO make this no longer a property but use access function like get_ik1_ik2_ik3_for_itri()
    @property
    def tri_index_array(self):
        """Returns a 2d numpy array of shape (ntri, 3) for indices [ik1, ik2, ik3]
        that satisfies the triangle inequality.
        e.g. [[0, 0, 0],
              [0, 0, 1],
              ...
              [21, 21, 21]]
        """
        return self._tri_index_array

    def get_ik1_ik2_ik3_for_itri(self, itri):
        """Returns a tuple of size 3 for ik1, ik2, ik3 given triangle index itri"""
        return tuple(self._tri_index_array[itri,:])

    def get_k1_k2_k3_for_itri(self, itri):
        """Returns a tuple of size 3 for k1, k2, k3 given triangle index itri"""
        return tuple(self._tri_array[itri,:])

    #TODO make this no longer a property but use access function like get_k1_k2_k3_for_itri()
    @property
    def tri_array(self):
        """Returns a 2d numpy array of shape (ntri, 3) for k values [k1, k2, k3]
        in that satisfies the triangle inequality (same units as input k to class).
        e.g. [[0.0007, 0.0007, 0.0007],
              [0.0007, 0.0007, 0.01],
              ...
              [0.1, 0.1, 0.1]]
        """
        return self._tri_array

    @property
    def indices_equilateral(self):
        """Indices of equilateral triangles where k1 = k2 = k3."""
        assert np.all(self._indices_equilateral == self.indices_equilateral2)
        return self._indices_equilateral

    @property
    def indices_equilateral2(self):
        (ik1, ik2, ik3) = self.get_ik1_ik2_ik3()
        ind12 = np.where(ik1 == ik2)[0]
        ind23 = np.where(ik2 == ik3)[0]
        indices = [ind for ind in ind12 if ind in ind23] 
        assert np.all(ik1[indices] == ik2[indices])
        assert np.all(ik2[indices] == ik3[indices])
        assert len(indices) == self._nk
        return indices

    @property
    def indices_k2_equal_k3(self):
        """Indices of isoceles triangles where k2 = k3."""
        return self._indices_k2_equal_k3

    def get_ik1_ik2_ik3(self):
        """Returns a tuple of 3 elements, each being a 1d numpy array of length ntri 
        for ik1, ik2, ik3."""
        tri_index_array = self.tri_index_array
        ik1 = tri_index_array[:,0].astype(int) #TODO put this in tri_index_array directly
        ik2 = tri_index_array[:,1].astype(int)
        ik3 = tri_index_array[:,2].astype(int)
        return (ik1, ik2, ik3)

    def get_k1_k2_k3(self):
        """Returns a tuple of 3 elements, each being a 1d numpy array of length ntri 
        for k1, k2, k3."""
        tri_array = self.tri_array
        k1 = tri_array[:,0]
        k2 = tri_array[:,1]
        k3 = tri_array[:,2]
        return (k1, k2, k3)

    def _get_tri_info(self):
        """Find eligible triangles and create the dictionaries and arrays used in this class."""
        itri = 0
        nk = self._nk
        k = self._k

        indices_equilateral = []
        indices_k2_equal_k3 = []
        
        tri_dict_tuple2index = {}
        tri_index_array = np.zeros((nk**3, 3), dtype=int)
        tri_array = np.zeros((nk**3, 3))

        for ik1 in np.arange(nk):

            indices_equilateral.append(itri)

            for ik2 in np.arange(ik1, nk):   
                
                indices_k2_equal_k3.append(itri)
                
                k1 = k[ik1]
                k2 = k[ik2]
                k3_array = k
                ik3_range = self._get_ik3_range_satisfying_triangle_inequality(k1, k2, k3_array)

                for ik3 in ik3_range:

                    k3 = k3_array[ik3]
                    tri_dict_tuple2index['%i, %i, %i'%(ik1, ik2, ik3)] = itri
                    tri_index_array[itri,:] = [ik1, ik2, ik3]
                    tri_array[itri,:] = [k1, k2, k3]
                    itri = itri + 1

        ntri = itri
        assert np.all(tri_index_array[ntri:-1, :] == 0)

        tri_index_array = tri_index_array[:ntri, :]
        tri_array = tri_array[:ntri, :]

        assert len(indices_k2_equal_k3) == self._nk * (self._nk + 1)/2, \
            (len(indices_k2_equal_k3), self._nk * (self._nk + 1)/2)
        
        return tri_dict_tuple2index, tri_index_array, tri_array, ntri, indices_equilateral, indices_k2_equal_k3

    @staticmethod
    def _get_ik3_range_satisfying_triangle_inequality(k1, k2, k3_array):
        ik3_min = np.min(np.where(k3_array >= k2))
        ik3_max = np.max(np.where(k3_array <= (k1 + k2)))
        ik3_array = np.arange(ik3_min, ik3_max+1)
        k3min = np.min(k3_array[ik3_array])
        k3max = np.max(k3_array[ik3_array])
        assert k3max <= (k1+k2)  
        assert k3min >= k2  
        return ik3_array


#TODO change TriangleSpec-->TriangleShapeSpec
class TriangleOrientationSpec(TriangleSpec):

    """
    Class with info for a set of triangles parametrized by (k1, k2, k3, a, b) 
    given 1d numpy arrays for the k values, a and b where a and b are two 
    generic parameters parametrizing the orientation of the triangle.

    This class is must be subclassed, with various functions provided, 
    including how to relate the parametrization to mu1, mu2, mu3 defined as
    
    mu1 = cos(theta1) = los dot k1;
    mu2 = cos(theta2) = los dot k2;
    mu3 = cos(theta3) = los dot k3;
    """

    def __init__(self, k, a, b, set_mu_to_zero=False):

        """
        Args:
            k: A 1d numpy array for the k values.
            a: A 1d numpy array for the values of the 1st orientation parameter. 
            b: A 1d numpy array for the values of the 2nd orientation parameter. 
        """

        super().__init__(k)

        self._set_mu_to_zero = set_mu_to_zero
        self.logger.info('You have set self._set_mu_to_zero to {}'.format(self._set_mu_to_zero)) 
        
        self._range_dict = self._get_range_dict()

        try:
            self._check_input_angle_range(a, b)
        except AnglesNotInRangeError as e:
            self.logger.error(e.message)
            sys.exit() #TODO do real error handling

        self._a = a
        self._b = b

        self._na = self._a.size
        self._nb = self._b.size
        self._nori = self._na * self._nb

        self._da, self._db = self._get_da_db_array()

        self._angle_array = self._get_angle_array()

        self._oriented_triangle_info = self._get_oriented_triangle_info()
        
    @property
    def nori(self):
        """An integer for the number of orientations per triangle shape."""
        return self._nori

    @property
    def na(self):
        """An integer for the number of a bins."""
        return self._na 

    @property
    def nb(self):
        """An integer for the number of b bins."""
        return self._nb
        
    @property
    def a(self):
        """1d numpy array for the input a."""
        return self._a

    @property
    def b(self):
        """1d numpy array for the input b."""
        return self._b

    @property
    def da(self):
        """A float for the da assuming linear spacing in a."""
        return self._da
    
    @property
    def db(self):
        """A float for the db, assuming linear spacing in b."""
        return self._db

    @property
    def a_in_deg(self):
        return self._a / np.pi * 180.0
        
    @property
    def b_in_deg(self):
        return self._b / np.pi * 180.0
    
    @property
    def angle_array(self):
        """2d numpy array where angle_array[iori, :] gives [a, b] in radians for iori-th orientation."""
        return self._angle_array
        
    @property
    def oriented_tri_array(self):
        """3d numpy array where oriented_tri_array[itri, iori, :] gives [k1, k2, k3, a, b] 
        for the itri-th triangle and iori-th orientation."""
        return self._oriented_triangle_info['oriented_triangle']

    @property
    def oriented_tri_index_array(self):
        """3d numpy array where oriented_tri_array[itri, iori, :] gives [ik1, ik2, ik3, ia, ib] 
        for the itri-th triangle and iori-th orientation."""
        return self._oriented_triangle_info['index']

    @property
    def mu_array(self):
        """3d numpy array where mu_array[itri, iori, :] gives [mu1, mu2, mu3] 
        for the itri-th triangle and iori-th orientation."""
        if self._set_mu_to_zero is True:
            return np.zeros_like(self._oriented_triangle_info['mu'])
        else:
            return self._oriented_triangle_info['mu']

    def get_mu1_mu2_m3_for_itri_and_iori(self, itri, iori):
        """Returns a tuple for (mu1, mu2, mu3) given itri and iori"""
        return tuple(self.mu_array[itri, iori, :])

    def _get_range_dict(self):
        raise NotImplementedError

    def _check_input_angle_range(self, a, b):

        min_a = self._range_dict['min_a']
        max_a = self._range_dict['max_a']
        min_b = self._range_dict['min_b']
        max_b = self._range_dict['max_b']
        
        if np.any(a > max_a):
            raise AnglesNotInRangeError(a, message='found a larger than {}'.format(max_a))
        elif np.any(a < min_a):
            raise AnglesNotInRangeError(a, message='found a smaller than {}'.format(min_a))
        elif np.any(b > max_b):
            raise AnglesNotInRangeError(b, message='found b larger than {}'.format(max_b))
        elif np.any(b < min_b):
            raise AnglesNotInRangeError(b, message='found b smaller than {}'.format(min_b))

    def _get_da_db_array(self):
        raise NotImplementedError

    def _get_angle_array(self):

        angle_array = np.zeros((self._nori, 2), dtype=float)

        iori = 0
        for ia, a in enumerate(self._a):
            for ib, b in enumerate(self._b):
                angle_array[iori, :] = np.array([a, b])
                iori = iori + 1
        return angle_array

    
    def _get_oriented_triangle_info(self):
        
        oriented_tri_mu_array = np.zeros((self._ntri, self._nori, 3))
        oriented_tri_index_array = np.zeros((self._ntri, self._nori, 5), dtype=int)
        oriented_tri_array = np.zeros((self._ntri, self._nori, 5))

        for itri in np.arange(self.ntri):

            ik1 = self.tri_index_array[itri][0]
            ik2 = self.tri_index_array[itri][1]
            ik3 = self.tri_index_array[itri][2]
            
            iori = 0
            for ia, a in enumerate(self._a):
                for ib, b in enumerate(self._b):
                    
                    k1 = self._tri_array[itri, 0]
                    k2 = self._tri_array[itri, 1]
                    k3 = self._tri_array[itri, 2]

                    (mu1, mu2, mu3) = self._get_mu1_mu2_mu3(k1, k2, k3, a, b)

                    oriented_tri_mu_array[itri, iori, :] = np.array([mu1, mu2, mu3])
                    oriented_tri_index_array[itri, iori, :] = np.array([ik1, ik2, ik3, ia, ib])
                    oriented_tri_array[itri, iori, :] = np.array([k1, k2, k3, a, b])

                    iori = iori + 1

        oriented_triangle_info = {}
        oriented_triangle_info['mu'] = oriented_tri_mu_array
        oriented_triangle_info['index'] = oriented_tri_index_array
        oriented_triangle_info['oriented_triangle'] = oriented_tri_array

        return oriented_triangle_info

    def _get_mu1_mu2_mu3(self, k1, k2, k3, a, b):
        raise NotImplementedError

class TriangleOrientationSpec_Theta1Phi12(TriangleOrientationSpec):
    """
    Class managing a list of triangles parametrized by 
    (k1, k2, k3, theta1, phi12) given a discretized array of k, 
    theta1 and phi12.

    We follow Scoccimarro 2015 page 4 for the definition of 
    theta1, theta12 and phi12: 
    
    theta1: angle between line-of-sight (los) and k1 vector
    theta12: angle between k1 and k2 vectors
    phi12: azimuthal angle of k2 vector in a frame where k1 is z direction.
        (More specifically:
        z' // k1, 
        x' in the same plane as los and k1, and k1 cross x' in 
            the same direction as los x k1. 
        y' \perp z, y \perp x, such that x, y, z 
        form a right-handed coordinates.)

    We relate theta1 and phi12 to mu1, mu2 which are defined as

    mu1 = cos(theta1) := los dot k1;
    mu2 = cos(theta2) := los dot k2;

    Note: Note that although the input variable is theta1 and phi12, 
    we expect them to be linearly spaced in cos(theta1) and phi12.
    This is assumed when calculating dcostheta1 and dphi12.
    """

    def __init__(self, k, theta1, phi12, set_mu_to_zero=False):

        """
        Args:
            k: A 1d numpy array of k values.
            theta1: A 1d numpy array of theta1 values.
            phi12: A 1d numpy array of phi12 values.
            
        Note:
            It is actually cos(theta1) that is being linearly sampled here.
        """

        super().__init__(k, theta1, phi12, set_mu_to_zero=set_mu_to_zero)

    def _get_range_dict(self):
        range_dict = {
            'min_a': 0.0,
            'max_a': np.pi,
            'min_b': 0.0,
            'max_b': 2*np.pi,
        }
        return range_dict

    def _get_da_db_array(self):
        da = np.cos(self._a[1]) - np.cos(self._a[0])
        db= self._b[1] - self._b[0]
        return da, db

    def _get_mu1(self, k1, k2, k3, theta1, phi12):
        return np.cos(theta1)

    def _get_mu2(self, k1, k2, k3, theta1, phi12):
        theta12 = self.get_theta12(k1, k2, k3)
        mu2 = np.cos(theta1) * np.cos(theta12) - np.sin(theta1) * np.sin(theta12) * np.cos(phi12)
        return mu2

    def _get_mu3(self, k1, k2, k3, theta1, phi12):
        theta12 = self.get_theta12(k1, k2, k3)
        mu3 = k2 * np.sin(theta12) * np.cos(phi12) * np.sin(theta1) - (k1 + k2*np.cos(theta12)) * np.cos(theta1)
        mu3 = mu3/k3
        return mu3

    #TODO switch calls of get_mu1, get_mu2, get_mu3 to this function 
    def _get_mu1_mu2_mu3(self, k1, k2, k3, theta1, phi12):
        mu1 = np.cos(theta1)

        theta12 = self.get_theta12(k1, k2, k3)
        mu2 = np.cos(theta1) * np.cos(theta12) - np.sin(theta1) * np.sin(theta12) * np.cos(phi12)
        
        mu3 = k2 * np.sin(theta12) * np.cos(phi12) * np.sin(theta1) - (k1 + k2*np.cos(theta12)) * np.cos(theta1)
        mu3 = mu3/k3

        return (mu1, mu2, mu3)

    @staticmethod
    def get_theta12(k1, k2, k3):
        """arccos always returns angle between [0, pi], 
        so theta12 is the same for two triangles of opposite handedness."""
        theta12 = np.arccos(0.5 * (-k1*k1 - k2*k2 + k3*k3) / (k1 * k2))
        return theta12
          
class TriangleOrientationSpec_MurMuphi(TriangleOrientationSpec):
    """
    Class with info for a set of triangles parametrized by (k1, k2, k3, mu_r, mu_phi) 
    given a discretized k list, where mu_r and mu_phi are the radial and angular
    coordinates of the mu1-mu2 plane,
    
    mu_r = sqrt(mu1^2 + mu2^2);
    mu_phi = arctan(mu1/mu2);

    or 

    mu1 = mu_r * cos(mu_phi)
    mu2 = mu_r * sin(mu_phi)
    
    where mu1 and mu2 are defined as
    mu1 := cos(theta1) = los dot k1;
    mu2 := cos(theta2) = los dot k2;

    They are related to theta1 and phi12 (the azimuthal angle in the frame formed by
        z' // k1, 
        x' in the same plane as los and k1, and k1 cross x' in the same direction as los x k1. 
        y' \perp z, y \perp x, such that x, y, z form a right-handed coordinates.)
        
    Note: We provide the transformation rules to (theta1, phi12)
    as well as the Jacobian wrt dtheta1 and dphi12. 
    """

    def __init__(self, k, mu_r, mu_phi, set_mu_to_zero=False):
    
        """
        Args:
            k: A 1d numpy array for k values.
            a: A 1d numpy array for mu_r values.
            b: A 1d numpy array for mu_phi values.
        """

        super().__init__(k, mu_r, mu_phi, set_mu_to_zero=set_mu_to_zero)

        self._Sigma_mu1_mu2 = self._get_Sigma_mu1_mu2()
        self._dOmega = self._get_dOmega()

    @property
    def Sigma_mu1_mu2(self):
        """
        A 2d numpy array for Sigma(mu1, mu2) such that Sigma[itri, iori] returns 
        Sigma(mu1, mu2) which is defined as:
            Sigma(mu1, mu2) dmu1 dmu2 = Sigma(mu1, mu2) mur dmur dmuphi 
            is the fraction of triangles with fixed shape in a dmu1-dmu2 bin, 
        and so integrating Sigma(mu1, mu2) dmu1 dmu2 over the whole sphere = 4pi.
        """
        return self._Sigma_mu1_mu2

    @property
    def dOmega(self): # TODO self._triangle_spec vs self.triangle_spec
        """
        Returns 2d numpy array such that dOmega[itri, iori] gives the 
        solid angle element dOmega = mur * dmur * dmuphi = dmu1 dmu2.
        """
        return self._dOmega

    def _get_range_dict(self):

        range_dict = {
            'min_a': 0.0,
            'max_a': np.sqrt(2),
            'min_b': 0.0,
            'max_b': 2*np.pi,
        }
        
        return range_dict

    def _get_da_db_array(self):
        da = np.cos(self._a[1]) - np.cos(self._a[0])
        db= self._b[1] - self._b[0]
        return da, db

    def _get_mu1_mu2_mu3(self, k1, k2, k3, mu_r, mu_phi):
        mu1 = mu_r * np.cos(mu_phi)
        mu2 = mu_r * np.sin(mu_phi)

        mu3 = -(k1*mu1 + k2*mu2)/k3
        return (mu1, mu2, mu3)

    def _get_theta1_phi12_from_mu_r_mu_phi(self, k1, k2, k3, mu_r, mu_phi):

        (mu1, mu2, mu3) = self._get_mu1_mu2_mu3(k1, k2, k3, mu_r, mu_phi)
        cos_theta1 = mu1

        theta1 = np.arccos(mu1)
        
        cos_theta12 = np.cos(self.get_theta12(k1, k2, k3))

        sin_theta1 = np.sqrt(1.0 - cos_theta1**2)
        sin_theta12 = np.sqrt(1.0 - cos_theta12**2)

        phi12 = np.arccos(cos_theta1 * cos_theta12 - mu2)/ (sin_theta1 * sin_theta12)

        return (theta1, phi12)

    def _get_mu1_mu2_mu3_from_theta1_phi12(self, k1, k2, k3, theta1, phi12):
        mu1 = np.cos(theta1)

        theta12 = self.get_theta12(k1, k2, k3)
        mu2 = np.cos(theta1) * np.cos(theta12) - np.sin(theta1) * np.sin(theta12) * np.cos(phi12)
        
        mu3 = k2 * np.sin(theta12) * np.cos(phi12) * np.sin(theta1) - (k1 + k2*np.cos(theta12)) * np.cos(theta1)
        mu3 = mu3/k3

        return (mu1, mu2, mu3)

    def _get_Sigma_mu1_mu2(self): 

        """
        Sigma(mu1, mu2) dmu1 dmu2 = Fraction of triangles with fixed shape in a 
        dmu1-dmu2 bin: Integrating Sigma(mu1, mu2) dmu1 dmu2 from -1 to 1 = 4pi.
        """

        Sigma = np.zeros((self.ntri, self.nori))
        
        for itri in range(self.ntri):

            (k1, k2, k3) = self.get_k1_k2_k3_for_itri(itri)
            theta12 = self.get_theta12(k1, k2, k3)
                
            for iori in range(self.nori):
                
                (mu1, mu2, mu3) = self.get_mu1_mu2_m3_for_itri_and_iori(itri, iori)
                arg = (np.sin(theta12))**2 - mu1**2 - mu2**2 + 2*np.cos(theta12)*mu1*mu2
                print('arg = {}'.format(arg))
                Sigma[itri, iori] = 1. / (2*np.pi * np.sqrt(arg))

        return Sigma

    def _get_dOmega(self): 
        """
        Returns 2d numpy array such that dOmega[itri, iori] gives the 
        solid angle element dOmega = mur * dmur * dmuphi = dmu1 dmu2.
        """
        dmur = self.da # float
        dmuphi = self.db # float

        dOmega = np.zeros((self.ntri, self.nori))
        
        for itri in range(self.ntri):
            for iori in range(self.nori):
                mur = self.angle_array[iori, :][0]
                dOmega[itri, iori] = mur * dmur * dmuphi
        #TODO use broadcast to eliminate itri for loop
        return dOmega

    @staticmethod
    def get_theta12(k1, k2, k3):
        """arccos always returns angle between [0, pi], 
        so theta12 is the same for two triangles of opposite handedness."""
        theta12 = np.arccos(0.5 * (-k1*k1 - k2*k2 + k3*k3) / (k1 * k2))
        return theta12
