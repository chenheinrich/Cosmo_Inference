import numpy as np
import sys

class AnglesNotInRangeError(Exception):

    def __init__(self, angle, message='.'):
        self.angle = angle
        self.message = 'Error: Input angles %s are not in allowed range: '%self.angle + message
        super().__init__(self.message)


class TriangleSpec():

    """Class managing a list of k1, k2, k3 satisfying triangle inequalities given discretized k list."""

    def __init__(self, k):
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
        ik1 = tri_index_array[:,0].astype(int)
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


class TriangleSpecTheta1Phi12(TriangleSpec):

    """Class managing a list of triangles parametrized by (k1, k2, k3, theta1, phi12) given a discretized k list.
    We follow Scoccimarro 2015 page 4 for the definition of theta1, theta12 and phi12.
    
    mu1 = cos(theta1) = los dot k1;
    mu2 = cos(theta2) = los dot k2;
    
    theta12 and phi12 are respectively the polar and azimuthal angle in the frame formed by
    z' // k1, 
    x' in the same plane as los and k1, and k1 cross x' in the same direction as los x k1. 
    y' \perp z, y \perp x, such that x, y, z form a right-handed coordinates.

    Note: Currently dtheta1 and dphi12 calculations assumes linear spacing 
    for the input cos(theta1) and phi12.
    """

    def __init__(self, k, theta1, phi12, set_mu_to_zero=False):
        super().__init__(k)
        self._set_mu_to_zero = set_mu_to_zero
        print('You have set self._set_mu_to_zero to {}'.format(self._set_mu_to_zero)) #TODO use logger
        
        try:
            self._check_input_angle_range(theta1, phi12)
        except AnglesNotInRangeError as e:
            print(e.message)
            sys.exit() #TODO do real error handling

        self._theta1 = theta1
        self._phi12 = phi12

        self._theta1_in_deg = self._theta1 / np.pi * 180.0
        self._phi12_in_deg = self._phi12 / np.pi * 180.0

        self._ntheta1 = self._theta1.size
        self._nphi12 = self._phi12.size

        #TODO might need a function to setup the most general case of theta1, phi12 input
        self._dmu1 = np.cos(self._theta1[1]) - np.cos(self._theta1[0])
        self._dphi12 = self._phi12[1] - self._phi12[0]

        self._nori = self._ntheta1 * self._nphi12

        self._angle_array = self._get_angle_array()

        self._oriented_triangle_info = self._get_oriented_triangle_info()
        
    @property
    def nori(self):
        """An integer for the number of orientations per triangle shape."""
        return self._nori
        
    @property
    def theta1(self):
        """1d numpy array for the input theta1."""
        return self._theta1 

    @property
    def dmu1(self):
        """A float for the dmu1 = d(cos(theta1)), assuming linear spacing in mu1."""
        return self._dmu1
    
    @property
    def dphi12(self):
        """A float for the dphi12, assuming linear spacing in phi12."""
        return self._dphi12
    
    @property
    def phi12(self):
        """1d numpy array for the input phi12."""
        return self._phi12

    @property
    def theta1_in_deg(self):
        return self._theta1_in_deg

    @property
    def phi12_in_deg(self):
        return self._theta1_in_deg
    
    @property
    def angle_array(self):
        """2d numpy array where angle_array[iori, :] gives [theta1, phi12] in radians for iori-th orientation."""
        return self._angle_array
        
    @property
    def oriented_tri_array(self):
        """3d numpy array where oriented_tri_array[itri, iori, :] gives [k1, k2, k3, theta1, phi12] 
        for the itri-th triangle and iori-th orientation."""
        return self._oriented_triangle_info['oriented_triangle']

    @property
    def oriented_tri_index_array(self):
        """3d numpy array where oriented_tri_array[itri, iori, :] gives [ik1, ik2, ik3, itheta1, iphi12] 
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

    @staticmethod
    def _check_input_angle_range(theta1, phi12):
        min_theta1 = 0.0
        max_theta1 = np.pi
        min_phi12 = 0.0
        max_phi12 = 2*np.pi
        
        if np.any(theta1 > max_theta1):
            raise AnglesNotInRangeError(theta1, message='found theta1 larger than {}'.format(max_theta1))
        elif np.any(theta1 < min_theta1):
            raise AnglesNotInRangeError(theta1, message='found theta1 smaller than {}'.format(min_theta1))
        elif np.any(phi12 > max_phi12):
            raise AnglesNotInRangeError(phi12, message='found phi12 larger than {}'.format(max_phi12))
        elif np.any(phi12 < min_phi12):
            raise AnglesNotInRangeError(phi12, message='found phi12 smaller than {}'.format(min_phi12))

    def _get_angle_array(self):

        angle_array = np.zeros((self._nori, 2), dtype=float)

        iori = 0
        for itheta1, theta1 in enumerate(self._theta1):
            for iphi12, phi12 in enumerate(self._phi12):
                angle_array[iori, :] = np.array([theta1, phi12])
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
            for itheta1, theta1 in enumerate(self._theta1):
                for iphi12, phi12 in enumerate(self._phi12):
                    
                    k1 = self._tri_array[itri, 0]
                    k2 = self._tri_array[itri, 1]
                    k3 = self._tri_array[itri, 2]
                    mu1 = self._get_mu1(k1, k2, k3, theta1, phi12)
                    mu2 = self._get_mu2(k1, k2, k3, theta1, phi12)
                    mu3 = self._get_mu3(k1, k2, k3, theta1, phi12)

                    oriented_tri_mu_array[itri, iori, :] = np.array([mu1, mu2, mu3])
                    oriented_tri_index_array[itri, iori, :] = np.array([ik1, ik2, ik3, itheta1, iphi12])
                    oriented_tri_array[itri, iori, :] = np.array([k1, k2, k3, theta1, phi12])

                    iori = iori + 1

        oriented_triangle_info = {}
        oriented_triangle_info['mu'] = oriented_tri_mu_array
        oriented_triangle_info['index'] = oriented_tri_index_array
        oriented_triangle_info['oriented_triangle'] = oriented_tri_array

        return oriented_triangle_info
                    
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

    @staticmethod
    def get_theta12(k1, k2, k3):
        """arccos always returns angle between [0, pi], 
        so theta12 is the same for two triangles of opposite handedness."""
        theta12 = np.arccos(0.5 * (-k1*k1 - k2*k2 + k3*k3) / (k1 * k2))
        return theta12
          


