import numpy as np

from theory.data_vector.data_vector import P3D

class Bispectrum3DVariance():

    def __init__(self, p3d, data_spec_bis):
        assert isinstance(p3d, P3D)
        self._p3d = p3d

        self._galaxy_ps = self._p3d.get('galaxy_ps')
        self._data_spec_ps = self._p3d._data_spec

        self._data_spec_bis = data_spec_bis
        self._triangle_spec = self._data_spec_bis.triangle_spec

        #HACK revive when we need number density and volume
        #self._number_density_in_invMpc = self._get_number_density_in_invMpc()
        #self._survey_volume_array = self._get_survey_volume_array()

        self._bis_error = self._get_Bggg_error()

    #TODO Use getter?
    @property
    def bis_error(self):
        return self._bis_error

    def _get_Bggg_error(self):
        """4d numpy array of shape (nb, nz, ntri, nori)"""
    
        nsample = self._data_spec_bis.nsample
        nb = nsample**3
        nz = self._data_spec_bis.nz
        ntri = self._triangle_spec.ntri
        #nori = self._triangle_spec.nori

        Bggg_error = np.zeros((nb, nz, ntri))
        
        for iz in range(nz):
            ib = 0
            for isample1 in range(nsample):
                for isample2 in range(nsample):
                    for isample3 in range(nsample):
                        Bggg_error[ib, iz, :] =  self._get_Bggg_error_diagonal(iz, isample1, isample2, isample3)
                        ib = ib + 1

        return Bggg_error

    def _get_Bggg_error_diagonal(self, iz, isample1, isample2, isample3):
        """1d numpy array of shape (ntri,), no mu-dependence"""

        imu = 0 #TODO make sure handling correctly.

        (ik1_array, ik2_array, ik3_array) = self._triangle_spec.get_ik1_ik2_ik3()
        var = np.zeros(self._triangle_spec.ntri)

        for itri in range(self._triangle_spec.ntri):
            ik1 = ik1_array[itri]
            ik2 = ik2_array[itri]
            ik3 = ik3_array[itri]
            var[itri] = self._get_observed_ps(iz, isample1, isample1, ik1, imu) \
                * self._get_observed_ps(iz, isample2, isample2, ik2, imu) \
                * self._get_observed_ps(iz, isample3, isample3, ik3, imu)
            #TODO ignore other 5 terms for now, so not accurate for isoceles and equilateral!!
            #fsky = 0.75 #HACK need to put this in properly

        #var /= (fsky * self._survey_volume_array[iz])
        
        return np.sqrt(var)

    #TODO to be completed!!
    #def _get_survey_volume_array(self):
     #   """Returns 1d numpy array of shape (nz, ) for the volume of the 
     #   redshift bins.

      #  Note: The calculation is approximate: We do not integrate dV over
      #  dz, but instead integrate dchi and use the bin center for the area.
      #  """
        
        # TODO might want to have more finely integrated volume
        # this would require getting comoving_radial_distance from camb
        # at more redshift values.
        # Might want to decouple this from requirements in theory module
        # where this is requested for now.
        #d_lo = self._p3d._grs_ingredients.get('comoving_radial_distance')
        #d_hi = self._p3d._grs_ingredients.get('comoving_radial_distance')
        #d_mid = self._p3d._grs_ingredients.get('comoving_radial_distance')
        #dist = d_mid
        #V = (4.0 * np.pi) * dist**2 * (d_hi - d_lo)
        #return V

    def _get_observed_ps(self, iz, isample1, isample2, ik, imu):
        ips = self._data_spec_ps._dict_isamples_to_ips['%i_%i'%(isample1, isample2)]
        ps = self._galaxy_ps[ips, iz, ik, imu]
        #if isample1 == isample2:
        #    ps += self._get_noise(isample1, iz)
        return ps

    def _get_noise(self, isample, iz):
        return 1.0/self._number_density_in_invMpc[isample, iz]



    

        


        
