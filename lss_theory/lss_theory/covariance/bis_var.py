import numpy as np

from lss_theory.data_vector.data_vector import PowerSpectrum3D

class Bispectrum3DVariance():

    def __init__(self, p3d, data_spec_bis, dict_bis_var, do_cvl_noise=False):
        assert isinstance(p3d, PowerSpectrum3D)
        self._p3d = p3d

        self._galaxy_ps = self._p3d.get('galaxy_ps')
        
        self._data_spec_ps = self._p3d._data_spec

        self._data_spec_bis = data_spec_bis
        self._do_folded_signal = self._data_spec_bis.do_folded_signal
        self._triangle_spec = self._data_spec_bis.triangle_spec

        self._fsky = dict_bis_var['fsky']
        self._survey_volume_array = self._get_survey_volume_array()

        self._do_cvl_noise = dict_bis_var['do_cvl_noise']
        self._ps_noise = self._get_noise()

        self._bis_error = self._get_Bggg_error()

    @property
    def bis_error(self):
        return self._bis_error

    def _get_Bggg_error(self):
        """3d numpy array of shape (nb, nz, ntri)"""
    
        nsample = self._data_spec_bis.nsample
        nb = nsample**3
        nz = self._data_spec_bis.nz
        ntri = self._triangle_spec.ntri

        Bggg_var = np.zeros((nb, nz, ntri))
        
        for iz in range(nz):
            ib = 0
            for isample1 in range(nsample):
                for isample2 in range(nsample):
                    for isample3 in range(nsample):
                        Bggg_var[ib, iz, :] =  self._get_Bggg_var_diagonal(iz, isample1, isample2, isample3)
                        ib = ib + 1

        Nmodes = self._get_Nmodes()[np.newaxis, :, :]
        V = self._survey_volume_array[np.newaxis, :, np.newaxis]
        Bggg_var *= V / Nmodes
        
        return np.sqrt(Bggg_var)

    def _get_Bggg_var_diagonal(self, iz, isample1, isample2, isample3):
        """1d numpy array of shape (ntri,), no mu-dependence"""

        imu = 0 #TODO handled currently now, but need to eliminate later
        # Need to make galaxy_power without AP effects
        # and call that. 
        # TODO Need to do this properly; is there FOG suppression now?
        # maybe have a power spectrum version that is good for covariance?

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
            # use below if need to refine
            #choose_terms_1 = np.array([1., 0., 0., 0., 0., 0.])
            #choose_terms_1to6 = np.array([1., 1., 1., 1., 1., 1.])
            #choose_terms_1and2 = np.array([1., 1., 0., 0., 0., 0.])
            #choose_terms_1and3 = np.array([1., 0., 1., 0., 0., 0.])
            #choose_terms_equilateral = choose_terms_1to6
            #choose_terms_isoceles_k1_k2 = choose_terms_1and3
            #choose_terms_isoceles_k2_k3 = choose_terms_1and2
            #choose_terms_scalene = choose_terms_1

        return var
    
    def _get_Nmodes(self):
        """1d numpy array of shape (nz, ntri) for the number of modes in each triangle shape bin."""

        if self._do_folded_signal is True:
            nori = self._data_spec_bis.triangle_spec.nori 
            Sigma = self._data_spec_bis.Sigma_scaled_to_4pi
            assert np.allclose(nori * Sigma, 1)
        else:
            Sigma = self._data_spec_bis.Sigma

        (k1, k2, k3) = self._data_spec_bis.triangle_spec.get_k1_k2_k3()
        (dk1, dk2, dk3) = self._data_spec_bis.get_dk1_dk2_dk3()

        K_triangle = 8.0 * (np.pi * np.pi) * (k1 * k2 * k3) * (dk1 * dk2 * dk3) * Sigma
        V = self._survey_volume_array

        Nmodes = (V**2)[:, np.newaxis] / (2*np.pi)**6 * K_triangle

        return Nmodes

    def _get_survey_volume_array(self):
        """Returns 1d numpy array of shape (nz,) for the volume of the 
        redshift bins in units of (Mpc)^3 with fsky included."""
        return self._fsky * self._p3d._grs_ingredients._get_survey_volume_array() 

    def _get_observed_ps(self, iz, isample1, isample2, ik, imu):

        ips = self._data_spec_ps._dict_isamples_to_ips['%i_%i'%(isample1, isample2)]
        
        ps = self._galaxy_ps[ips, iz, ik, imu]

        if isample1 == isample2:
            if self._do_cvl_noise is False:
                ps += self._ps_noise[isample1, iz]

        return ps

    def _get_noise(self):
        """"Returns 2d numpy array of shape (nsample, nz) for the power spectrum
        noise, where noise = 1/number density, and is in unit of (Mpc)^3."""

        number_density = self._p3d._grs_ingredients._get_number_density_array()
        noise = 1./number_density

        return noise

    



    

        


        
