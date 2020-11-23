import numpy as np

class DataSpec():

    """
    Sample Usage:
        d = DataSpec(survey_par, data_spec_dict)

    You can direclty access variables: z, sigz, k, dk, mu, dmu, nps, 
        nsample, nz, nk, nmu, shape = (nps, nz, nk, nmu).
    These cannot be set from outside the class, 
        but can be accessed as e.g. k = d.k, because we 
        used the @property decorator around a getter.
    """

    def __init__(self, survey_par, data_spec_dict):
        self._survey_par = survey_par
        self._dict = data_spec_dict
        self._setup_specs()
        self._setup_derived_specs

    def _setup_specs(self):
        self._setup_z()
        self._setup_sigz()
        self._setup_k()
        self._setup_mu()

    def _setup_z(self):
        self._z = self._survey_par.get_zmid_array()
    
    def _setup_sigz(self):
        self._sigz = self._survey_par.get_sigz_array()

    def _setup_k(self):
        self._k = np.logspace(np.log10(self._dict['kmin']), np.log10(self._dict['kmax']), self._dict['nk'])
        self._dk = self._get_dk(self._k)

    def _get_dk(self, k):
        """Return dk array.

        Note: This assumes that k is uniformly spaced in log space. 
        It computes dk by taking the difference in linear space between the
        log bin-centers, assuming that the first and last bin are only half 
        the usual log bin size."""

        #TODO change this prescription AFTER test against old code!!
        logk = np.log(k)
        logk_mid = (logk[:-1] + logk[1:]) / 2
        dk = np.zeros(k.size)
        dk[1:-1] = np.exp(logk_mid)[1:] - np.exp(logk_mid)[0:-1]
        dk[0] = np.exp(logk_mid[0]) - np.exp(logk[0])
        dk[-1] = np.exp(logk[-1]) - np.exp(logk_mid[-1])
        return dk

    def _setup_mu(self):
        self._mu_edges = np.linspace(0, 1, self._dict['nmu'] + 1)
        self._mu = (self._mu_edges[:-1] + self._mu_edges[1:]) / 2.0
        self._dmu = self._mu_edges[1:] - self._mu_edges[:-1]

        assert self._mu.size == self._dmu.size, ('mu and dmu do not have the same size: {}, {}'.format(
            self._mu.size, self._dmu.size))
    
    def _setup_derived_specs(self):
        self._setup_n()
        self._setup_shape()

    def _setup_n(self):
        self._nsample = self._dict['nsample']
        self._nps = int(self._dict['nsample'] * (self._dict['nsample'] + 1) / 2)
        self._nz = self._z.size
        self._nk = self._k.size
        self._nmu = self._mu.size

        assert self._nmu == self._dict['nmu']
        assert self._nk == self._dict['nk'] - 1 #TODO to be changed so input nk means nkbin
    
    def _setup_shape(self):
        self._shape = (self._nps, self._nz, self._nk, self._nmu)

    @property
    def z(self):
        return self._z
    
    @property
    def sigz(self):
        return self._sigz
    
    @property
    def k(self):
        return self._k

    @property
    def mu(self):
        return self._mu

    @property
    def dk(self):
        return self._dk

    @property
    def dmu(self):
        return self._dmu

    @property
    def shape(self):
        return self._shape

    @property
    def nps(self):
        return self._nps

    @property
    def nz(self):
        return self._nz

    @property
    def nk(self):
        return self._nk

    @property
    def nmu(self):
        return self._nmu
    
    def set_k_and_mu_actual(self, ap_perp, ap_para):
        """Return four 3-d numpy arrays of shape (nz, nk, mu) 
        for the actual values of k_perp, k_para, k and mu
        given the two AP factors in directions perpendicular to 
        and parallel to the line-of-sigh, ap_perp and ap_para, 
        each specified as a 1-d numpy array of size self._d.size:
            k_perp = k_perp|ref * D_A(z)|ref / D_A(z),
            k_para = k_para|ref * (1/H(z))|ref / (1/H(z)),
        where
            k// = mu * k,
            kperp = sqrt(k^2  - k//^2) = k sqrt(1 - mu^2).
        """
        
        assert ap_perp.shape == self._z.shape, (ap_perp.shape, self._z.shape)
        assert ap_para.shape == self._z.shape, (ap_para.shape, self._z.shape)

        k_perp_ref = self._k[:, np.newaxis] * \
            np.sqrt(1. - (self._mu**2)[np.newaxis, :])
        k_para_ref = self._k[:, np.newaxis] * self._mu[np.newaxis, :]

        k_actual_perp = k_perp_ref[np.newaxis, :, :] * \
            ap_perp[:, np.newaxis, np.newaxis]

        k_actual_para = k_para_ref[np.newaxis, :, :] * \
            ap_para[:, np.newaxis, np.newaxis]

        k_actual = np.sqrt(k_actual_perp**2 + k_actual_para**2)

        mu_actual = k_actual_para/k_actual

        self._k_actual_perp = k_actual_perp
        self._k_actual_para = k_actual_para
        self._k_actual = k_actual
        self._mu_actual = mu_actual

    
    



