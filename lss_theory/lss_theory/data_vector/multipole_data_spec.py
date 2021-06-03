import numpy as np

from lss_theory.utils.logging import class_logger

from lss_theory.data_vector.data_spec import Bispectrum3DBaseSpec, Bispectrum3DRSDSpec

class BispectrumMultipoleSpec(Bispectrum3DBaseSpec):

    """
    Sample Usage: 
        d = BispectrumMultipoleSpec(survey_par, multipole_data_spec_dict)

    You can direclty access variables: z, sigz, k, dk, mu, dmu, nps, 
        nsample, nz, nk, nmu, shape = (nps, nz, nk, nmu).
    """
    #TODO update documentation above

    #TODO add a different dictionary and treatment bloew
    def __init__(self, survey_par, data_spec_dict):
        super().__init__(survey_par, data_spec_dict)

        self._lmax = data_spec_dict['multipole_info']['lmax']
        self._do_nonzero_m = data_spec_dict['multipole_info']['do_nonzero_m']

        self._setup_nlm()
        self._setup_lm_list()

        self._b3d_rsd_spec = self._get_b3d_rsd_spec(survey_par, data_spec_dict)
        self._overwrite_shape_for_bis_mult()

    def _get_b3d_rsd_spec(self, survey_par, data_spec_dict):
        b3d_rsd_spec_dict = data_spec_dict.copy()
        b3d_rsd_spec_dict.pop('multipole_info', None)
        b3d_rsd_spec = Bispectrum3DRSDSpec(survey_par, b3d_rsd_spec_dict)
        return b3d_rsd_spec

    @property
    def b3d_rsd_spec(self):
        return self._b3d_rsd_spec

    @property
    def nlm(self):
        return self._nlm 

    @property
    def lmax(self):
        return self._lmax
    
    @property
    def do_nonzero_m(self):
        return self._do_nonzero_m
    
    @property
    def lm_list(self):
        return self._lm_list

    def _setup_nlm(self):
        self._nlm = np.sum([2*l+1 for l in range(0, self._lmax+1)])

    def _setup_lm_list(self):
        self._lm_list = [(l, m) for l in range(0, self._lmax+1) for m in range(-l, l+1)]

    def _overwrite_shape_for_bis_mult(self):
        self._shape = (self.nb, self.nz, self.ntri, self.nlm)
        self._transfer_shape = (self.nsample, self.nz, self.ntri, self.nlm)
    


