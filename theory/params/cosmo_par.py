import numpy as np

class CosmoPar(object):
    def __init__(self, cosmo_par_file):
        self.cosmo_par_file = cosmo_par_file #TODO hacked, to be changed

    def set_par_planck2018(self):
        self.logger.info('Using Planck 2018 best-fit cosmology')
        # fiducial cosmology (flat LCDM) - Planck 2018 best-fit (baseline 2.5):
        self.Om_m     	= 0.3158
        self.Om_L     	= 0.6842
        self.Om_K       = 0.0
        self.Om_r       = 0.0
        self.sigma_8   	= 0.8120
        self.n_s    	= 0.96589
        self.A_s		= 2.10037291389e-9
        self.k0 		= 0.05
        self.alpha_spec  = 0.
        self.Om_b		= 0.04937574338
        self.h0			= 0.6732
        self.c 			= 299792.458 # to be checked
        self.YHe         = 0.245398
        self.z_reio      = 7.680
        self.b1_perc     = 0.0
        self.bz1         = 0.0 # more like a \Delta bz1, a deviation from b(z1) = 1
        self.bz2         = 0.0
        self.bz3         = 0.0
        self.bz4         = 0.0
        self.bz5         = 0.0
        self.b2z1        = 0.0
        self.b2z2        = 0.0
        self.b2z3        = 0.0
        self.b2z4        = 0.0
        self.b2z5        = 0.0
        #self.set_Om_bh2()

    def set_Om_bh2(self):
        self.Om_bh2 = self.Om_b * self.h0 * self.h0

    def set_Om_b(self):
        self.Om_b = self.Om_bh2 / self.h0 / self.h0

    def set_prior_Om_b(self): #assumes self.prior['Om_bh2'] is defined
        self.prior['Om_b'] = self.prior['Om_bh2']/(self.h0 * self.h0)

    def set_prior_planck2018(self):
        self.prior = {}
        self.prior['n_s'] = 0.0044
        self.prior['Om_m'] = 0.0084
        self.prior['h0'] = 0.0060
        self.prior['A_s'] = 0.031e-9
        self.prior['sigma_8'] = 0.0073
        # May need later
        self.prior['As_e2tau'] = 0.012e-9
        self.prior['Om_mh2'] = 0.0013
        self.prior['Om_Lambda'] = 0.0084
        self.prior['Om_bh2'] = 0.00015
        self.prior['b1_perc'] = 25
        self.set_prior_Om_b()

    # not used now
    def get_derived(self, param):
        if param == 'Om_bh2':
            return self.Om_b * self.h0 * self.h0
        elif param == 'Om_mh2':
            return self.Om_m * self.h0 * self.h0    

    def get_par(self, param):
        if hasattr(self, param) == False:
            value = self.get_derived(param)
            setattr(self, param, value)
        return getattr(self, param)
