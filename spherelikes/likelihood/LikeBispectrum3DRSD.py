import numpy as np
import os
import pickle
import sys

from cobaya.likelihood import Likelihood

from spherelikes.utils.log import LoggedError, class_logger
from theory.utils.profiler import profiler

class LikeBispectrum3DRSD(Likelihood): 
    #TODO might subclass from a common base class in the future

    def initialize(self):
        """
        Prepares any computation, importing any necessary code, files, etc.
        """
        self.logger = class_logger(self)
        self.setup()

    def get_requirements(self):
        """
        Returns dictionary specifying quantities calculated by a theory code are needed
        """
        return {'galaxy_bis': None}

    #@profiler
    def logp(self, **params_values):
        """
        Taking a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.
        """

        # TODO Placeholder for future derived parameter and foreground calculations.
        #derived_param = self.provider.get_param('derived_param')
        my_foreground_amp = params_values['my_foreground_amp']

        if self.is_reference_likelihood is True:
            print('is_reference_likelihood = True == > return chi2=0.')
            chi2 = 0.0
        else:
            delta = self.simulated_data - self.get_sampled_data()
            (nb, nz, ntri, nori) = delta.shape
            
            chi2 = 0.0
            
            if self.cov_type == "full":

                for iz in range(nz):
                    for itri in range(ntri):
                        delta_tmp = (np.transpose(delta[:, iz, itri, :])).ravel()
                        invcov_tmp = self.invcov[:, :, iz, itri]
                        tmp = np.matmul(invcov_tmp, delta_tmp)
                        chi2 += np.matmul(delta_tmp, tmp)
            
            elif self.cov_type == "diagonal_in_orientation":

                msg = "You have specified cov_type = diagonal_in_orientation, but \
                    invcov loaded from file is has the wrong shape, \
                    need len(invcov.shape) == 5"
                assert len(self.invcov.shape) == 5, (msg)
                #TODO raise error instead of using assert
                for iz in range(nz):
                    for itri in range(ntri):
                        for iori in range(nori):
                            delta_tmp = delta[:, iz, itri, iori]
                            invcov_tmp = self.invcov[:, :, iz, itri, iori]
                            tmp = np.matmul(invcov_tmp, delta_tmp)
                            chi2 += np.matmul(delta_tmp, tmp)
            
            else:
                msg = "You specified cov_type = %s, but needs to be\
                    'full' or 'diagonal_in_orientation'."%(self.cov_type)
                raise LoggedError(self.logger, msg)
            self.logger.debug('chi2 = {}'.format(chi2))
            
        return -chi2 / 2

    def setup(self):
        self.logger.info('Setting up likelihood ...')

        if self.is_reference_likelihood is False:
            self.simulated_data = self.load_simulated_data()
            self.invcov = self.load_invcov()
        else:
            print('==> Not loading inverse covariance and simulated data vector.')

        self.logger.info('... Done.')

    def load_invcov(self):

        self.logger.info('Loading invcov ...')

        (nb, nz, ntri, nori) = self.simulated_data.shape

        if self.cov_type == "full":
            expected_shape = (nb*nori, nb*nori, nz, ntri)
        elif self.cov_type == "diagonal_in_orientation":
            expected_shape = (nb, nb, nz, ntri, nori)
        else:
            msg = "You specified cov_type = %s, but needs to be\
                    'full' or 'diagonal_in_orientation'."%(self.cov_type)
            raise LoggedError(self.logger, msg)

        try:
            
            invcov = np.load((self.invcov_path), allow_pickle=True)
            self.logger.info('Done loading invcov.')
            if invcov.shape != expected_shape:
                msg = 'Inverse covariance at %s does not match data vector dimensions. \n'\
                         % self.invcov_path \
                    + 'Loaded invcov has shape %s, but expecting shape = %s \
                        given cov_type = %s and simulated_data with shape %s.' %\
                        (invcov.shape, expected_shape, \
                        self.cov_type, self.simulated_data.shape)
                raise LoggedError(self.logger, msg)
            
            return invcov

        except FileNotFoundError as e:
            msg = '%s \n' % e \
                + 'Inverse covariance matrix does not exist. Run python scripts/generate_covariance.py first.'
            raise LoggedError(self.logger, msg)

    def load_simulated_data(self):
        try:
            return self.load_npy_file(self.sim_data_path) 
        except FileNotFoundError as e:
            msg = '%s' % e + '\n Simulated data vector does not exist.' \
                + '\n Run python scripts/generate_data.py first.'
            raise LoggedError(self.logger, msg)

    def load_npy_file(self, path):
        results = np.load(path)
        return results

    def get_sampled_data(self):
        galaxy_bis = self.provider.get_galaxy_bis()
        nuisance_model = self.get_nuisance_model()
        sample = galaxy_bis + nuisance_model
        return sample

    def get_nuisance_model(self): #TODO to implement
        return 0.0
