import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import sys
import os
from cobaya.yaml import yaml_load_file

from spherelikes.model import ModelCalculator


class CovCalculator():

    def __init__(self, data, args):
        self.data = data
        self.args = args
        self.galaxy_ps = self.data['galaxy_ps']
        self.nsample = self.data['aux']['nsample']
        self.make_dict()
        self.load_number_density()
        print(self.dict_ips_from_sample_pair['0,1'], 'expect 1')
        print(self.dict_ips_from_sample_pair['1,1'], 'expect 5')
        print(self.dict_sample_pair_from_ips['5'], 'expect (1, 1)')
        print(self.dict_sample_pair_from_ips['1'], 'expect (0, 1)')

    def get_cov(self):
        """Returns the covariance matrix for all power spectra between different 
        galaxy samples, assuming it is diagonal in z, k and mu. 

        We are also assuming that the covariance is symmetric under j1j2 <--> j1'j2':
            Cov[P_{j1, j2} P{j1', j2'}] = Cov[P_{j1', j2'} P{j1, j2}] 
        where j1, j2, j1', j2' are indices of the galaxy samples forming the 
        power spectra, and the z, k, mu dependence in P_{j1, j2}(z, k, mu) have been
        suppressed here for reasons mentioned above. 
        """
        # Careful: Do not change elements in self.data_1d, it would modify self.data
        # from outside of this class since it is created w/ np.ravel().
        self.get_data_1d()
        self.shape = self.galaxy_ps.shape
        (self.nps, self.nz, self.nk, self.nmu) = self.shape
        ntot = np.prod(self.shape)

        self.cov = np.zeros((ntot, ntot))
        for ips1 in range(self.nps):
            for ips2 in range(self.nps):
                print('(ips1, ips2) = ({} {})'.format(ips1, ips2))
                block_size = np.prod(self.shape[1:])
                ind1 = ips1 * block_size
                ind2 = ips2 * block_size
                block = self.get_block(ips1, ips2)
                self.cov[ind1:(ind1 + block_size), ind2:(ind2 + block_size)]\
                    = block
                # Note: if covriance expression is no longer symmetric in ips1<-->ips2, remove this.
                self.cov[ind1:(ind1 + block_size), ind2:(ind2 + block_size)]\
                    = block

        self.block_size = block_size

        return self.cov

    def get_and_save_invcov(self):
        self.get_cov()
        self.get_invcov()
        self.save()

    def save(self, fn=None):
        fn = os.path.join(self.args['output_dir'], 'fid.covmat')
        fn_invcov = os.path.join(self.args['output_dir'], 'fid.invcov')
        np.save(fn, self.cov)
        np.save(fn_invcov, self.invcov)
        print('Saved covariance matrix: {}'.format(fn))
        print('Saved inverse covariance matrix: {}'.format(fn_invcov))

    def do_test_cov_block_nxn(self, n):
        cov0 = self.cov[:2 * self.block_size, :2 * self.block_size]
        cov0 = cov0 * 1e-9
        self.plot_mat(cov0, plot_name='plot_block_%ix%i.png' % (n, n))

        print(np.linalg.det(cov0))
        inv = np.linalg.inv(cov0)
        self.get_inv_test(cov0, inv)

    def get_inv_test(self, mat, invmat, threshold=1e-3):
        id0 = np.diag(np.ones(mat.shape[0]))
        id1 = np.matmul(mat, invmat)
        id2 = np.matmul(invmat, mat)
        ind_big1 = np.where(id1 - id0 > threshold)
        ind_big2 = np.where(id2 - id0 > threshold)
        print(ind_big1)
        print(ind_big2)
        for i in range(ind_big1[0].size):
            x = ind_big1[0][i]
            y = ind_big1[1][i]
            print('(id1 - id0)[{},{}] = {}'.format(x,
                                                   y, id1[x, y] - id0[x, y]))
        for i in range(ind_big2[0].size):
            x = ind_big2[0][i]
            y = ind_big2[1][i]
            print('(id2 - id0)[{},{}] = {}'.format(x,
                                                   y, id2[x, y] - id0[x, y]))

    def get_invcov(self, rescale=1.0):
        try:
            self.invcov = np.linalg.inv(self.cov * rescale)
            print('self.invcov.shape = {}'.format(self.invcov.shape))
            return self.invcov
        except Exception as e:
            print('Error: {}'.format(e))

    def get_noise(self, a, b):
        """Return shot noise = 1/n if the same galaxy sample, for all galaxy redshift.
        Args:
            a, b are integers denoting indices (starting at 0) for galaxy samples."""
        if a == b:
            noise = np.array([1. / self.number_density[a, iz]
                              * np.ones((self.nk, self.nmu))
                              for iz in range(self.nz)])
        else:
            noise = np.zeros((self.nz, self.nk, self.nmu))
        print('noise.shape = {}'.format(noise.shape))
        # TODO turn into test:
        assert noise.shape == (self.nz, self.nk, self.nmu)
        print('a = {}, b = {}, noise = {}'.format(a, b, noise))
        # TODO turn into test:
        # get_noise(1, 1)[0,0,0] == 8.13008130e+01
        # get_noise(1, 1)[-1,0,0] == 1.47275405e+06
        return noise

    def load_number_density(self):
        # TODO need to change from h/Mpc to 1/Mpc kind of units
        pars = yaml_load_file(self.args['input_survey_pars'])
        # (nsample, nz)
        # TODO need to do this in a way that's independent of number of samples.
        self.number_density = np.array(
            [pars['numdens1'], pars['numdens2'], pars['numdens3'], pars['numdens4'], pars['numdens5']])
        print('self.number_density = {}'.format(self.number_density))

    def get_galaxy_ps(self, a, b):
        ips = self.dict_ips_from_sample_pair['%i,%i' % (a, b)]
        ps = self.galaxy_ps[ips, :, :, :]
        return ps

    def get_observed_ps(self, a, b):
        ps = self.get_galaxy_ps(a, b) + self.get_noise(a, b)
        return ps

    def get_block(self, ips1, ips2):
        (a, b) = self.dict_sample_pair_from_ips['%i' % ips1]
        (c, d) = self.dict_sample_pair_from_ips['%i' % ips2]
        cov = self.get_observed_ps(a, c) * self.get_observed_ps(b, d) \
            + self.get_observed_ps(a, d) * self.get_observed_ps(b, c)
        array = np.ravel(cov)
        print('array.size = {}, expect {}'.format(
            array.size, np.prod(self.shape[1:])))
        block = np.diag(array)
        print('block.shape = {}'.format(block.shape))

        return block

    def test_block(self):

        # nsample = (-1 + np.sqrt(1 + 8 * self.nps))/2.0 # put in utils later
        nsample = 5
        a = np.random.randint(nsample)
        b = np.random.randint(nsample)
        c = np.random.randint(nsample)
        d = np.random.randint(nsample)
        ips1 = self.dict_ips_from_sample_pair['%i,%i' % (a, b)]
        ips2 = self.dict_ips_from_sample_pair['%i,%i' % (c, d)]
        block = self.get_block(ips1, ips2)
        # self.get_observed_ps(a,c)
        ips_ac = self.dict_ips_from_sample_pair['%i,%i' % (a, c)]
        ips_bd = self.dict_ips_from_sample_pair['%i,%i' % (b, d)]
        ips_ad = self.dict_ips_from_sample_pair['%i,%i' % (a, d)]
        ips_bc = self.dict_ips_from_sample_pair['%i,%i' % (b, c)]

        iz = np.random.randint(self.nz)
        ik = np.random.randint(self.nk)
        imu = np.random.randint(self.nmu)

        idx = iz * self.nk * self.nmu + ik * self.nmu + imu
        block_entry = block[idx, idx]
        expected = (self.galaxy_ps[ips_ac, iz, ik, imu] + 1. / self.number_density[a, iz] * (a == c)) \
            * (self.galaxy_ps[ips_bd, iz, ik, imu] + 1. / self.number_density[b, iz] * (b == d)) \
            + (self.galaxy_ps[ips_ad, iz, ik, imu] + 1. / self.number_density[a, iz] * (a == d)) \
            * (self.galaxy_ps[ips_bc, iz, ik, imu] + 1. / self.number_density[b, iz] * (b == c))
        msg = 'block[{}, {}]={}, \n  expected={} for PS({},{}) x PS({},{}), for (iz, ik, imu) = ({}, {}, {})'.format(
            iz, iz, block_entry, expected, a, b, c, d, iz, ik, imu)
        print(msg)
        assert block_entry == expected, (msg)

        self.plot_mat(block, plot_name='plot_block.png')

    def make_dict(self):
        """Creates two dictionaries to map between power spectrum index and galaxy sample pairs.

        Note: For example, (j1, j2) = (0,1) is mapped to ips = 0, (1,1) is mapped to ips = 5 
        if there are 5 different galaxy samples. 

        In general, the power spectrum index ips is mapped to (j1, j2) the galaxy sample indices,
        where j1 < j2, while both (j1, j2) and (j2, j1) are mapped to the same power spectrum index.
        """
        self.dict_ips_from_sample_pair = {}
        self.dict_sample_pair_from_ips = {}
        ips = 0
        for j1 in range(self.nsample):
            for j2 in range(j1, self.nsample):
                self.dict_sample_pair_from_ips['%i' % ips] = (j1, j2)
                self.dict_ips_from_sample_pair['%i,%i' % (j1, j2)] = ips
                self.dict_ips_from_sample_pair['%i,%i' % (j2, j1)] = ips
                print('(j1, j2) = ({}, {}), ips = {}'.format(j1, j2, ips))
                ips = ips + 1

    def get_data_1d(self):
        self.data_1d = self.galaxy_ps.ravel()
        print('self.data_1d.shape = {}'.format(self.data_1d.shape))
        assert self.data_1d.size == np.prod(self.galaxy_ps.shape), \
            ('self.data_1d.size = {}, expect {}'
                .format(self.data_1d.size, np.prod(self.galaxy_ps.shape)))
        return self.data_1d

    def plot_mat(self, mat, plot_name=None):
        fig, ax = plt.subplots()
        mat1 = np.ma.masked_where(mat <= 0, mat)
        # print('np.where(mat<=0)'np.where(mat<=0))
        cs = ax.imshow(mat1, norm=LogNorm(), cmap='RdBu')
        cbar = fig.colorbar(cs)
        if plot_name is None:
            plot_name = os.path.join(self.args['output_dir'], 'plot_mat.png')
        fig.savefig(plot_name)
        print('Saved plot: {}'.format(plot_name))


def main():
    args = {
        'model_name': 'covariance_debug',
        'model_yaml_file': './inputs/sample_fid_model.yaml',
        'cobaya_yaml_file': './inputs/sample.yaml',
        'input_survey_pars': './inputs/survey_pars/survey_pars_v28_base_cbe.yaml',
        'output_dir': './data/covariance/',
    }

    model_calc = ModelCalculator(args)
    results = model_calc.get_results()

    cov_calc = CovCalculator(results, args)
    cov_calc.get_and_save_invcov()

    cov_calc.test_block()  # TODO turn into unit test

    # TODO use to test later
    # for n in range(cov_calc.nps):
    #    print('n = ', n)
    #    cov_calc.do_test_cov_block_nxn(n)

    # for exponent in [0,9]:
    #    rescale = 10**(-exponent)
    #    print('rescale = ', rescale)
    #    invcov = np.linalg.inv(cov * rescale)
    #    cov_calc.get_inv_test(cov * rescale, invcov, threshold=1e-3)


if __name__ == '__main__':
    main()
