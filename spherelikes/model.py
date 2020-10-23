import pickle
import os
import sys
import shutil
import pathlib
import yaml
import numpy as np
from collections import namedtuple

from cobaya.yaml import yaml_load_file, yaml_dump_file
from cobaya.yaml import yaml_dump
from cobaya.model import get_model
from cobaya.tools import sort_cosmetic

from spherelikes.params import CobayaPar, SurveyPar
from spherelikes.theory.PowerSpectrum3D  import make_dictionary_for_bias_params

class ModelCalculator():

    # TODO should add docstrings, so it's obvious calling it from an outside script what to do.

    def __init__(self, args):

        model_args = namedtuple('ModelArgs', sorted(args))
        self.args = model_args(**args)

        self.model_name = self.args.model_name
        self.cosmo_par_file = self.args.cosmo_par_file
        self.cobaya_par_file = self.args.cobaya_par_file
        self.survey_par_file = self.args.survey_par_file
        self.output_dir = self.args.output_dir
        self.is_reference_model = self.args.is_reference_model
        self.is_reference_likelihood = self.args.is_reference_likelihood
        self.fix_default_bias = self.args.fix_default_bias

        self.cobaya_par = CobayaPar(self.cobaya_par_file)
        self.survey_par = SurveyPar(self.survey_par_file)

        self.setup_paths()
        self.setup_model()

    def setup_paths(self):
        pathlib.Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.fname = os.path.join(
            self.output_dir, self.model_name + '.pickle')

    def setup_model(self):
        self._make_model()
        self._make_initial_pars()
        self._update_pars()

    def get_and_save_results(self):
        results = self.get_results()
        self.test_results()
        self.save_results()
        return results

    def get_results(self):
        """Returns a dictionary of cosmological results and auxiliary quantities.

        - results['aux'] is a dictionary with keys k, z, mu.
        - results['Hubble'] and results['angular_diameter_distance'] give H(z) and D_A(z) at a list of z.
        - results['galaxy_ps'] give a 4-d numpy array for the galaxy power spectra.
        - results['logposterior'] gives the log posterior.
        """

        # Cobaya: Cosmological observables requested this way always correspond
        # to the last set of parameters with which the likelihood was evaluated.
        self.logposterior = self.model.logposterior(self.point)

        aux = self._get_auxiliary_variables()
        redshifts = aux['z']

        theory1 = self.model.theory["camb"]

        Hubble = theory1.get_Hubble(redshifts)
        angular_diameter_distance = theory1.get_angular_diameter_distance(
            redshifts)

        galaxy_ps = self.model.provider.get_galaxy_ps()

        H0 = theory1.get_param('H0')

        #TODO change this to deal only with SurveyPar
        self.survey_pars = yaml_load_file(self.args.survey_par_file)
        z_lo = np.array(self.survey_pars['zbin_lo'])
        z_hi = np.array(self.survey_pars['zbin_hi'])
        z_mid = 0.5 * (z_lo + z_hi)

        d_lo = theory1.get_comoving_radial_distance(z_lo)
        d_hi = theory1.get_comoving_radial_distance(z_hi)
        d_mid = theory1.get_comoving_radial_distance(z_mid)

        self.results = {
            'aux': aux,
            'Hubble': Hubble,
            'angular_diameter_distance': angular_diameter_distance,
            'comoving_radial_distance': d_mid,
            'comoving_radial_distance_lo': d_lo,
            'comoving_radial_distance_hi': d_hi,
            'galaxy_ps': galaxy_ps,
            'logposterior': self.logposterior,
            'H0': H0,
        }

        return self.results

    def save_results(self):
        pickle.dump(self.results, open(self.fname, "wb"))
        print('Results saved to %s.' % self.fname)
        self.copy_yaml_files()

    def test_results(self):
        name = self.cobaya_par.get_spherex_theory()
        theory = self.model.theory[name]
        ap = theory.get_AP_factor()
        if self.is_reference_model is True:
            assert np.all(ap == np.ones(theory.nz)), (ap, np.ones(theory.nz))

    def load_results(self):
        results = pickle.load(open(self.fname, "rb"))
        return results

    def copy_yaml_files(self):
        """Saves a copy of input model yaml file and cobaya sampler yaml file in the output directory."""
        yaml_path = os.path.join(
            self.output_dir, self.model_name + '.cosmo.yaml'
        )
        shutil.copy(self.cosmo_par_file, yaml_path)
        print('Copied over input model yaml file to %s.' % yaml_path)

        yaml_path = os.path.join(
            self.output_dir, self.model_name + '.cobaya.yaml'
        )
        shutil.copy(self.cobaya_par_file, yaml_path)
        print('Saved upated cobaya yaml file to %s.' % yaml_path)

        yaml_path = os.path.join(
            self.output_dir, self.model_name + '.chains.yaml'
        )
        try:
            yaml_dump_file(yaml_path, self.args._asdict())
        except OSError as e:
            os.remove(yaml_path)
            print('Warning: you are overwriting file at {}'.format(yaml_path))
            yaml_dump_file(yaml_path, self.args._asdict())

    def _make_model(self):
        info = yaml_load_file(self.cobaya_par_file)
        info = self._set_is_reference(info)
        self.model = get_model(info)

    def _make_initial_pars(self):
        self.point = dict(zip(
            self.model.parameterization.sampled_params(),
            self.model.prior.sample(ignore_external=True)[0]
        ))

    def _update_pars(self):
        fid_info = yaml_load_file(self.cosmo_par_file)
        print('fid_info ', fid_info)
        self.point.update(fid_info)
        if self.fix_default_bias is True:
            bias_params = make_dictionary_for_bias_params(\
                self.survey_par, \
                fix_to_default=self.fix_default_bias,\
                include_latex=False)
            print('bias_params', bias_params)
            self.point.update(bias_params)

    def _set_is_reference(self, info):
        info = self._overwrite_key_in_categories_with_value(
            info, 'is_reference_model', ['theory'], self.is_reference_model)
        info = self._overwrite_key_in_categories_with_value(
            info, 'is_reference_likelihood', ['likelihood'], self.is_reference_likelihood)
        return info

    def _overwrite_key_in_categories_with_value(self, info, key_to_set, list_categories, value):

        components = []
        for category in list_categories:
            for component in info[category].keys():
                if key_to_set in info[category][component].keys():
                    info[category][component][key_to_set] = value
                    components.append(category + ':, ' + component)

        print("Found components with key {} in category {}: {}".format(
            key_to_set, category, components))
        print('... Overwritten key {} = {} for these components.'.format(
            key_to_set, value))

        return info

    # TODO need a more general interface here
    def _get_auxiliary_variables(self):
        """Return a dictionary aux with keys 'k', 'mu' and 'z' used for computing results.
        """
        name = self.cobaya_par.get_spherex_theory()
        theory = self.model.theory[name]
        theory.must_provide(galaxy_ps={}, ap={})
        k = theory.k
        mu = theory.mu
        z = theory.z
        nsample = theory.nsample
        nps = theory.nps
        dk = theory.dk
        dmu = theory.dmu
        aux = {'k': k, 'mu': mu, 'z': z, 'nsample': nsample,
               'nps': nps, 'dk': dk, 'dmu': dmu}
        return aux


def main():

    CWD = os.getcwd()
    args = {
        'model_name': 'ref',
        'cosmo_par_file': CWD + '/inputs/cosmo_pars/planck2018_fiducial.yaml',
        'cobaya_par_file': CWD + '/inputs/cobaya_pars/ps_base.yaml',
        'survey_par_file': CWD + '/inputs/survey_pars/survey_pars_v28_base_cbe.yaml',
        'output_dir': CWD + '/data/ps_base/',
    }

    fid_calculator = ModelCalculator(args)
    fid_calculator.get_and_save_results()
    fid_calculator.load_results()

if __name__ == '__main__':
    main()
