import os
import numpy as np
import json
import copy
import importlib

from lss_theory.params.cosmo_par import CosmoPar
from lss_theory.utils.file_tools import mkdir_p
class MetadataNotCompatibleError(Exception):
    """Raised when metadata from .json file is not as expected.
    Attributes:
        diff: difference in dictionary
        message: explanation of the error
    """

    def __init__(self, diff, message="Metadata not compatible."):
        self.diff = diff
        self.message = message + " (diff = {})".format(self.diff)
        super().__init__(self.message)


def get_dir(parent_dir, run_name, param, h):
    """Used by both Derivative and DerivativeConvergence classes
    to make the same format of directory name"""
    
    from lss_theory.utils.misc import strp
    #HACK remove if no longer needed
    #h_string = '_'.join([strp(h, fmt='%1.4e') for h in h_list])
    #params_string = '_'.join(params_list)
    h_string = strp(h, fmt='%1.4e') 
    
    dir_name = '%s/%s/%s/h_%s/'%(parent_dir, run_name, param, h_string)
    
    return dir_name


def unpack_param_list_from_def(params_list_in, param_set_def):
    """Returns parameter list with parameter sets unpacked"""

    params_list = copy.deepcopy(params_list_in)

    for param in params_list_in:
        print('param = {}'.format(param))
        if param[0] == "*":
            params_list.extend(param_set_def[param])
            params_list.remove(param)
    
    return params_list


class SingleDerivative():
    """
    Calculates derivative of a signal wtih respect to a single parameter.

    The derivatives are calculated numerically using a 5-pt stencil 
    by calling the theory code for computing the signal.

    We save the derivative data in a .npy file, and its metadata in 
    a .json file, unless overwrite = False and these files already exists.

    In comparing the metadata.json file to determine if the same derivative
    exists on disk, we omit the fields: ...
    """

    def __init__(self, info, ignore_cache, do_save, parent_dir, \
        func_get_signal_for_info):
        self._info = copy.deepcopy(info)
        self._ignore_cache = ignore_cache
        self._do_save = do_save
        self._parent_dir = parent_dir
        self._func_get_signal_for_info = func_get_signal_for_info

        self._setup()
        self._calculate()

    @property
    def data(self):
        return self._derivative

    @property
    def metadata(self):
        return self._metadata

    @property 
    def cosmo_par(self):
        return self._cosmo_par

    @property
    def param(self):
        return self._param

    @property
    def h_frac(self):
        return self._h_frac
        
    @property
    def h(self):
        return self._h
    
    @property
    def pfid(self):
        return self._pfid

    @property
    def data_path(self):
        return self._derivative_path

    @property
    def metadata_path(self):
        return self._metadata_path

    def _setup(self):
        self._cosmo_par = CosmoPar(self._info['cosmo_par_file'])
        self._param = copy.deepcopy(self._info['SingleDerivative']['param'])
        
        self._h_frac = self._info['SingleDerivative']['h_frac']
        self._h = self._info['SingleDerivative']['h']
        self._pfid = self._info['SingleDerivative']['pfid']

        self._setup_dir()
        self._setup_paths()
        self._setup_metadata()

    def _setup_dir(self):
        parent_dir = self._parent_dir
        run_name = self._info['run_name'] 
        self._dir = get_dir(parent_dir, run_name, self._param, self.h)
        print('self._dir = {}'.format(self._dir))
        mkdir_p(self._dir)

    def _setup_paths(self):
        self._derivative_path = os.path.join(self._dir, 'derivative.npy')
        self._metadata_path = os.path.join(self._dir, 'metadata.json')

    def _setup_metadata(self):
        self._metadata = copy.deepcopy(self._info)

    def _calculate(self):
        is_cache_usable = self._check_is_cache_usable()
        do_calculate = (is_cache_usable == False) or (self._ignore_cache == True)

        if do_calculate:
            self._derivative = self._get_derivative()
            if self._do_save == True:
                self._save()
        else:
            self._load_data()

    def _check_is_cache_usable(self):
        is_file_on_disk = os.path.exists(self._derivative_path) 
        if os.path.exists(self._metadata_path):
            is_metadata_same = self._check_is_metadata_same()
        else:
            is_metadata_same = False
        is_cache_usable = is_file_on_disk and is_metadata_same
        return is_cache_usable

    def _check_is_metadata_same(self):
        try:
            self._compare_metadata()
        except MetadataNotCompatibleError as e:
            print(e.message)
            return False
        else: 
            return True

    def _compare_metadata(self):

        metadata_on_file = self._load_metadata()
        
        from deepdiff import DeepDiff
        diff = DeepDiff(self._metadata, metadata_on_file, \
            ignore_string_case=True, \
            ignore_numeric_type_changes=True, \
            significant_digits=5,
            exclude_paths={
                "root['AllDerivatives']",
                "root['fisher']",
                "root['run_name']",
                }
            )
        if diff != {}:
            raise MetadataNotCompatibleError(diff)
    
    def _load_metadata(self):
        with open(self._metadata_path) as json_file:
            metadata_on_file = json.load(json_file)
        return metadata_on_file

    def _get_derivative(self):
        
        pvalues = self.pfid + np.array([-2, -1, 1, 2]) * self.h
        print('pvalues = {}'.format(pvalues))

        signals = np.array([self._func_get_signal_for_info(\
            self._get_info_for_param_and_pvalue(self.param, pvalue)) \
                for pvalue in pvalues])

        five_point_coeff = np.array([-1, 8.0, -8.0, 1.0])/(12.0 * self.h)
        
        derivative = np.sum(np.array(
                [five_point_coeff[i] * signals[i,...] for i in range(4)]
            ), axis=0)

        return derivative
    
    def _get_info_for_param_and_pvalue(self, param, pvalue):
        info = copy.deepcopy(self._info)
        info['overwrite_cosmo_par'] = {param: pvalue}    
        return info    

    def _save(self):
        self._save_data()
        self._save_metadata()  
        print('Saved derivatives in {} (see metadata at {})'.format(self.data_path, self.metadata_path))
    
    def _save_data(self):
        np.save(self.data_path, self.data)    

    def _save_metadata(self):
        with open(self.metadata_path, 'w') as json_file:
            json.dump(self._metadata, json_file, sort_keys=True, indent=4)

    def _load_data(self):
        self._derivative = np.load(self.data_path)    



class AllDerivatives():

    def __init__(self, info,
        ignore_cache, do_save, parent_dir):

        self._info = copy.deepcopy(info)
        self._ignore_cache = ignore_cache
        self._do_save = do_save
        self._parent_dir = parent_dir

        self._setup()
        self._calculate()
        
    @property
    def data(self):
        return self._all_derivatives

    @property
    def metadata(self):
        return self._metadata
    
    @property
    def params_list(self):
        return self._params_list
    
    @property
    def cosmo_par(self):
        return self._cosmo_par
    
    @property
    def other_par(self):
        return self._other_par

    def _setup(self):
        self._setup_cosmo()
        self._setup_params_list()
        self._setup_h_h_frac_and_pfid()
        self._setup_updated_metadata()

    def _setup_cosmo(self):
        self._cosmo_par = CosmoPar(self._info['cosmo_par_file'])
        self._other_par = self.get_other_par(self._info)

    def get_other_par(self, info):
        raise NotImplementedError
        
    #TODO could make the subclass implement this instead
    def _setup_params_list(self):
        self._params_list_in = copy.deepcopy(self._info['AllDerivatives']['params'])
        
        single_info = copy.deepcopy(self._info)
        self._param_set_def = self.get_param_set_definition(single_info)
        
        self._params_list = unpack_param_list_from_def(self._params_list_in, self._param_set_def)
        self._check_params_list_is_supported()
        
        self._nparam = len(self._params_list)

    def get_param_set_definition(self, info):
        raise NotImplementedError
    
    #TODO could make the subclass implement this instead
    def _check_params_list_is_supported(self):
        supported_params_list = self._other_par.params_list + self._cosmo_par.params_list
        for param in self._params_list:
            assert param in supported_params_list
        #TODO raise exception properly

    def _setup_h_h_frac_and_pfid(self):

        h_frac = self._info['AllDerivatives']['h_frac']
        self._h_frac_list  = self._h_frac_value_to_list(h_frac)
    
        self._h_list = []
        self._pfid_list = []

        for iparam, param in enumerate(self._params_list):
            pfid = self._get_pfid_for_param(param)
            h_frac = self._h_frac_list[iparam]
            h = pfid * h_frac

            # if h is specified, then overwrite h obtained from h_frac information with h.
            # this is useful for parameters with fiducial value = 0 like fnl.
            h_overwrite = self._info['AllDerivatives']['h'].get(param)
            print('param = {}, h_overwrite = {}'.format(param, h_overwrite))
            
            if h_overwrite is not None:
                h = h_overwrite
                h_frac = h/pfid
                self._h_frac_list[iparam] = h_frac
            
            self._h_list.append(h)
            self._pfid_list.append(pfid)

            print('param = {}, pfid = {}, h = {}, h_frac = {}'.format(param, pfid, h, h_frac))

    def _h_frac_value_to_list(self, value):
        """Returns a list for h_frac for every parameter. If only a value is 
        specificed for h_frac, then it returns a list of same length as the 
        params_list with the same h_frac value."""

        if not isinstance(value, list):
            value = [value] * self._nparam

        return value
    
    # TODO could also make subclass implement this instead
    def _get_pfid_for_param(self, param):
        try:
            pfid = getattr(self._cosmo_par, param)
            return pfid
        except AttributeError as attribute_error:
            # if not a cosmological parameter, then search in params_dict
            # for nuisance parameters:
            try:
                pfid = getattr(self._other_par, param)
                return pfid
            except AttributeError as attribute_error2:
                print('Could not find parameter {}, got AttributeError: {}'\
                    .format(param, attribute_error2))
    
    def _calculate(self):
        self._derivatives_list= self._get_derivatives_list()
        self._all_derivatives = self._get_all_derivatives()

    def _setup_updated_metadata(self):
        metadata = copy.deepcopy(self._info)
        metadata['AllDerivatives']['h_frac'] = self._h_frac_list
        metadata['AllDerivatives']['h'] = self._h_list
        self._metadata = metadata

    def _get_derivatives_list(self):
        """Returns a list of Derivative instances."""

        derivatives_list = []

        for iparam, param in enumerate(self._params_list):
            
            print('\n Getting derivative instance for a single parameter {}/{}: {} \n'.format(iparam+1, self._nparam, param))
            
            info = copy.deepcopy(self._info)
            info['SingleDerivative'] = {}
            info['SingleDerivative']['param'] = param
            info['SingleDerivative']['h_frac'] = self._h_frac_list[iparam]
            info['SingleDerivative']['h'] = self._h_list[iparam]
            info['SingleDerivative']['pfid'] = self._pfid_list[iparam]

            derivative = SingleDerivative(info, \
                ignore_cache=self._ignore_cache, do_save=self._do_save,
                parent_dir=self._parent_dir, \
                func_get_signal_for_info = self._get_signal_for_info)
                
            derivatives_list.append(derivative)

        return derivatives_list

    def _get_signal_for_info(self, info): 
        raise NotImplementedError

    def _get_all_derivatives(self):
        """Returns a numpy array of shape (nparam, *signal_shape), where signal_shape is 
        the shape of a single parameter derivative which is the same as the shape of the 
        signal."""
        all_derivatives = np.array([self._derivatives_list[iparam].data \
            for iparam in range(self._nparam)])
        return all_derivatives


#TODO
class DerivativeConvergence():

    """
    This class uses the class AllDerivatives to compute the derivatives 
    wrt to a set of parameters and check convergence for each parameter. 
    The final (converged) specifications are saved in a .json file, 
    that can be used to easily get the converged derivatives through the
    AllDerivatives which either loads or recalculate the derivatives.
    """

    def __init__(self, info, module_name, class_name, 
            ignore_cache=False, do_save=True, eps = 1e-3, parent_dir=None):

        self._info = copy.deepcopy(info)

        module = importlib.import_module(module_name)
        self._DerivativeClass_ = getattr(module, class_name)

        self._ignore_cache = ignore_cache
        self._do_save = do_save
        self._eps = eps
        self._parent_dir = parent_dir

        self._params_list_in = copy.deepcopy(self._info['derivatives']['params'])

        derivatives = self._DerivativeClass_(copy.deepcopy(info), \
                ignore_cache=False, do_save=False)
        print('derivatives.params_list = {}'.format(derivatives.params_list))
        self._params_list = derivatives.params_list

        self._param_set_def = getattr(module, 'get_param_set_definition')(info)
        self._params_list = unpack_param_list_from_def(self._params_list_in, self._param_set_def)

        self._h_frac_list = self._get_h_frac_list()

        self._is_converged_list, self._converged_h_frac_list, \
            self._converged_h_list, self._all_derivatives = \
            self._get_derivatives()
        self._metadata = self._get_metadata()
        self._print_status()

        self._setup_dir()
        self._setup_paths()
        self._save()
        
    @property
    def data(self):
        return self._all_derivatives

    @property
    def metadata(self):
        return self._metadata

    @property
    def cosmo_par(self):
        return self._derivative_fid.cosmo

    def _get_h_frac_list(self):
        h_frac = self._info['derivatives']['h_frac']
        if not isinstance(h_frac, list):
            h_frac = [h_frac] * len(self._params_list)
            self._info['derivatives']['h_frac'] = h_frac
        return h_frac

    def _get_metadata(self):
        metadata = copy.deepcopy(self._info)
        metadata['derivatives']['is_converged'] = self._is_converged_list
        metadata['derivatives']['h_frac'] = self._converged_h_frac_list
        metadata['derivatives']['h'] = self._converged_h_list
        return metadata
        
    def _setup_dir(self):
        run_name = self._info['run_name'] 
        self._dir = get_dir(self._parent_dir, run_name, \
            self._params_list, self._converged_h_list)
        print('self._dir = {}'.format(self._dir))
        mkdir_p(self._dir)

    def _setup_paths(self):
        self._metadata_path = os.path.join(self._dir, 'metadata.json')

    def _get_derivatives(self):
        self._setup_fiducial_derivatives()
        is_converged_list, converged_h_frac_list, converged_h_list, all_derivatives  = \
            self._iterate_all_derivatives_until_convergence()
        return is_converged_list, converged_h_frac_list, converged_h_list, all_derivatives


    def _print_status(self):

        if all(self._is_converged_list) == True:
            print('Got converged derivatives:')
        else:
            print('Some derivatives did not converge:')
        print('    is_converged_list = {}'.format(self._is_converged_list))
        print('    for parameters {}'.format(self._params_list))
        
    def _setup_fiducial_derivatives(self):
        info = copy.deepcopy(self._info)
        derivatives = self._DerivativeClass_(info, \
            ignore_cache=self._ignore_cache, do_save=True)
        self._derivative_fid = derivatives
        print(self._derivative_fid._h_list, 'self._derivative_fid._h_list')
        self._der_fid = derivatives.data

    def _iterate_all_derivatives_until_convergence(self):
        nparam = len(self._params_list)
        is_converged_list = []
        converged_h_frac_list = []
        converged_h_list = []
        shape = (nparam, *self._der_fid.shape[1:])

        all_derivatives = np.zeros(shape)
        for iparam, param in enumerate(self._params_list):
            print('Checking convergence for parameter {}/{}: {}'.format(iparam+1, nparam, param))
            convergence_info = self._iterate_derivatives_until_convergence(iparam)
            is_converged_list.append(convergence_info['is_converged'])
            converged_h_frac_list.append(convergence_info['h_frac'])
            converged_h_list.append(convergence_info['h'])
            all_derivatives[iparam, ...] = convergence_info['derivative']
        return is_converged_list, converged_h_frac_list, converged_h_list, all_derivatives

    def _iterate_derivatives_until_convergence(self, iparam, max_iter=6):

        info_prev = copy.deepcopy(self._info)
        der_prev = (self._der_fid[iparam, ...])[np.newaxis]
        is_converged = False

        # update params to a list of one element with the desired param
        desired_param = self._params_list[iparam]
        info_prev['derivatives']['params'] = [desired_param] 

        # determine whether to use h or h_frac
        use_h = desired_param in info_prev['derivatives']['h'].keys()
        
        if not use_h:
            h_frac_list = self._info['derivatives']['h_frac']
            h_frac_value = h_frac_list[iparam]
            info_prev['derivatives']['h_frac'] = h_frac_value

        i_iter = 0
        while ((not is_converged) and (i_iter < max_iter)):

            print('DerivativeConvergence: i_iter = {}'.format(i_iter))

            if i_iter > 0:
                der_prev = der_new
                info_prev = copy.deepcopy(info_new)

            info_new = copy.deepcopy(info_prev)
            
            if use_h:
                h_value = info_prev['derivatives']['h'][desired_param] 
                print('h_value = ', h_value)
                info_new['derivatives']['h'][desired_param] = 0.5 * h_value

                print('Calling derivative class with h = {}'\
                    .format(info_new['derivatives']['h'][desired_param]))
                    
            else:
                h_frac_value = info_prev['derivatives']['h_frac']
                h_frac_new = 0.5 * h_frac_value
                info_new['derivatives']['h_frac'] = h_frac_new

                print('Calling derivative class with h_frac = {}'\
                    .format(info_new['derivatives']['h_frac']))

                if i_iter == 0:
                    h_value = self._derivative_fid._h_list[0]
                else:
                    h_value = derivatives._h_list[0]
                
            derivatives = self._DerivativeClass_(info_new, \
                ignore_cache=self._ignore_cache, do_save=self._do_save)
            der_new = derivatives.data

            is_converged = self._check_convergence(der_prev, der_new)
            print('   i_iter = {}, is_converged = {}'.format(i_iter, is_converged))

            i_iter = i_iter + 1
        
        print('parameter = {}'.format(desired_param))

        if is_converged == True:
            print('   Derivatives convergence reached with eps = {}'.format(self._eps))
        else:
            print('   Derivatives not converged (eps = {}) but max_iter = {} reached'.format(self._eps, max_iter))
        
        if not use_h:
            print('   h_frac = {}'.format(h_frac_value))
        print('   h = {}'.format(h_value))
    
        print('\n')

        convergence_info = {
            'is_converged': is_converged,
            'h_frac': h_frac_value if not use_h else None,
            'h': h_value,
            'derivative': der_prev,
        }
        return convergence_info

    def _check_convergence(self, der1, der2):
        is_converged = False
        frac_diff = np.abs((der2-der1)/der1)
        max_frac_diff = np.nanmax(frac_diff[frac_diff != np.inf])
        print('max_frac_diff = {}'.format(max_frac_diff))
        if (frac_diff >= self._eps).any():
            ind = np.where(frac_diff >= self._eps)
            print('ind = {}'.format(ind))
            print('der1[ind] = {}'.format(der1[ind]))
            if np.allclose(der1[ind], np.zeros(der1[ind].shape)):
                is_converged = True
            print('is_converged = {}'.format(is_converged))
        else:
            is_converged = True
        return is_converged

    def _print_status(self):

        if all(self._is_converged_list) == True:
            print('Got converged derivatives:')
        else:
            print('Some derivatives did not converge:')
        print('    is_converged_list = {}'.format(self._is_converged_list))
        print('    for parameters {}'.format(self._params_list))
        
    def _setup_fiducial_derivatives(self):
        info = copy.deepcopy(self._info)
        derivatives = self._DerivativeClass_(info, \
            ignore_cache=self._ignore_cache, do_save=True)
        self._derivative_fid = derivatives
        print(self._derivative_fid._h_list, 'self._derivative_fid._h_list')
        self._der_fid = derivatives.data


