import re
import sys
import subprocess
import shutil
import os
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('PDF')

from app.misc import utils_bis
from app.misc import utils
from app.misc import file_tools as ftl

logger = utils.file_logger(__file__)


def shrink_k(psm_mat_before, karray_before, k1=5e-2, k2=5e-1, fac1=15, fac2=5, fac3=5):

    kind1 = np.max(np.where(karray_before <= k1))
    kind2 = np.max(np.where(karray_before <= k2))
    logger.info("shrink_k: kind1 = {}, kind2 = {}".format(kind1, kind2))

    psm_mat = np.vstack((np.vstack(
        (psm_mat_before[0:kind1:fac1, :], psm_mat_before[kind1:kind2:fac2, :])), psm_mat_before[kind2::fac3, :]))
    karray = np.hstack((np.hstack(
        (karray_before[0:kind1:fac1], karray_before[kind1:kind2:fac2])), karray_before[kind2::fac3]))

    if np.isclose(karray[-1], karray_before[-1], rtol=1e-6, atol=0.0) == False:
        psm_mat = np.vstack((psm_mat, psm_mat_before[-1, :]))
        karray = np.hstack((karray, karray_before[-1]))

    return psm_mat, karray


def shrink_z(psm_mat_before, zarray_before, fac=2):

    psm_mat = psm_mat_before[:, ::fac]
    zarray = zarray_before[::fac]

    if np.isclose(zarray[-1], zarray_before[-1], rtol=1e-6, atol=0.0) == False:
        psm_mat = np.vstack((psm_mat, psm_mat_before[:, -1]))
        zarray = np.hstack((zarray, zarray_before[-1]))

    return psm_mat, zarray


def get_psm(do_calculate_ps, camb_or_class, do_nonlinear, zarray_ps, ip,
            do_mg, match_sigma8, cosmo_name, cis, use_original_sampling, do_high_pre, changed_pars):

    tag_cosmo = utils.get_tag_cosmo(do_nonlinear, cosmo_name, match_sigma8, do_mg, cis, changed_pars)        

    #output_root = utils.get_output_root(
    #    do_mg, cis, do_nonlinear, camb_or_class, ip, tag_cosmo)
    output_root = utils.get_output_root(camb_or_class, ip, do_nonlinear, tag_cosmo) #do_nonlinear needed for camb for now

    fname = ip.class_dir + output_root + '_background.dat'
    found_file = os.path.exists(fname) 
    if (do_calculate_ps == True) or (found_file == False):
        if found_file == True:
            logger.info('We found file {} exists, but will overwrite it since do_calculate_ps = {}...'.format(fname, do_calculate_ps))       
        else:
            logger.info('Did not find file {}, calculating now...'.format(fname))
        logger.info('==> Calculating Pm(k) from scratch by calling {}'.format(camb_or_class))
        if camb_or_class == 'camb':
            call_camb(zarray_ps, do_nonlinear, output_root, ip, cis)
        elif camb_or_class == 'class':
            call_class(zarray_ps, do_nonlinear, output_root, ip,
                       do_mg, match_sigma8, cis, do_high_pre)
        else:
            logger.info(
                "\n Warning: camb_or_class needs to be camb or class\n ")

    if camb_or_class == 'camb':
        psm_mat, zarray, karray = load_and_setup_psm(zarray_ps,
                                                     do_nonlinear, output_root, ip)
    elif camb_or_class == 'class':
        psm_mat, zarray, karray = load_and_setup_psm_class(zarray_ps,
                                                           do_nonlinear, output_root, ip)
    else:
        logger.info('\n Warning: camb_or_class needs to be camb or class\n, \
            now you have {}'.format(camb_or_class))

    # psm = utils.interp_func_2d(psm_mat, zarray, karray)
    # psm = utils.interp_func_2d(psm_mat, zarray, karray, \
    #    do_logf = False, do_logx = False, do_logy = False) # less k sample size

    psize_before = psm_mat.size
    psm_mat_before = psm_mat
    karray_before = karray
    zarray_before = zarray

    logger.info("psm_mat.shape = {}, karray.shape = {}, zarray.shape = {}\n"
                .format(psm_mat.shape, karray.shape, zarray.shape))

    if do_nonlinear > 0:
        if do_mg == True:  # in case there are features and need a different scheme
            k1 = 0.05
            k2 = 5e-1
            fac1 = 10
            fac2 = 5
            fac3 = 10  # test for halofit mg: cM and cB separately
        else:
            k1 = 0.05
            k2 = 5e-1
            fac1 = 45
            fac2 = 5
            fac3 = 10  # test for halofit gr
    else:
        if do_mg == True:
            k1 = 0.05
            k2 = 5e-1
            fac1 = 10
            fac2 = 5
            fac3 = 15  # tested for linear mg
        else:
            k1 = 0.05
            k2 = 5e-1
            fac1 = 45
            fac2 = 5
            fac3 = 15  # tested for linear gr

    if use_original_sampling == True:
        k1 = 0.05
        k2 = 5e-1
        fac1 = 1
        fac2 = 1
        fac3 = 1

    logger.info("using original sampling = {}".format(use_original_sampling))

    psm_mat_final, karray = shrink_k(
        psm_mat_before, karray_before,
        k1=k1, k2=k2, fac1=fac1, fac2=fac2, fac3=fac3)
    logger.info("do_nonlinear = {}, do_mg = {}".format(do_nonlinear, do_mg))
    logger.info("using k1 = {}, k2 = {}, fac1 = {}, fac2 = {}, fac3 = {}".format(
        k1, k2, fac1, fac2, fac3))

    # psm_mat, zarray = shrink_z(psm_mat, zarray_before, fac=2)

    # psm_z0 = psm_mat[:,0] # assume psm[:,0] is z = 0
    psm = utils.interp_func_2d(psm_mat_final, zarray, karray,
                               do_logf=True, do_logx=False, do_logy=True,
                               fmat_ref=psm_mat_before, \
                               # x_ref = zarray_before, \
                               # scal_cols_by_vec = 1./psm_z0
                               y_ref=karray_before, method='cubic')

    logger.debug('size before and after: {}, {}'.format(
        psize_before, psm_mat_final.size))
    logger.debug('psm_mat.shape = {}, karray.shape = {}, zarray.shape = {}'.format(
        psm_mat_final.shape, karray.shape, zarray.shape))

    psm_z0 = psm_mat_final[:, 0]

    return psm, psm_z0, karray, output_root


def call_camb(zarray_ps, do_nonlinear,
              camb_output_root, ip, transfer_interp_matterpower=None):
    nz_camb = zarray_ps.size
    cosmo = ip.cosmo
    logger.info(
        "Calling camb with nz_camb = {}; zarray_ps = {}; \
            do_nonlinear = {}, camb_outpu_root = {}".format(
            nz_camb, zarray_ps, do_nonlinear, camb_output_root
        ))
# Call camb to generate matter power spectrum given z bin centers
# do_nonlinear = 3
# 0: linear, 1: non-linear matter power (HALOFIT), 2: non-linear CMB lensing (HALOFIT),
# 3: both non-linear matter power and CMB lensing (HALOFIT)
# zarray_ps = (d.z_bin_edges[1:] + d.z_bin_edges[:-1])/2.0 #zarray_ps = array([ 0.1,  0.3,  0.5,  0.7,  0.9,  1.3,  1.9,  2.5,  3.1])
    print('Calling camb with As = ', cosmo.A_s)

    cur_dir = os.getcwd()
    os.chdir(ip.camb_dir)

    fn_ini_sample = 'params_sample_bispec_spherex.ini'
    fn_ini = 'params_' + camb_output_root + '.ini'
    shutil.copyfile(fn_ini_sample, fn_ini)
    ftl.modify_param_in_file(fn_ini, 'output_root', camb_output_root)
    As = cosmo.A_s
    ftl.modify_param_in_file(fn_ini, 'scalar_amp(1)             ', str(As))
    ftl.modify_param_in_file(
        fn_ini, 'scalar_spectral_index(1)  ', str(cosmo.n_s))
    ftl.modify_param_in_file(fn_ini, 'use_physical', 'F')
    ftl.modify_param_in_file(fn_ini, 'omega_baryon   ', str(cosmo.Om_b))
    ftl.modify_param_in_file(fn_ini, 'omega_lambda   ', str(cosmo.Om_L))
    ftl.modify_param_in_file(fn_ini, 'omega_cdm      ',
                             str(cosmo.Om_m-cosmo.Om_b))
    ftl.modify_param_in_file(fn_ini, 'hubble', 		str(
        cosmo.h0*100.0))  # probably not used
    ftl.modify_param_in_file(
        fn_ini, 'transfer_kmax           ', str(ip.camb_transfer_kmax))
    ftl.modify_param_in_file(
        fn_ini, 'transfer_k_per_logint   ', str(ip.camb_transfer_k_per_logint))

    for i_zbin in np.arange(0, nz_camb):
        ftl.modify_param_in_file(fn_ini, 'transfer_redshift(%i)' % (
            i_zbin+1), str(zarray_ps[nz_camb-1-i_zbin]))
        ftl.modify_param_in_file(fn_ini, 'transfer_filename(%i)' % (
            i_zbin+1), 'transfer_%i.dat' % (i_zbin+1))
        ftl.modify_param_in_file(fn_ini, 'transfer_matterpower(%i)' % (
            i_zbin+1), 'matterpower_%i.dat' % (i_zbin+1))
    # z = 0
    ftl.modify_param_in_file(
        fn_ini, 'transfer_redshift(%i)' % (nz_camb+1), '0')
    ftl.modify_param_in_file(fn_ini, 'transfer_filename(%i)' %
                             (nz_camb+1), 'transfer_0.dat')
    ftl.modify_param_in_file(fn_ini, 'transfer_matterpower(%i)' %
                             (nz_camb+1), 'matterpower_0.dat')
    ftl.modify_param_in_file(fn_ini, 'transfer_num_redshifts', str(nz_camb+1))
    ftl.modify_param_in_file(fn_ini, 'do_nonlinear', do_nonlinear)

    if transfer_interp_matterpower != None:
        ftl.modify_param_in_file(
            fn_ini, 'transfer_interp_matterpower', transfer_interp_matterpower)

    cmd = ["./camb", fn_ini]
    logfile = camb_output_root + '.out'
    with open(logfile, 'w') as f:
        rc = subprocess.call(cmd, stdout=f)
        print('    camb call done.')
        sys.stdout.flush()
    paramname = 'at z =  0.000 sigma8 (all matter) ='
    sigma_8_current = float(ftl.get_string_from_file(
        logfile, paramname, 35, 44, 0))
    print('    sigma_8_current = ', sigma_8_current)
    if np.isclose(sigma_8_current, cosmo.sigma_8, rtol=1e-6, atol=0.0) == False:
        new_As = As * (cosmo.sigma_8/sigma_8_current)**2
        print('    recalculate with As = %4.3e --> %4.3e' % (As, new_As))
        ftl.modify_param_in_file(fn_ini, 'scalar_amp(1)            ', new_As)
        cmd = ["./camb", fn_ini]
        logfile = camb_output_root + '.out'
        with open(logfile, 'w') as f:
            rc = subprocess.call(cmd, stdout=f)
            print('    camb call done.')
            sys.stdout.flush()
        paramname = 'at z =  0.000 sigma8 (all matter) ='
        sigma_8_current = float(ftl.get_string_from_file(
            logfile, paramname, 35, 44, 0))
        print('    sigma_8_current = ', sigma_8_current)
        cosmo.A_s = new_As
        print('    set cosmo.A_s to ', cosmo.A_s)
    else:
        print('    leaving cosmo.A_s as ', cosmo.A_s)
    print('... DONE.')
    print('')
# As =  2.1e-9 -> sigma8 (all matter) = 0.8
# so new As = (0.817/0.8)**2*2.1e-9 = 2.19019828e-9 (As propto sigma8^2)
    os.chdir(cur_dir)


def call_class(zarray_ps, do_nonlinear, class_output_root,
               ip, do_mg, match_sigma8, cis, do_high_pre):

    cosmo = ip.cosmo
    nz_ps = zarray_ps.size
    # print('[call_class]: nz_ps = ', nz_ps)
    # print('[call_class]: zarray_ps = ', zarray_ps)
    # print('[call_class]: do_nonlinear = ', do_nonlinear)
    print('[call_class]: class_output_root = ', class_output_root)
    print('\n')

    if do_mg == True:
        class_parameters_smg = utils.get_class_parameters_smg(cis)
    else:
        class_parameters_smg = ip.class_parameters_smg  # not used for GR
        # Call class to generate matter power spectrum given z bin centers
        # do_nonlinear = 3
        # 0: linear, 1: non-linear matter power (HALOFIT), 2: non-linear CMB lensing (HALOFIT),
        # 3: both non-linear matter power and CMB lensing (HALOFIT)
        # zarray_ps = (d.z_bin_edges[1:] + d.z_bin_edges[:-1])/2.0 #zarray_ps = array([ 0.1,  0.3,  0.5,  0.7,  0.9,  1.3,  1.9,  2.5,  3.1])
    # print('Calling class with As = ', d.as_spec)
    
    cur_dir = os.getcwd()
    os.chdir(ip.class_dir)

    if do_mg == True:
        fn_ini_sample = 'bis_mg_sample.ini'
    else:
        fn_ini_sample = 'bis_gr_sample.ini'

    fn_ini = os.path.basename(class_output_root) + '.ini'

    if do_high_pre == True:
        # fn_pre = 'pk_ref_high_precision.pre'
        fn_pre = ''
    else:
        fn_pre = ''

    shutil.copyfile(fn_ini_sample, fn_ini)

    ftl.modify_param_in_file(fn_ini, 'root ', class_output_root + '_')
    # defaults are planck best-fit here
    As = cosmo.A_s
    ftl.modify_param_in_file(fn_ini, 'A_s ', str(As))
    ftl.modify_param_in_file(fn_ini, 'n_s ', str(cosmo.n_s))
    ftl.modify_param_in_file(fn_ini, 'YHe ', str(cosmo.YHe))
    ftl.modify_param_in_file(fn_ini, 'z_reio ', str(cosmo.z_reio))
    if hasattr(cosmo, 'Om_bh2'):
        assert cosmo.Om_bh2 == cosmo.Om_b * cosmo.h0 * cosmo.h0, (cosmo.Om_bh2, cosmo.Om_b * cosmo.h0 * cosmo.h0)
    ftl.modify_param_in_file(fn_ini, 'omega_b ', str(
        cosmo.Om_b * cosmo.h0 * cosmo.h0))
    ftl.modify_param_in_file(fn_ini, 'omega_cdm ', str(
        (cosmo.Om_m-cosmo.Om_b) * cosmo.h0 * cosmo.h0))
    ftl.modify_param_in_file(fn_ini, 'h ', 		str(cosmo.h0))

    ftl.modify_param_in_file(fn_ini, 'P_k_max_h/Mpc ',
                             str(ip.class_Pkmax_hMpc))
    ftl.modify_param_in_file(fn_ini, 'gravity_model ',
                             str(ip.class_gravity_model))
    ftl.modify_param_in_file(fn_ini, 'parameters_smg ',
                             class_parameters_smg)  # str(ip.class_parameters_smg))

    z_pk = np.hstack((0.0, zarray_ps))
    z_pk_str = str(list(z_pk))[1:-1]
    print('z_pk_str', z_pk_str)
    ftl.modify_param_in_file(fn_ini, 'z_pk ', z_pk_str)

    if do_nonlinear == 0:
        ftl.modify_param_in_file(fn_ini, 'non linear ', '')
    else:
        ftl.modify_param_in_file(fn_ini, 'non linear ', 'HALOFIT')

    cmd = ["./class", fn_ini, fn_pre]

    logfile = class_output_root + '.out'  # [:-1]
    with open(logfile, 'w') as f:
        rc = subprocess.call(cmd, stdout=f)
        print('    class call done.')
        sys.stdout.flush()

    with open(logfile, 'r') as f:
        filecontent = f.read()
        try:
            if ('error' in filecontent) or ('Error' in filecontent):
                raise ValueError('class call had an error', filecontent)
        except ValueError as err:
            print(err.args)

    if match_sigma8 == True:
        paramname = ' -> sigma8='
        line = ftl.get_line_from_file(logfile, paramname, 0)
        print('line', line)
        line = line[11:19]
        print('line', line)
        sigma_8_current = float(re.findall("\d+\.\d+", line)[0])
        # print('utils.extract_params:', utils.extract_params(line, [('sigma8=','(')] ) )
        # sigma_8_current = float( utils.extract_params( line, [('sigma8=','(')] ) [0] )
        # sigma_8_current = float(ftl.get_string_from_file(logfile, paramname, 11, 19, 0))
        print('    sigma_8_current = ', sigma_8_current)

        if np.isclose(sigma_8_current, cosmo.sigma_8, rtol=1e-6, atol=0.0) == False:
            new_As = As * (cosmo.sigma_8/sigma_8_current)**2
            print('    recalculate with As = %4.3e --> %4.3e' % (As, new_As))
            ftl.modify_param_in_file(fn_ini, 'A_s ', new_As)
            cmd = ["./class", fn_ini, fn_pre]
            logfile = class_output_root + '.out'  # [:-1]
            with open(logfile, 'w') as f:
                rc = subprocess.call(cmd, stdout=f)
                print('    class call done.')
                sys.stdout.flush()
            line = ftl.get_line_from_file(logfile, paramname, 0)
            print('line', line)
            line = line[11:19]
            print('line', line)
            sigma_8_current = float(re.findall("\d+\.\d+", line)[0])
            print('    sigma_8_current = ', sigma_8_current)
            cosmo.A_s = new_As
            print('    set cosmo.A_s to ', cosmo.A_s)
        else:
            print('    leaving cosmo.A_s as ', cosmo.A_s)

    print('... DONE.')
    print('')
# As =  2.1e-9 -> sigma8 (all matter) = 0.8
# so new As = (0.817/0.8)**2*2.1e-9 = 2.19019828e-9 (As propto sigma8^2)
    os.chdir(cur_dir)

# --------- Load and setup interpolation functions for matter power spectrum -------

# Load matter power spectrum into a matrix:
# karray - k/h [h/Mpc];
# psm_mat - first col: z = 0;
# 	col2+ matter power spectrum [(Mpc/h)^3] for z in zarray_ps,
# 	increasing z with col number;


# col1 of psm is k (not k/h)
def load_and_setup_psm(zarray_ps, do_nonlinear, camb_output_root, ip):
    nz_camb = zarray_ps.size
    camb_outdir = ip.camb_dir

    psm_tmp = np.genfromtxt(camb_dir + camb_output_root +
                            '_matterpower_0.dat', names=True)  # load z = 0
    psm_mat = np.empty((psm_tmp['kh'].size, nz_camb+1))
    karray = psm_tmp['kh']  # k (not k/h)
    psm_mat[:, 0] = psm_tmp['P']  # z=0
    print('got here?????')
    for iter in np.flipud(np.arange(1, nz_camb+1)):  # nz_camb, nz_camb-1, ..., 1
        psm_tmp = np.genfromtxt(
            camb_outdir + camb_output_root + '_matterpower_%i.dat' % iter, names=True)
        try:
            if (np.any(np.isclose(karray, psm_tmp['kh'], rtol=1e-6, atol=0.0) == False)):
                raise ValueError(
                    "get_ps.py: k/h column of test_psm_%s.dat does not agree" % iter)
        except ValueError as ve:
            sys.exit(ve)
        psm_mat[:, nz_camb-iter+1] = psm_tmp['P']  # 1, 2 ... nz_camb

    zarray = np.hstack((np.array([0.0]), zarray_ps))

    return psm_mat, zarray, karray
# psm: col1 is karray, col2 is the smallest z>0, col before last is biggest z>0, last col is z = 0
# psm_mat: col1 is z = 0, col2 is the smallest z>0, last col is biggest z>0


def load_and_setup_psm_class(zarray_ps, do_nonlinear, class_output_root, ip):

    cur_dir = os.getcwd()
    logger.debug('load_and_setup_psm_class curdir = {}'.format(cur_dir))
    outdir = ip.class_dir
    nz_camb = zarray_ps.size

    fn = outdir + class_output_root + '_z1_pk.dat'
    data = np.genfromtxt(fn)
    karray = data[:, 0]  # h/Mpc
    psm_mat = np.empty((karray.size, nz_camb+1))

    zarray = np.hstack((np.array([0.0]), zarray_ps))

    for iz, z in enumerate(zarray):

        fn = outdir + class_output_root + '_z%i_pk.dat' % (iz+1)
        if do_nonlinear != 0:
            fn = fn.replace('pk', 'pk_nl')

        logger.debug("loading file {}".format(fn))

        # check if z in file is the same
        paramname = '# Matter power spectrum P(k) at redshift'
        z_file = float(ftl.get_string_from_file(fn, paramname, 43, 100))

        logger.debug(" => z_file = {}, z wanted = {}".format(z_file, z))

        try:
            if np.isclose(z_file, z, rtol=1e-5, atol=0.0) == False:
                raise ValueError(
                    "get_ps.py [load_and_setup_psm_class]: z_file = %s is not the same as z = %s from file " % (z_file, z))
        except ValueError as ve:
            sys.exit(ve)

        data = np.genfromtxt(fn)
        karray_tmp = data[:, 0]  # h/Mpc

        # check if karray is the same as previous files
        try:
            if (np.any(np.isclose(karray_tmp, karray, rtol=1e-6, atol=0.0) == False)):
                raise ValueError(
                    "get_ps.py [load_and_setup_psm_class]: k/h column of %s does not agree" % (fn))
        except ValueError as ve:
            sys.exit(ve)

        psm_mat[:, iz] = data[:, 1]  # (Mpc/h)^3

    return psm_mat, zarray, karray


def load_psm_z0(do_nonlinear, camb_output_root):  # col1 of psm is k (not k/h)
    camb_outdir = camb_dir
    # tag_nonlinear = utils_bis.get_tag_nonlinear(do_nonlinear)
    # camb_output_root = camb_output_root + tag_nonlinear

    print('[load_psm_z0]: camb_output_root = ', camb_output_root)
    psm_tmp = np.genfromtxt(camb_dir + camb_output_root +
                            '_matterpower_0.dat', names=True)  # load z = 0
    psm_z0 = np.empty((psm_tmp['kh'].size, 2))
    karray = psm_tmp['kh']  # k (not k/h)
    psm_z0[:, 0] = karray
    psm_z0[:, 1] = psm_tmp['P']   # z=0
    print('\n')
    return psm_z0  # this is just P not k^3 /(2pi)^3 * P
