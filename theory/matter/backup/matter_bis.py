import numpy as np
from app.misc import constants as cst
from app import mg
from app.misc import utils
import sys
import os

logger = utils.file_logger(__file__)

def plot_check_psm_smooth():
    global psm_z0, psm_smooth
    print('plotting psm_smooth to check if its braodpower matches roughly psm_z0')
    fn = 'data/planck_camb_56106182_matterpower_smooth_z0.dat'
    psm_smooth = np.genfromtxt(fn)
    fig, ax = plt.subplots()
    ax.loglog(psm_z0[:, 0], psm_z0[:, 1], 'b.',
              markersize=1, label='Pm(k, z=0)')
    ax.loglog(psm_smooth[:, 0], psm_smooth[:, 1], 'r.',
              markersize=1, label=r'Pm(k, z=0) smooth')
    ax.legend()
    ax.grid()
    ax.set_xlabel(r'$k$')
    ax.set_ylabel(r'$P_m(k)$')
    pfname = plot_dir + 'plot_check_pm_smooth.pdf'
    plt.savefig(pfname)
    print("Plot saved: ", pfname)

def get_dot_k1k2(k1, k2, k3):
    return 0.5 * (-k1**2 - k2**2 + k3**2)

def get_F2(k1, k2, k3):  # 5th digit different for equilateral k=
    cos = get_dot_k1k2(k1, k2, k3) / k1 / k2
    ans = 5.0 / 7.0 + 0.5 * (k1 / k2 + k2 / k1) * cos + 2.0 / 7.0 * cos**2
    return ans



def get_alpha(z_array, k_array, cosmo, psm_zk):  # psm
    # need to use cosmo.h0 here.
    Deltasqu = cosmo.A_s * (k_array / (cosmo.k0 / cosmo.h0))**(cosmo.n_s - 1.0)
    ps0 = (2.0 * np.pi**2) / (k_array**3) * Deltasqu
    #alpha = 5.0/3.0 * np.sqrt(psm.get_value(z_array, k_array)/ps0)
    alpha = 5.0 / 3.0 * np.sqrt(psm_zk / ps0)
    return alpha
    # alpha = delta/(9/10*phi_P) = 10/9 * sqrt(Pm/Pphi) = 10/9 * sqrt(Pm/(4/9*P_R)) = 5/3 sqrt(Pm/P_R)
    # where P_phi = 4/9 *P_R (Dodelson Modern Cosmology 2nd Edition Eq. 6.97)


def get_b_nl_j(z_array, k_array, fnl, cosmo, psm_zk, los):  # psm
    b_ELG_Da = 0.84
    alpha = get_alpha(z_array, k_array, cosmo, psm_zk)
    a_array = 1.0 / (1.0 + z_array)
    b_lin = b_ELG_Da / (los.get_da(a_array) / los.get_da(1.0))
    nl = 2.0 * fnl * (b_lin - 1.0) * cst.delta_c / alpha
    return nl, alpha

# Eq. 5.7, 5.9 of 1011.1513
def get_Bmmm_of_z_and_k(z_array, k1_array, k2_array, k3_array, fnl, cosmo, psm, los, *args):
    pmk1 = psm.get_value(z_array, k1_array)
    pmk2 = psm.get_value(z_array, k2_array)
    pmk3 = psm.get_value(z_array, k3_array)
    # with fnl = 1, this is dbj(k,z)/dfnl
    nl1, alpha1 = get_b_nl_j(z_array, k1_array, fnl, cosmo, pmk1, los)
    nl2, alpha2 = get_b_nl_j(z_array, k2_array, fnl, cosmo, pmk2, los)
    nl3, alpha3 = get_b_nl_j(z_array, k3_array, fnl, cosmo, pmk3, los)
    ##prefac1 = (b_lin + nl1) * (b_lin + nl2) * (b_lin + nl3)
    t1 = 2.0 * (alpha3 / (alpha1 * alpha2)) * fnl + \
        2.0 * get_F2(k1_array, k2_array, k3_array)
    t2 = 2.0 * (alpha2 / (alpha1 * alpha3)) * fnl + \
        2.0 * get_F2(k1_array, k3_array, k2_array)
    t3 = 2.0 * (alpha1 / (alpha2 * alpha3)) * fnl + \
        2.0 * get_F2(k2_array, k3_array, k1_array)
    term1 = t1 * pmk1 * pmk2 + t2 * pmk1 * pmk3 + t3 * pmk2 * pmk3
    # TO CHECK: previously ans2 = prefac2 * (t1 * pmk1 * pmk2 + t2 * pmk1 * pmk3 + t3 * pmk2 * pmk3)
    # factor of 2 missing here, c.f. Eq. 5.5, unless their b10 is different than
    # Elisabeth's b2! --- CHECK LATER
    Bmmm = term1  # + term2  (wrong do not need b20 term2 here here)
    return Bmmm


def get_F2_terms(k1_array, k2_array, k3_array, z_array, cosmo, bis):
   
    ##prefac1 = (b_lin + nl1) * (b_lin + nl2) * (b_lin + nl3)
    F2_1 = 2.0 * get_F2(k1_array, k2_array, k3_array)
    F2_2 = 2.0 * get_F2(k1_array, k3_array, k2_array)
    F2_3 = 2.0 * get_F2(k2_array, k3_array, k1_array)

    except ValueError as err:
        logger.error('err.args = {}'.format(err.args))
        sys.exit()
    
    return F2_1, F2_2, F2_3

def get_fnl_coefficients(z_array, k1_array, k2_array, k3_array, fnl, \
    cosmo, pmk1, pmk2, pmk3, los):
    if fnl < 1e-4:  # not a tested number, just approximation
        t01 = 0.0
        t02 = 0.0
        t03 = 0.0
    else:  # with fnl = 1, this is dbj(k,z)/dfnl
        nl1, alpha1 = get_b_nl_j(z_array, k1_array, fnl, cosmo, pmk1, los)
        nl2, alpha2 = get_b_nl_j(z_array, k2_array, fnl, cosmo, pmk2, los)
        nl3, alpha3 = get_b_nl_j(z_array, k3_array, fnl, cosmo, pmk3, los)
        t01 = 2.0 * (alpha3 / (alpha1 * alpha2)) * fnl
        t02 = 2.0 * (alpha2 / (alpha1 * alpha3)) * fnl
        t03 = 2.0 * (alpha1 / (alpha2 * alpha3)) * fnl
    return t01, t02, t03

# Eq. 5.7, 5.9 of 1011.1513
def get_Bmmm_of_ell(l1, l2, l3, bis):
    """Returns an array of Bmmm(k,z) for array of k_i corresponding to l_i/chi/h0, i = 1,2,3; 
    and pk12 = Pm(k_1) * Pm(k_2), pk23 = Pm(k_2) * Pm(k_3), pk13 = Pm(k_3) * Pm(k_1).
    Args:
            l1: integer.
            l2: integer.
            l3: integer.
            F2_flag: Integer 0, 1 or 2 
                    0 for normal F2; 1 for F2 from lambda(t) for MG;
                    2 for F2_abc for nonlinear GR only, not MG.
            bis: An instance of bispectrum defined in bispec.py.
    """

    F2_flag = bis.F2_flag
    los = bis.los
    z_array = los.z_l_array
    chi_array = los.chi_array
    cosmo = bis.args.cosmo
    fnl = bis.args.fnl

    opt_kcut = bis.args.opt_kcut
    kcut = bis.args.kcut

    k1_array = l1 / chi_array / cosmo.h0  # k is in h/Mpc
    k2_array = l2 / chi_array / cosmo.h0  # k is in h/Mpc
    k3_array = l3 / chi_array / cosmo.h0  # k is in h/Mpc
        
    pmk1 = bis.psm_dict[l1]
    pmk2 = bis.psm_dict[l2]
    pmk3 = bis.psm_dict[l3]

    t01, t02, t03 = get_fnl_coefficients(z_array, k1_array, k2_array, k3_array, fnl, \
        cosmo, pmk1, pmk2, pmk3, los)

    F2_1, F2_2, F2_3 = get_F2_terms(F2_flag, k1_array, k2_array, \
        k3_array, z_array, cosmo, bis)

    t1 = t01 + F2_1
    t2 = t02 + F2_2
    t3 = t03 + F2_3
    pk12 = pmk1 * pmk2
    pk13 = pmk1 * pmk3
    pk23 = pmk2 * pmk3
    term1 = t1 * pk12 + t2 * pk13 + t3 * pk23
    #Bmmm = term1  # + term2  (wrong do not need b20 term2 here here)
    # TO CHECK: previously ans2 = prefac2 * (t1 * pmk1 * pmk2 + t2 * pmk1 * pmk3 + t3 * pmk2 * pmk3)
    # factor of 2 missing here, c.f. Eq. 5.5, unless their b10 is different than
    # Elisabeth's b2! --- CHECK LATER

    if opt_kcut == 0:
        Bmmm = term1
    elif opt_kcut == 1:
        if k1_array[1] > k1_array[0]: # ascending
            ind1 = np.max(np.where(k1_array <= kcut))
            ind2 = np.max(np.where(k2_array <= kcut))
            ind3 = np.max(np.where(k3_array <= kcut))
            ind_max = np.min([ind1, ind2, ind3])
            ind = np.arange(ind_max+1) # including value at ind_max
            assert all(k1_array[ind] < kcut)
            Bmmm = np.zeros(chi_array.size)
            Bmmm[ind] = term1[ind] 
        elif k1_array[1] < k1_array[0]:  #descending
            Bmmm = np.zeros(chi_array.size)
            try:
                ind1 = np.min(np.where(k1_array <= kcut))
                ind2 = np.min(np.where(k2_array <= kcut))
                ind3 = np.min(np.where(k3_array <= kcut))
                ind_max = np.max([ind1, ind2, ind3])
                ind = np.arange(ind_max, k1_array.size) # including value at ind_max
                assert all(k1_array[ind] < kcut)
                Bmmm[ind] = term1[ind] 
            except ValueError: # all zero
                logger.debug('l1, l2, l3 = {}, {}, {}'.format(l1, l2, l3))
    return Bmmm, pk12, pk23, pk13

#TODO no longer used
def get_Bmmm_eq(z_array, k1_array, fnl, cosmo, psm, los, use_F2_abc, bis):
    #print('get_Bmmm_eq: fnl = ', fnl)
    pmk1 = psm.get_value(z_array, k1_array)
    # with fnl = 1, this is dbj(k,z)/dfnl
    nl1, alpha1 = get_b_nl_j(z_array, k1_array, fnl, cosmo, pmk1, los)
    if use_F2_abc == False:
        t1 = 2.0 / alpha1 * fnl + 2.0 * get_F2(k1_array, k1_array, k1_array)
    else:
        t1 = 2.0 / alpha1 * fnl + 2.0 * \
            get_F2_abc(k1_array, k1_array, k1_array, z_array, bis)
    term1 = 3.0 * t1 * pmk1 * pmk1
    # TO CHECK: previously ans2 = prefac2 * (t1 * pmk1 * pmk2 + t2 * pmk1 * pmk3 + t3 * pmk2 * pmk3)
    # factor of 2 missing here, c.f. Eq. 5.5, unless their b10 is different than
    # Elisabeth's b2! --- CHECK LATER
    Bmmm = term1
    return Bmmm
