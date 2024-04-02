import numpy as np
import emcee

from ctypes import *
from scipy.stats import gaussian_kde
from multiprocessing import Pool

import os
from scipy.stats import multivariate_normal

libc = cdll.LoadLibrary("./evolv2_merger.so")
np.set_printoptions(suppress=True)

def evolv2(m1, m2, logtb, e, alpha_1, alpha_2, vk1, theta1, phi1, omega1, vk2, theta2, phi2, omega2, acc_lim1, acc_lim2, qHG, qGB, logZ):
    m2, m1 = np.sort([m1,m2],axis=0)
    tb = 10**logtb
    metallicity = 10**logZ
    sigma = 0

    z = byref(c_double(metallicity))
    zpars = np.zeros(20).ctypes.data_as(POINTER(c_double))
    acc_lim = acc_lim.flatten().ctypes.data_as(POINTER(c_double))
    qkstar2 = byref(c_double(qHG))
    sigma = byref(c_double(0.0))
    qkstar3 = byref(c_double(qGB))
    natal_kick = np.zeros((2,5))
    #natal_kick[0,0] = vk1
    #natal_kick[0,1] = phi1
    #natal_kick[0,2] = theta1
    #natal_kick[0,3] = omega1
    #natal_kick[0,4] = 3
    #natal_kick[1,0] = vk2
    #natal_kick[1,1] = phi2
    #natal_kick[1,2] = theta2
    #natal_kick[1,3] = omega2
    #natal_kick[1,4] = 3
    natal_kick = natal_kick.T.flatten().ctypes.data_as(POINTER(c_double))
    alpha = np.zeros(2)
    alpha[0] = alpha_1
    alpha[1] = alpha_2
    alpha = alpha.flatten().ctypes.data_as(POINTER(c_double))
    
    libc.evolv2_global_(z,zpars,acc_lim,alpha,qkstar2,qkstar3,natal_kick)

    mass = np.array([m1,m2]).ctypes.data_as(POINTER(c_double))
    mass0 = np.array([m1,m2]).ctypes.data_as(POINTER(c_double))
    epoch = np.array([0.0,0.0]).ctypes.data_as(POINTER(c_double))
    ospin = np.array([0.0,0.0]).ctypes.data_as(POINTER(c_double))
    tb = byref(c_double(tb))
    ecc = byref(c_double(e))
    tphysf = byref(c_double(13700.0))
    dtp = byref(c_double(0.0))
    rad = np.array([0.0,0.0]).ctypes.data_as(POINTER(c_double))
    lumin = np.array([0.0,0.0]).ctypes.data_as(POINTER(c_double))
    massc = np.array([0.0,0.0]).ctypes.data_as(POINTER(c_double))
    radc = np.array([0.0,0.0]).ctypes.data_as(POINTER(c_double))
    menv = np.array([0.0,0.0]).ctypes.data_as(POINTER(c_double))
    renv = np.array([0.0,0.0]).ctypes.data_as(POINTER(c_double))
    B_0 = np.array([0.0,0.0]).ctypes.data_as(POINTER(c_double))
    bacc = np.array([0.0,0.0]).ctypes.data_as(POINTER(c_double))
    tacc = np.array([0.0,0.0]).ctypes.data_as(POINTER(c_double))
    tms = np.array([0.0,0.0]).ctypes.data_as(POINTER(c_double))
    bhspin = np.array([0.0,0.0]).ctypes.data_as(POINTER(c_double))
    tphys = byref(c_double(0.0))
    bkick = np.zeros(20).ctypes.data_as(POINTER(c_double))
    kick_info = np.zeros(34).ctypes.data_as(POINTER(c_double)) # Fortran treat n-D array differently than numpy
    bpp_index_out = byref(c_int64(0))
    bcm_index_out = byref(c_int64(0))
    kick_info_out = np.zeros(34).ctypes.data_as(POINTER(c_double))
    t_merge = byref(c_double(0.0))
    m_merge = np.array([0.0,0.0])
    bpp_out=np.zeros([1000,43]).flatten().ctypes.data_as(POINTER(c_double))
    kstar = np.array([1,1]).ctypes.data_as(POINTER(c_double))
    
    libc.evolv2_(kstar,mass,tb,ecc,z,tphysf,
    dtp,mass0,rad,lumin,massc,radc,
    menv,renv,ospin,B_0,bacc,tacc,epoch,tms,
    bhspin,tphys,zpars,bkick,kick_info,
    bpp_index_out,bcm_index_out,kick_info_out,
    t_merge,m_merge.ctypes.data_as(POINTER(c_double)),bpp_out)
    bpp = np.hstack((bpp, np.ones((bpp.shape[0], 1))*np.random.uniform(0, 1000000000))
    bpp = bpp_out._arr.reshape(43,1000)[:,0:bpp_index_out._obj.value].T
    return t_merge._obj.value,np.sort(m_merge)[::-1],kick_info_out._arr.reshape(17,2).T,bpp

#Porb: 185.63 +/- 0.05
#Eccentricity: 0.45  +/- 0.01 
#MBH: 9.80  +/- 0.20
#Mstar: 0.93 +/- 0.05
mean = np.array([24.0, 2.6])
cov = np.array([[1.5**2, 0,], [0, 0.2**2]])
rv = multivariate_normal(mean, cov)
#m1, m2, tb, e, alpha, vk1, theta1, phi1, omega1, vk2, theta2, phi2, omega2, acc_lim, qc_kstar2, qc_kstar3, Z
m1lo = 5.0
m2lo = 5.0
tblo = 5.0
elo = 0.0
alphalo_1 = 0.1
alphalo_2 = 0.1
vklo = 0.0
thetalo = 0.0
philo = -90.0
omegalo = 0.0
acc_lim_1lo = 0.0
acc_lim_2lo = 0.0
qc_kstar2lo = 0.5
qc_kstar3lo = 0.5
Zlo = 0.0001

m1hi = 150.0
m2hi = 150.0
tbhi = 5000.0
ehi = 0.9
alphahi_1 = 20.0
alphahi_2 = 20.0
vkhi = 300.0
thetahi = 360.0
phihi = 90.0
omegahi = 360
acc_lim_1hi = 0.0
acc_lim_2hi = 0.0
qc_kstar2hi = 10.0
qc_kstar3hi = 10.0
Zhi = 0.03

#m1, m2, logtb, e, alpha_1, alpha_2, vk1, theta1, phi1, omega1, vk2, theta2, phi2, omega2, acc_lim, logZ
lower_bound = np.array([m1lo, m2lo, np.log10(tblo), elo, alphalo_1, alphalo_2, vklo, thetalo, philo, omegalo, vklo, thetalo, philo, omegalo, acc_lim_1lo, acc_lim_2lo, qc_kstar2lo, qc_kstar3lo, np.log10(Zlo)])
upper_bound = np.array([m1hi, m2hi, np.log10(tbhi), ehi, alphahi_1, alphahi_2, vkhi, thetahi, phihi, omegahi, vkhi, thetahi, phihi, omegahi, acc_lim_1hi, acc_lim_2hi, qc_kstar2hi, qc_kstar3hi, np.log10(Zhi)])

lower_bound[5] = lower_bound[4]
upper_bound[5] = upper_bound[4]


def likelihood(coord):
    for i in range(len(coord)):
        if (coord[i]<lower_bound[i] or coord[i]>upper_bound[i]):
            return -np.inf
    result = evolv2(*coord)
    gw_coord = np.array([result[1][0],result[1][1]])
    if np.any(gw_coord) == 0.0: return -np.inf
    return rv.logpdf(gw_coord) 

n_dim = len(lower_bound)
n_walkers = 1024
p0 = np.random.uniform(lower_bound, upper_bound, size=(n_walkers, len(lower_bound)))
n_steps = 1000

sampler = emcee.EnsembleSampler(n_walkers, n_dim, likelihood)#, pool=pool)
sampler.run_mcmc(p0, n_steps, progress=True)

#for p in p0:
#print(p)
#_ = likelihood(p)


np.savez('./gw150914_alpha_test',nwalkers=n_walkers,n_steps=n_steps,chain=sampler.chain)
