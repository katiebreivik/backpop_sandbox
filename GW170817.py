import numpy as np
import emcee

from ctypes import *
from scipy.stats import gaussian_kde
from multiprocessing import Pool

import os
from scipy.stats import multivariate_normal

libc = cdll.LoadLibrary("./evolv2_merger.so")
np.set_printoptions(suppress=True)

def evolv2(m1, m2, tb, e, alpha, vk1, theta1, phi1, omega1, vk2, theta2, phi2, omega2, acc_lim, qc_kstar2, qc_kstar3, Z):
    m1 = m1
    m2 = m2
    
    m2, m1 = np.sort([m1,m2],axis=0)
    tb = 10**tb
    metallicity = 10**Z
    sigma = 0

    z = byref(c_double(metallicity))
    zpars = np.zeros(20).ctypes.data_as(POINTER(c_double))
    alpha = byref(c_double(alpha))
    acc_lim = byref(c_double(acc_lim))
    q3 = byref(c_double(qc_kstar2))
    sigma = byref(c_double(0.0))
    q4 = byref(c_double(qc_kstar3))
    natal_kick = np.zeros((2,5))
    natal_kick[0,0] = vk1
    natal_kick[0,1] = phi1
    natal_kick[0,2] = theta1
    natal_kick[0,3] = omega1
    natal_kick[0,4] = 3
    natal_kick[1,0] = vk2
    natal_kick[1,1] = phi2
    natal_kick[1,2] = theta2
    natal_kick[1,3] = omega2
    natal_kick[1,4] = 3
    natal_kick = natal_kick.T.flatten().ctypes.data_as(POINTER(c_double))
    libc.evolv2_global_(z,zpars,alpha,acc_lim,q3,q4,natal_kick)

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
    
    bpp = bpp_out._arr.reshape(43,1000)[:,0:bpp_index_out._obj.value].T
    return t_merge._obj.value,np.sort(m_merge)[::-1],kick_info_out._arr.reshape(17,2).T,bpp

#Porb: 185.63 +/- 0.05
#Eccentricity: 0.45  +/- 0.01 
#MBH: 9.80  +/- 0.20
#Mstar: 0.93 +/- 0.05
mean = np.array([1.46, 1.27])
cov = np.array([[0.12**2, 0,], [0, 0.09**2]])
rv = multivariate_normal(mean, cov)
#m1, m2, tb, e, alpha, vk1, theta1, phi1, omega1, vk2, theta2, phi2, omega2, acc_lim, qc_kstar2, qc_kstar3, Z
m1lo = 5.0
m2lo = 5.0
tblo = 5.0
elo = 0.0
alphalo = 0.1
vklo = 0.0
thetalo = 0.0
philo = -90.0
omegalo = 0.0
acc_limlo = 0.0
qc_kstar2lo = 0.5
qc_kstar3lo = 0.5
Zlo = 0.0001

m1hi = 60.0
m2hi = 60.0
tbhi = 5000.0
ehi = 0.9
alphahi = 20.0
vkhi = 300.0
thetahi = 360.0
phihi = 90.0
omegahi = 360
acc_limhi = 1.0
qc_kstar2hi = 10.0
qc_kstar3hi = 10.0
Zhi = 0.03

lower_bound = np.array([m1lo, m2lo, np.log10(tblo), elo, alphalo, vklo, thetalo, philo, omegalo, vklo, thetalo, philo, omegalo, acc_limlo, qc_kstar2lo, qc_kstar3lo, np.log10(Zlo)])
upper_bound = np.array([m1hi, m2hi, np.log10(tbhi), ehi, alphahi, vkhi, thetahi, phihi, omegahi, vkhi, thetahi, phihi, omegahi, acc_limhi, qc_kstar2hi, qc_kstar3hi, np.log10(Zhi)])

def likelihood(coord):
    for i in range(len(coord)):
        if (coord[i]<lower_bound[i] or coord[i]>upper_bound[i]):
            return -np.inf
    result = evolv2(*coord)
    gw_coord = np.array([result[1][0],result[1][1]])
    return rv.logpdf(gw_coord) 


#result = evolv2(25.0, 8.0, 2500.0, 0.5, 1.0, 30.0, thetahi, phihi, omegahi, vkhi, thetahi, phihi, omegahi, acc_limhi, 
#                qc_kstar2hi, qc_kstar3hi, np.log10(Zhi))
#
#print(np.round(result[3][:,[0,1,2,3,4,6,10]], 2))
#

n_dim = len(lower_bound)
n_walkers = 1024
p0 = np.random.uniform(lower_bound, upper_bound, size=(n_walkers, len(lower_bound)))
p0 = p0*np.random.normal(1.0,0.01,size=(n_walkers,n_dim))
n_steps = 100

#with Pool() as pool:

sampler = emcee.EnsembleSampler(n_walkers, n_dim, likelihood)#, pool=pool)
sampler.run_mcmc(p0, n_steps, progress=True)

np.savez('./gw170817_test',nwalkers=n_walkers,n_steps=n_steps,chain=sampler.chain[:,::10])
