#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 18:14:39 2020

@author: rdamseh

"""
import numpy as np
from numpy import fft as fourier
from math import sin, cos


def BuildT2Image(binary_image, so2_image, B0, T2_tissue):
       
    #T2 volume (in msec)
    T2_image = np.ones_like(binary_image)*T2_tissue;
    
    #T2 formula as a function of deoxyhemoglobin
    x,y,z=np.where(binary_image>0)
    a=2.74*B0-0.6
    b=12.67*(B0**2)*(1-so2_image[(x,y,z)])**2
    T2_image[(x,y,z)] = 1000*(1/(a+b));
    
    return T2_image.astype('float32')


def BuildFFTDeltaChiImage(binary_image, so2_image, hct_image, delta_chi):
    
    susept_image = delta_chi*hct_image*(1-so2_image)   
    
    #mean_delta_chi = np.mean(susept_image[binary_image>0].astype('float32'))
    susept_image_fft=fourier.fftshift(fourier.fftn(fourier.fftshift(susept_image)))

    return susept_image_fft


def BuildFFTKernalPerturbedB(map_size, B1_omega, B1_phi, B0):
    
    voxel_radius=0.5 # radius of each voxel
    c = B0*2/np.pi*voxel_radius**3 # constant


    #degree to rad
    B1_omega*=(np.pi/180)
    B1_phi*=(np.pi/180)
    
    
    # get r vector based on omega and phi
    r = np.array([cos(B1_phi)*sin(B1_omega),\
          sin(B1_phi)*sin(B1_omega),\
          cos(B1_omega)])
    r_norm=np.linalg.norm(r)# equal to one
        
    
    # get set v of vectors
    origin=(np.array(map_size))/2.0
    vx, vy, vz=np.meshgrid(range(map_size[0]), 
                           range(map_size[1]), 
                           range(map_size[2]), indexing='ij')
    
    v=np.array([vx.ravel(), vy.ravel(), vz.ravel()]).T.astype('float32')
    v-=origin
    del vx, vy, vz
    
    v_norm=np.sqrt(np.sum(v**2, axis=1))
    v_norm[v_norm==0]=0.866 # fix to avoid norm=0, which leads to nan values afterward
    
    # compute angle between set v of vectors and r vector
    cos_theta=np.dot(v, r)/(v_norm*r_norm)
   
    # compute PertB
    pert_B=c*(v_norm**-3)*(3*(cos_theta**2)-1) 
    pert_B=np.reshape(pert_B, map_size)

    pert_B_fft=fourier.fftshift(fourier.fftn(fourier.fftshift(pert_B)))

    return pert_B_fft


def ComputeT2andDeltaB(binary_image, 
                       so2_image, 
                       hct_image,
                       B0=7.0,
                       delta_chi=4*np.pi*0.264e-6, # susceptibility of deoxygenated blood
                       B1_omega=90.0,
                       B1_phi=0.0):

    map_size=binary_image.shape
    
    T2_tissue=1000*(1.74*B0+7.77)**(-1), #in msec (from Uludag 2009)

    susept_image_fft = BuildFFTDeltaChiImage(binary_image=binary_image,  
                                           so2_image=so2_image, 
                                           hct_image=hct_image, 
                                           delta_chi=delta_chi)
    del hct_image
    
    
    T2_image=BuildT2Image(binary_image=binary_image, 
                          so2_image=so2_image, 
                          B0=B0, 
                          T2_tissue=T2_tissue)
    
    del binary_image, so2_image
    
    
    
    pert_B_fft = BuildFFTKernalPerturbedB(map_size=map_size, 
                                        B1_omega=B1_omega, 
                                        B1_phi=B1_phi,
                                        B0=B0)
    
    delta_B=np.real(fourier.ifftshift(fourier.ifftn(fourier.ifftshift(pert_B_fft*susept_image_fft))))
    
    return T2_image, delta_B