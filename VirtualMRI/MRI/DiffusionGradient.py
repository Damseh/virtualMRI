#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 12:53:15 2020

@author: rdamseh
"""

import numpy as np 
import scipy.integrate as integrate

class DiffusionGradient:
    
    def __init__(self, shape=(10,10,10), 
                 b=10, #s/mm2
                 delta_big=.5, #ms
                 delta_small=.25, #ms
                 gamma=2.675e5): #rad/Tesla/ms)):
        
        self.shape=shape
        self.b=b
        self.delta_big=delta_big
        self.delta_small=delta_small
        self.gamma=gamma
        
        r=int(np.array(self.shape).min()/2.0)
        self.r=r

        self.CalculateGradMag()

    def CalcSphericalVoxel(self):
    
        r=self.r
        r1,r2,r3=np.array(self.shape)/2.0
        
        xrange, yrange, zrange = np.meshgrid(np.arange(-r2, r2, 1),
                                         np.arange(-r1, r1, 1),
                                         np.arange(-r3, r3, 1))
       
        sphere = (xrange**2 + yrange**2 + zrange**2) < (r**2)
        
        return sphere
                
    def GetGradientImage(self, phi=45, theta=45):
        
        '''
        This creates a gradint in the image space
        Inputs:
            shape: 3D image size
            phi: angle in degree between the gradient direction and the z axis
            theta: angle in degree between gradient direction (projected in the x-y plane) and the x axis
        '''
        
        ##### coordinates
        #            y   ^ z
        #           \   /
        #           \ /
        #    --------------> x
        #          /\
        #         / \
        #           v
        ###########
    
        angles=[phi, theta]
        angles=(np.array(angles)/180.0)*np.pi
        
        dx=np.sin(angles[0])*np.cos(angles[1])
        dy=np.sin(angles[0])*np.sin(angles[1])
        dz=np.cos(angles[0])
        
        self.length=abs(self.shape[0]*dx) + abs(self.shape[1]*dy) + abs(self.shape[2]*dz)
        
        
        x=np.ones(self.shape[0]+1)*dx
        x=integrate.cumtrapz(x)-1
        x=np.repeat(x[:, None], repeats=self.shape[1], axis=1)
        
        y=np.ones(self.shape[1]+1)*dy
        y=integrate.cumtrapz(y)-1
        y=np.repeat(y[None, :], repeats=self.shape[0], axis=0)
        
        xy=x+y
        xy=np.repeat(xy[:, :, None], repeats=self.shape[2], axis=2)
        
        
        z=np.ones(self.shape[2]+1)*dz
        z=integrate.cumtrapz(z)-1
        z=np.repeat(z[None, :], repeats=self.shape[1], axis=0)
        z=np.repeat(z[None, :, :], repeats=self.shape[0], axis=0)
    
        im = z+xy
        im=(im-im.min())/(im.max()-im.min())
        self.Grad=im


    def CalculateGradMag(self): 
    
        b=self.b*1000.0 #from s/mm2 ----> ms/mm2
        
        a1=self.delta_big-(self.delta_small/3.0)
        a2=self.gamma**2*self.delta_small**2
        
        g=b/(a1*a2)
        g=np.sqrt(g) #T/mm
        g=g/1000.0 #T/um
        
        self.GradVAl=g
    
    def GetGradient(self, phi, theta):
        
        self.GetGradientImage(phi=phi, theta=theta)  
        g=self.Grad*self.GradVAl*self.length
        return g 


        