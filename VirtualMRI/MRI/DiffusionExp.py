#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 18:11:51 2020

@author: rdamseh
"""


import VirtualMRI as vmri
from time import time
import numpy as np 
import configparser
import ast

class MyConfig:
    
    def __init__(self):
        self.config={}
    
    def add_section(self, name):
        self.config[name]=dict()
    
    def add_value(self, section, valname, val):
        self.config[section][valname]=val
    
    def set(self, section, valname, val):
        self.config[section][valname]=val 
        
    def get(self, section, valname):
        return str(self.config[section][valname])
    
    def getboolean(self, section, valname):
        if type(self.config[section][valname])==str:
            if self.config[section][valname]=='yes' or self.config[section][valname]=='true':
                return True
            else:
                return False
        else:
            return bool(self.config[section][valname])        
   
    def getfloat(self, section, valname):
        return float(self.config[section][valname])    
   
    def getint(self, section, valname):
        return int(self.config[section][valname])



class DiffusionExp:
    
    def __init__(self, configpath=None, 
                 radius_scaling=0.5, 
                 hct_values=[0.44, 0.33, 0.44],
                 B0=7,
                 with_T1=True,
                 with_T2=True,
                 with_spin_labeling=True,
                 with_diffusion=True,
                 d=0.8,
                 TE=5.0 ,
                 T1=1590.0,
                 delta_small=1.0, 
                 delta_big=4.0, 
                 dt=0.5,
                 echo_spacing=1.0,
                 b_values=[0, 20],
                 phi = [30, 60, 90],
                 theta = [0, 30,  60,  90,  135,  175],
                 n_protons=1e5):
        
        self.n_protons=n_protons
        
        if configpath is None:
            
            self.config=MyConfig()
            self.config.add_section('MRI')
            self.config.add_section('Gradients')
            self.config.add_section('Mapping') 
            self.config.add_value('Mapping', 'radius_scaling', radius_scaling)
            self.config.add_value('Mapping', 'hct_values', hct_values)
            self.config.add_value('MRI', 'B0', B0)
            self.config.add_value('MRI', 'with_T1', with_T1)
            self.config.add_value('MRI', 'with_T2', with_T2)
            self.config.add_value('MRI', 'with_spin_labeling', with_spin_labeling)
            self.config.add_value('MRI', 'with_diffusion', with_diffusion)
            self.config.add_value('MRI', 'd', d)
            self.config.add_value('MRI', 'TE', TE)
            self.config.add_value('MRI', 'T1', T1)
            self.config.add_value('MRI', 'delta_small', delta_small)
            self.config.add_value('MRI', 'delta_big', delta_big)
            self.config.add_value('MRI', 'dt', dt)
            self.config.add_value('MRI', 'echo_spacing', echo_spacing)
            self.config.add_value('Gradients', 'b_values', b_values)
            self.config.add_value('Gradients', 'phi', phi) 
            self.config.add_value('Gradients', 'theta', theta)      
        
        else:
            self.config=configparser.ConfigParser()
            self.config.read(configpath, encoding='utf-8-sig')        

    def Run(self, g, name='MRIexp'):
        
        '''
        each node in g should have the following attributes:
            position: 'pos' -->[x,y,z] (um)
            radius: 'r' (um)
            type: 'type' --> 1 for arter., 2 for vein, 3 for capp.
            Partial pressure of oxygen: 'po2' mmHg
            velocity: 'velocity' (um/s)
            flow: 'flow' mm3/s
            Oxygen saturarion: 'so2' 
            flow unit direction (x component): 'dx' (um) --> this simple to get from the directed edge in the graph 
            flow unit direction (y component): 'dy' (um)
            flow unit direction (z component): 'dz' (um)

        example:
        >> g.node[i]
        {'pos': array([ 32.06107622, 100.90446505, 109.54899617]),
         'r': 3.0,
         'type': 3,
         'po2': 44.319120092354204,
         'velocity': 1866.5747605007025,
         'flow': 5.277615799468621e-05,
         'so2': 0.5628292413921282,
         'dx': -0.002575490000001679,
         'dy': 0.21470156999998835,
         'dz': -0.3561572099999921}
            
        '''
                     
        ###################################### 
        ##### create 3d maps for anatomy, so2, hct and velocity #####
        t0=time()
        print('\nCreate anatomical, SatO2, Htc and velocity maps ...')
        mapping=vmri.Mapping.CreateCylinderMappings(g, to_return=['binary', 
                                                         'velocity', 
                                                         'so2', 
                                                         'hct', 
                                                         'gradient', 
                                                         'propagation'])
        
        maps = mapping.GetOutput(radius_scaling=self.config.getfloat('Mapping','radius_scaling'),
                                 hct_values=ast.literal_eval(self.config.get('Mapping','hct_values')))
        
        
        print('Time : %s' %(time()-t0))
        vd=np.sum(maps['binary'])/np.sum(np.ones_like(maps['binary']))*100.0
        print('vascular density = %s%s' %(np.round(vd, decimals=4), ' %'))
        
        ###################################### 
        ##### compute 3d maps of T2 paramter and delta B ####
        t0=time()
        print('\nCreate T2 and delta B maps ...')
        T2_image, delta_B = vmri.Tools.MRI_tools.ComputeT2andDeltaB(binary_image=maps['binary'], 
                                               so2_image=maps['so2'],
                                               hct_image=maps['hct'],
                                               B0=self.config.getfloat('MRI','B0'))
        del maps['so2'], maps['hct']
        print('Time : %s' %(time()-t0))   
                 
        ###### get MRI signal #######
        MRI_expirement = vmri.MRI.DiffusionSignal(binary_image=maps['binary'],
                                         delta_B=delta_B, 
                                         T2_image=T2_image, # ms
                                         vx_image=maps['vx'], # um/s
                                         vy_image=maps['vy'], # um/s
                                         vz_image=maps['vz'], # um/s
                                         dt=self.config.getfloat('MRI','dt'), # ms
                                         TE =self.config.getfloat('MRI','TE'), # ms
                                         T1=self.config.getfloat('MRI','T1'), # *1e-3 s
                                         T1on=self.config.getboolean('MRI','with_T1'),
                                         T2on=self.config.getboolean('MRI','with_T2'),
                                         echo_spacing=self.config.getfloat('MRI','echo_spacing'),
                                         delta_big=self.config.getfloat('MRI','delta_big'),
                                         delta_small=self.config.getfloat('MRI','delta_small'),
                                         diff_coeff=self.config.getfloat('MRI','d'),
                                         n_protons=self.n_protons,
                                         apply_diffusion=self.config.getboolean('MRI','with_diffusion'),
                                         apply_spin_labeling=self.config.getboolean('MRI','with_spin_labeling'))
    
        ##### genrate signal attenuation based on different b_values and gradient orientations
        phi=ast.literal_eval(self.config.get('Gradients','phi'))
        theta=ast.literal_eval(self.config.get('Gradients','theta'))
        phi_theta=np.meshgrid(phi,theta)
        phi_theta=np.array([phi_theta[0].T.ravel(), phi_theta[1].T.ravel()]).T
        phi_theta=np.vstack(([0, 90], phi_theta))
        
        for b_value in ast.literal_eval(self.config.get('Gradients','b_values')): 
            if b_value==0:
                MRI_expirement.InitiateSpins()
                MRI_expirement.InitiateGradient(b_value=b_value)
                MRI_expirement.GetSignal(saveangles=False,
                                         savespin=False,
                                         savepath='') 
                MRI_expirement.AppendLabel([0,0,0])
            else:
                for i in phi_theta:
                    MRI_expirement.InitiateSpins()
                    MRI_expirement.InitiateGradient(b_value=b_value,
                                                    phi=i[0],
                                                    theta=i[1])
                    print('--Get signal at theta1 = '+str(i[0])+', theta2 = '+str(i[1]))
                    MRI_expirement.GetSignal() 
                    MRI_expirement.AppendLabel([b_value, i[0], i[1]])                    
                    
        self.Exp=MRI_expirement
        
                
    def SaveExp(self):
        pass
                    