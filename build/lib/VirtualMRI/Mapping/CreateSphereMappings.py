#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 14:55:00 2020

@author: rdamseh
"""

import numpy as np
from tqdm import tqdm 

class CreateSphereMappings:
    
    '''
    
    This class create 3D maps based on oriented cylinders built at each graph edge 
    
    '''
        
    
    def __init__(self, g, to_return=['binary', 
                                     'velocity', 
                                     'so2', 
                                     'hct', 
                                     'gradient', 
                                     'propagation']):
        
        self.g=g
        self.GetImSize()
        
        # set the needed outputs
        self.tags={'binary':0, 
                   'velocity':0, 
                   'so2':0, 
                   'hct':0, 
                   'gradient':0, 
                   'propagation':0}
        
        for i in to_return:
            self.tags[i]=1



    def GetImSize(self):
        
        pos=np.array(self.g.GetNodesPos())
        pos=pos-np.min(pos, axis=0)[None, :]
        
        for i, p in zip(self.g.GetNodes(), pos):
            self.g.node[i]['pos']=p
        
        real_s = np.max(pos, axis=0) # real image size
        new_s=real_s
        
        maxr=np.max(self.g.GetRadii())
        
        new_s=tuple((np.ceil(new_s+(2*maxr+1))).astype(int)) # image size after padding
        
        print('Image size: '+str(new_s)) 

        self.real_s = real_s
        self.new_s = new_s
        self.niter = self.g.number_of_edges()


    def get_sphere_infos(self, g, radius_scaling=None):
        
        info=dict()
        
        if self.tags['binary']:

            e=g.GetEdges()
            pos1=np.array([g.node[i[0]]['pos'] for i in e])
            pos2=np.array([g.node[i[1]]['pos'] for i in e])
            
            radius1=np.array([g.node[i[0]]['d'] for i in e])
            radius2=np.array([g.node[i[1]]['d'] for i in e]) 
            radius=(radius1+radius2)/2.0# diameter
            radius*=0.5 # diameter to radius
            
            if radius_scaling is not None:
                radius*=radius_scaling
                
            info['pos1']=pos1
            info['pos2']=pos2
            info['radius']=radius
            
            
        if self.tags['so2']:
            
            so21=np.array([g.node[i[0]]['so2'] for i in e])
            so22=np.array([g.node[i[1]]['so2'] for i in e]) 
            info['so21']=so21
            info['so22']=so22     
        
        if self.tags['hct']:
            types=np.array([g.node[i[0]]['type'] for i in e])
            info['types']=types    
        
        if self.tags['velocity']:
        
            velocity=np.array([g.node[i[0]]['velocity'] for i in e])
            dx=np.array([g.node[i[0]]['dx'] for i in e])
            dy=np.array([g.node[i[0]]['dy'] for i in e])
            dz=np.array([g.node[i[0]]['dz'] for i in e])
            
            info['velocity']=velocity
            info['dx']=dx
            info['dy']=dy
            info['dz']=dz
        
        
        if self.tags['propagation']:
            try:
                label1=np.array([g.node[i[0]]['label'] for i in e])
                label2=np.array([g.node[i[1]]['label'] for i in e])
            
                info['label1']=label1
                info['label2']=label2   
            except:
                print('--Cannot return \'propagation\'; no labels on input graph!')   
                self.tags['propagation']=False
        return info
    

    def GetOutput(self, 
                 resolution=1.0, 
                 radius_scaling=None,
                 hct_values=[0.44, 0.33, 0.44]):
        '''
        
        Input:
            resolution: This in the number of points interplated at each graph edge
            radius_scaling: This factor used to increase/decrease the overll radius size
            hct_values: A list in the format [hct_in_arteriols, hct_in_cappilaries, hct_in_venules]        
        '''
        
        info = self.get_sphere_infos(self.g, radius_scaling=radius_scaling)
        real_s, new_s, niter = self.real_s, self.new_s, self.niter

        if self.tags['binary']:
            binary_image=np.zeros(new_s) 
        
        if self.tags['so2']:
            so2_image = np.zeros(new_s) 
        
        if self.tags['hct']:
            hct_values=np.array(hct_values)
            info['hct']=hct_values[info['types']]
            hct_image = np.zeros(new_s) 
        
        if self.tags['velocity']:
            vx_image = np.zeros(new_s) 
            vy_image = np.zeros(new_s) 
            vz_image = np.zeros(new_s) 
            vel_image = np.zeros(new_s)  # velocity
        
        if self.tags['gradient']:
            grad_image = np.zeros(new_s)  # gradient image to capture propagation across graph
        
            
        for idx in tqdm(range(niter)):
            
            if self.tags['binary']:
                p1, p2, r = info['pos1'][idx], info['pos2'][idx], info['radius'][idx]            
            
            if self.tags['so2']:
                s1, s2 = info['so21'][idx], info['so22'][idx]
            else:
                s1, s2 = 0, 0
            
            if self.tags['hct']:
                h = info['hct'][idx]
            else:
                h = 0
                
            if self.tags['velocity']:
                velo = info['velocity'][idx]
            else:
                velo = 0
               
            if self.tags['propagation']:
                l1, l2 = info['label1'], info['label2']
            else:
                l1, l2 = 0, 0
                        
                
                
            xrange, yrange, zrange = np.meshgrid(np.arange(-r, r+1, .5),
                                                         np.arange(-r, r+1, .5),
                                                         np.arange(-r, r+1, .5))
                                                 
            
            # direction of this segment
            vec=p2-p1
            vec_amp=np.sqrt((vec**2).sum())# norm
            interpolate=vec_amp/resolution
            vec_norm=vec/vec_amp
            
            # create smooth gradient on the sphere
            dot=xrange*vec_norm[0]+yrange*vec_norm[1]+zrange*vec_norm[2]
            dot=((dot-dot.min())/(dot.max()-dot.min()))
    
            
            sphere = ((xrange**2 + yrange**2 + zrange**2) <= r**2)*dot
            x0, y0, z0 = np.where(sphere)
    
            
            if interpolate<1:
                interpolate=1
                
            p=(p2-p1)/interpolate # (sphere position) to fill gaps between graph nodes
            s=(s2-s1)/interpolate # interpolate so2 values between two graph nodes
            l=(l2-l1)/interpolate # interpolate level of propagation
            
            x = x0+p1[0]
            y = y0+p1[1]
            z = z0+p1[2]
            
            ss=s1
            ll=l1
            
            c0=(np.round(x0).astype(int), np.round(y0).astype(int), np.round(z0).astype(int))
            sphere=sphere*l
            
            
            for i in range(int(interpolate)):
                     
                
                    c=(np.round(x).astype(int), np.round(y).astype(int), np.round(z).astype(int))
                    
                    if self.tags['binary']:
                        binary_image[c]=1
                    
                    if self.tags['so2']:
                        so2_image[c]=ss
                    
                    if self.tags['hct']:
                        hct_image[c]=h
                    
                    if self.tags['velocity']:
                        vx_image[c]=velo*vec_norm[0]
                        vy_image[c]=velo*vec_norm[1]
                        vz_image[c]=velo*vec_norm[2]
                        vel_image[c]=velo
                    
                    if self.tags['gradient']:
                        grad_image[c]=ll+sphere[c0]
                    
                    x = x + p[0]
                    y = y + p[1]
                    z = z + p[2]
                    ss += s
                    ll += l
    
        real_s=real_s.astype(int)
        ends=((new_s-real_s)/2.0).astype(int)
    
        ret=dict()
    
        # cropping to recover origional image size
        if self.tags['binary']:
            binary_image=binary_image[ends[0]:-ends[0], 
                                      ends[1]:-ends[1], 
                                      ends[2]:-ends[2]]
            ret['binary'] = binary_image.astype(int)
            
        if self.tags['so2']:
            so2_image=so2_image[ends[0]:-ends[0], 
                                ends[1]:-ends[1], 
                                ends[2]:-ends[2]]
            ret['so2'] = so2_image.astype('float32')
            
        if self.tags['hct']:
            hct_image=hct_image[ends[0]:-ends[0], 
                                ends[1]:-ends[1], 
                                ends[2]:-ends[2]]    
            ret['hct'] = hct_image.astype('float32')
        
        if self.tags['velocity']:
            
            vx_image=vx_image[ends[0]:-ends[0], 
                                ends[1]:-ends[1], 
                                ends[2]:-ends[2]]  
            ret['vx'] = vx_image.astype('float32')
            vy_image=vy_image[ends[0]:-ends[0], 
                                ends[1]:-ends[1], 
                                ends[2]:-ends[2]]   
            ret['vy'] = vy_image.astype('float32')
            vz_image=vz_image[ends[0]:-ends[0], 
                                ends[1]:-ends[1], 
                                ends[2]:-ends[2]]   
            ret['vz'] = vz_image.astype('float32')
            vel_image=vel_image[ends[0]:-ends[0], 
                                ends[1]:-ends[1], 
                                ends[2]:-ends[2]]  
            ret['velocity'] = vel_image.astype('float32')
            
        if self.tags['gradient']:
            grad_image=grad_image[ends[0]:-ends[0], 
                                ends[1]:-ends[1], 
                                ends[2]:-ends[2]]            
            ret['gradient']= grad_image.astype('float32')
            
            
        return ret








