#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 14:55:00 2020

@author: rdamseh
"""

import numpy as np
from tqdm import tqdm 

class CreateCylinderMappings:
    
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
        self.tags={'binary':1, 
                   'velocity':0, 
                   'so2':0, 
                   'hct':0, 
                   'gradient':0, 
                   'propagation':0}
        
        for i in to_return:
            self.tags[i]=1
        
    def GetImSize(self):
        
        # shift graph geometry to start from zero coordinates
        # and
        # set min radius to 2.0

        min_rad=2.0
        pos=np.array(self.g.GetNodesPos())
        pos=pos-np.min(pos, axis=0)[None, :]
        
        rad=np.array(self.g.GetRadii())
        rad[rad<min_rad]=min_rad
        
        maxr=np.max(rad)

        for i, p, r in zip(self.g.GetNodes(), pos, rad):
            self.g.node[i]['pos']=p+maxr
            self.g.node[i]['r']=r      
        
        # get image size to be constructed
        real_s = np.max(pos, axis=0) # real image size
        new_s=real_s
        
        
        new_s=tuple((np.ceil(new_s+(2*maxr))).astype(int)) # image size after padding
        
        print('Image size: '+str(new_s)) 

        self.real_s = real_s
        self.new_s = new_s
        self.niter = self.g.number_of_edges()

    def cylinder(self, direction, radius, length):
        
        '''
        Create a image cylinder  
        '''
        
        r=length+2*radius
        r=int(r)
        
        #print('r value', r)
        xrange, yrange, zrange = np.meshgrid(np.arange(-r, r+1),
                                             np.arange(-r, r+1),
                                             np.arange(-r, r+1), indexing='ij')
        size=np.shape(xrange)
        
        direction=direction.astype(float)
        va=np.sqrt((direction**2).sum())
        vnorm=direction/va
        
        p=np.array([xrange.ravel(), yrange.ravel(), zrange.ravel()]).T
        p=p.astype(float)
        amp=np.sqrt(np.sum(p**2, axis=1))
        amp[amp<1]=1
        
        cos=np.abs(np.sum(p*vnorm, axis=1)/amp)
        cos[cos>1]=1
        sin=np.sqrt(1-cos**2)
    
        shape0=(amp*sin)<radius # radius constrain
        shape1=(amp*cos<length) # length constrain
        
        a1=amp*cos-length
        a2=amp*sin
        shape2=(((a1**2+a2**2)**0.5)<(radius)) # rounded end constrain
        
        shape=shape0*(shape2+shape1)
        
        shape=np.reshape(shape, xrange.shape)
        c0 = np.where(shape)
        
        dot=np.sum(p*vnorm, axis=1)
        dot=((dot-dot.min())/(dot.max()-dot.min()))
        shape=shape*dot.reshape(shape.shape)
        
        return c0, size   


    def get_cylinder_infos(self, g, radius_scaling=None):
        
        info=dict()
        
        if self.tags['binary']:

            e=g.GetEdges()
            pos1=np.array([g.node[i[0]]['pos'] for i in e])
            pos2=np.array([g.node[i[1]]['pos'] for i in e])
            
            radius1=np.array([g.node[i[0]]['r'] for i in e])
            radius2=np.array([g.node[i[1]]['r'] for i in e]) 
            radius=(radius1+radius2)/2.0# radius
            
            if radius_scaling is not None:
                radius*=radius_scaling
                
            info['pos1']=pos1
            info['pos2']=pos2
            info['radius']=radius
            
            vec=pos2-pos1
            vec_amp=np.sqrt(np.sum(vec**2, axis=1))# norm
            vec_amp[vec_amp==0]=1.0 # avoid divide by zero
            vec_norm=vec/vec_amp[:, None]
            
            # for edges of length < 2 set to length to 3 to avoid diconnedted maps
            vec_amp[vec_amp<2.0]=2.0

            
            info['vec_amp']=vec_amp
            info['vec_norm']=vec_norm
            
            
        if self.tags['so2']:
            
            so21=np.array([g.node[i[0]]['so2'] for i in e])
            so22=np.array([g.node[i[1]]['so2'] for i in e]) 
            info['so21']=so21
            info['so22']=so22     
        
        if self.tags['hct']:
            types=np.array([g.node[i[0]]['type'] for i in e])
            if types.max()==3: types-=1 # types should be 0-->Art., 1-->Vein, 2-->Capp
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
                 radius_scaling=None,
                 hct_values=[0.33, 0.44, 0.44]):
        '''
        
        Input:
            resolution: This in the number of points interplated at each graph edge
            radius_scaling: This factor used to increase/decrease the overll radius size
            hct_values: A list in the format [hct_in_arteriols, hct_in_venules, hct_in_cappilaries]        
        '''
        
        info = self.get_cylinder_infos(self.g, radius_scaling=radius_scaling)
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
            
                # direction of this segment
                vec_amp, vec_norm = info['vec_amp'][idx], info['vec_norm'][idx]
                
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
        

            c, shape  = self.cylinder(vec_norm, radius=r, length=vec_amp)
            
            x0, y0, z0 = c[0], c[1], c[2]       
                                
            pos=(p1+p2)/2.0
            
            # this to align to middle of the cylinder
            sub=np.array(shape)/2.0
            x = x0-sub[0]
            y = y0-sub[1]
            z = z0-sub[2]
            
            # this to align in the middle of the edge
            x=x+pos[0]
            y=y+pos[1]
            z=z+pos[2]
            
            ss=s1
            ll=l1
            
            c=(x.astype(int), y.astype(int), z.astype(int))
            
            if np.size(np.array(c))==0:
                print('Zero length edge: '+str(idx))
                continue

                
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
                grad_image[c]=ll
    
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



