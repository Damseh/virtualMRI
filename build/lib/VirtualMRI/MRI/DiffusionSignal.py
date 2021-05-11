#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 18:13:25 2020

@author: rdamseh
"""
import numpy as np
from tqdm import tqdm
from VirtualMRI.MRI.Sequence import Sequence
from VirtualMRI.MRI.DiffusionGradient import DiffusionGradient
import scipy.io as sio
from time import time

class DiffusionSignal:

    def __init__(self,
                binary_image,
                delta_B, 
                T2_image, #  ms
                vx_image, # mm/s
                vy_image, # mm/s
                vz_image, # mm/s
                grad_image=None, # this is to model the propagation 
                vel_image=None, # mm/s
                n_protons=int(1e6),
                n_protons_all=1,
                T1on = False,
                T2on = True,
                TR = 1000.0,
                compute_phase = 0.0,
                compute_A = 0.0,
                dt = 0.2, # ms
                T1 = 1590.0, # ms
                TE = 5.0, # ms
                echo_spacing = 1.0,
                delta_big=2.0,
                delta_small=1.0,
                b_value=10.0,
                phi=45,
                theta=45,
                gamma=2.675e5, # rad/Tesla/msec (gyromagenatic ratio)
                diff_coeff=0.8,
                apply_spin_labeling=False,
                apply_diffusion=True,
                exp_name='MRIexp',
                savepath=''): #Le Bihan 2013 Assumed isotropic; %Proton (water) diffusion coefficient(um2/msec)


        self.binary_image = binary_image
        self.delta_B = delta_B 
        # should be in sec
        self.T2_image = T2_image
        self.vx_image = vx_image
        self.vy_image = vy_image
        self.vz_image = vz_image
        self.vel_image = vel_image
        self.n_protons = n_protons
        self.n_protons_init = n_protons
        self.n_protons_all = n_protons_all
        self.T1on = T1on
        self.T2on = T2on
        self.TR = TR 
        self.compute_phase = compute_phase
        self.compute_A = compute_A 
        self.dt = dt 
        self.T1 = T1
        self.TE = TE
        self.echo_spacing = echo_spacing 
        self.delta_big = delta_big
        self.delta_small = delta_small
        self.b_value = b_value
        self.phi = phi
        self.theta = theta
        self.gamma = gamma
        self.diff_coeff = diff_coeff
        self.apply_spin_labeling = apply_spin_labeling
        self.apply_diffusion=apply_diffusion

        
        self.diff_sigma=np.sqrt(2*self.diff_coeff*self.dt)
        self.map_size=self.T2_image.shape   
        
        #### timing and sequence
        self.sequence=Sequence(Type='DiffSE', 
                               TE=self.TE, 
                               echo_spacing=self.echo_spacing, 
                               delta_big=self.delta_big,
                               delta_small=self.delta_small,
                               dt=self.dt).Sequence 
                               
        self.sequence_toplot=Sequence(Type='DiffSE', 
                               TE=self.TE, 
                               echo_spacing=self.echo_spacing, 
                               delta_big=self.delta_big,
                               delta_small=self.delta_small,
                               dt=.001).Sequence                                
                          
        self.signals=[]
        self.b_values=[]
        self.phi_values=[]
        self.theta_values=[]
        self.mean_phase=[]
        self.name=exp_name
        
        self.shape=np.shape(self.binary_image)
        self.center=np.array(self.shape)/2.0
        self.labels=[]
        
        # get boundary and flow gradients 
        if grad_image is not None:
            
            grad_image_boundary=self.get_gradimage(self.binary_image)
            grad_image_vessel=self.get_gradimage(grad_image)

         
            # normalized boundary gradients
            self.boundary=self.get_norm(grad_image_boundary)>0
            self.grad_image_boundary=self.get_normgrad(grad_image_boundary)

        
            # normalized vessel gradients 
            grad_image_vessel=self.get_normgrad(grad_image_vessel)
            grad_image=[self.grad_image_boundary[i]+grad_image_vessel[i] for i in [0,1,2]]
            self.grad_image=self.get_normgrad(grad_image)
        
        
    def get_gradimage(self, x):
        '''
        compute  image gradients
        '''        
        return(np.gradient(x))
        
    def get_norm(self, gr):
        '''
        computre norm o gradients
        '''
        return(np.sqrt(np.sum(np.array([i**2 for i in gr]), axis=0)))
        
    def get_normgrad(self, gr, boundary=False):
        '''
        compute normal gradients at the boundary of vessels
        to be applied for relecting moving spins if hits vessels wall
        '''
        
        normal=self.get_norm(gr)
        
        if boundary:
            b=normal>0
        
        normal[normal==0]=1
        ret=[i/normal for i in gr]
        
        if boundary:
            return ret, b
        else:
            return ret
        
    
    def InitiateSpins(self):

        if self.apply_spin_labeling:
            self.InitiateSpinsSL()
        else:
            np.random.seed(999)
            self.pos_protons=np.random.rand(int(self.n_protons_init), 3)*np.array(self.map_size) # protons positions
        
        self.n_protons=len(self.pos_protons)
        print('Number of protons: ', self.n_protons)
                
    def InitiateSpinsSL(self):
        
        self.pos_protons=np.empty((0, 3))
        idx=0
        while len(self.pos_protons)<self.n_protons_init:
            np.random.seed(999+10*idx)
            idx+=1
            pos_protons=np.random.rand(int(1e7), 3)*np.array(self.map_size) # protons positions
            ind=tuple([pos_protons[:,i].astype(int) for i in [0,1,2]])
            valid_pos_ind=(self.binary_image[ind]>0)
            pos_protons=pos_protons[valid_pos_ind]

            left=int(self.n_protons_init)-len(self.pos_protons)
            if left>1e7:
                left=int(1e7)
            self.pos_protons=np.vstack((self.pos_protons, pos_protons[0:left])) 
            
        self.pos_protons_copy=self.pos_protons.copy()
        
    def InitiateGradient(self, b_value=None, phi=None, theta=None):
     
        if b_value is not None:
            self.b_value=b_value
            
        if phi is not None:
            self.phi=phi
            
        if theta is not None:
            self.theta=theta
            
        #### Diffusion gradient
        self.DiffusionGradient=DiffusionGradient(shape=self.map_size, 
                             b=self.b_value, 
                             delta_big=self.delta_big, #ms
                             delta_small=self.delta_small, #ms
                             gamma=self.gamma)
        
        self.gradient=self.DiffusionGradient.GetGradient(phi=self.phi, theta=self.theta)
    
        
        print('max delta_B: ', self.delta_B.max())
        print('max gradient: ', self.gradient.max())
    
    
    
    def __UpdateSpinsPos(self, vx, vy, vz, ret=False):
        
        # dt should be in sec
        self.shift=np.array([vx, vy, vz]).T*self.dt*1e-3 # shift vectors 
        valid_pos=np.less_equal(self.pos_protons+self.shift, np.array(self.map_size)) # check if new pos do not exceed space borders
        valid_pos=valid_pos[:,0]*valid_pos[:,1]*valid_pos[:,2]
        self.pos_protons+=self.shift*valid_pos[:,None] # update positions  
                
        if ret:
            return self.shift
   
    
    def __UpdateSpinsPosWithReplacement(self, vx, vy, vz, ret=False):
        
#        def randpos_invessel(im, n):
#            
#            shape=np.array(im.shape)
#            cont =1
#            pos=np.zeros((n,3))
#            ind1=0
#            while cont:
#                p=np.random.rand(n, 3)*shape[None,:]
#                p=p.astype(int)
#                ind=(p[:,0],p[:,1],p[:,2])
#                check=im[ind].ravel()
#                p=p[check>0]
#                l=len(p)    
#                if l+ind1>n:
#                    l=n-ind1
#                pos[ind1:ind1+l]=p[:l]
#                ind1=ind1+l
#                if ind1==n:
#                    cont=0
        # dt should be in sec
        self.shift=np.array([vx, vy, vz]).T*self.dt*1e-3 # shift vectors 
        valid_pos=np.less_equal(self.pos_protons+self.shift, np.array(self.map_size)) # check if new pos do not exceed space borders
        valid_pos=valid_pos[:,0]*valid_pos[:,1]*valid_pos[:,2]
        self.pos_protons+=self.shift*valid_pos[:, None] # update positions  
        
        novalid_pos=np.bitwise_not(valid_pos)
        np.random.shuffle(self.pos_protons_copy)
        self.pos_protons[novalid_pos>0]= self.pos_protons_copy[:sum(novalid_pos>0)]# update positions  

        if ret:
            return self.shift    
    
    def __UpdateSpinsPosNew(self, imageind, ret=False):
        
        # dt should be in sec
        
        self.shift=np.array([self.grad_image[0][imageind],
                        self.grad_image[1][imageind],
                        self.grad_image[2][imageind]]).T*self.vel_image[imageind][:, None]*self.dt*1e-3 # shift vectors 
        
        valid_pos=np.less_equal(self.pos_protons+self.shift, np.array(self.map_size)) # check if new pos do not exceed space borders
        valid_pos=valid_pos[:,0]*valid_pos[:,1]*valid_pos[:,2]
        self.pos_protons+=self.shift*valid_pos[:,None] # update positions  
                
        if ret:
            return self.shift    
    
    def __UpdateSpinsPosOneReflection(self, vx, vy, vz, ret=False):
            
        def Update(shift):
            
            # valid if wihtin image domain
            newpos=self.pos_protons+shift
            valid1=np.less_equal(newpos, np.array(self.map_size)) 
            valid1=valid1[:,0]*valid1[:,1]*valid1[:,2]
            
            # valid if within vessles boundary
            ind=tuple([newpos[valid1][:, i].astype(int) for i in [0,1,2]])
            valid2=self.binary_image[ind]>0 
    
            valid=valid1.copy()
            valid[valid]=valid2  
            
            # protons that need reflectance
            toreflect=valid1.copy()
            toreflect[toreflect]=np.bitwise_not(valid2)
        
            # update valide positions 
            self.pos_protons[valid]+=shift[valid] 
            
            return valid, toreflect
            
        def UpdateReflect(toreflect, newshift):
            
            # apply reflection if within image domain
            newpos=self.pos_protons[toreflect]+newshift
            valid=np.less_equal(newpos, np.array(self.map_size)) 
            valid=valid[:,0]*valid[:,1]*valid[:,2]
            self.pos_protons[toreflect][valid] += newshift[valid]
                                         
            return 
        
        # dt should be in sec
        self.shift=np.array([vx, vy, vz]).T*self.dt*1e-3 # shift vectors 
        valid, toreflectall=Update(self.shift)
        
        self.notreflected=toreflectall.copy()
        
        substeps=10.0
        ds=self.shift/10.0 # sub shifts
        subshift=np.zeros_like(self.shift)
        
        # reflectance if hits vessels boundary
        for i in range(int(substeps)):
            
            subshift[toreflectall]=subshift[toreflectall]+ds[toreflectall] # small increase of shift 
            newpos=self.pos_protons[toreflectall]+subshift[toreflectall]
            
            # check when to reflect (when reaches the boundary)
            idx=tuple([newpos[:,i].astype(int) for i in [0,1,2]])
            val=self.boundary[idx]==1 
            
            #print('Number of spins that hit vessels wall' , len(val)) 
            
            idxx=(idx[0][val], idx[1][val], idx[2][val])
            grad=np.array([self.grad_image_boundary[0][idxx], 
                           self.grad_image_boundary[1][idxx], 
                           self.grad_image_boundary[2][idxx]]).T

            toreflect=toreflectall.copy()
            toreflect[toreflect]=val
    
            # decompose shift vetor into prepend and prarallel to grad_norm vectors            
            w1=(np.sum(self.shift[toreflect]*grad, axis=1)/\
                np.linalg.norm(self.shift[toreflect], axis=1))[:, None]*self.shift[toreflect] # parallel
            w2=self.shift[toreflect]-w1 # prepend
            
            # compute new shift
            ratio=(i+1)/substeps # ratio of the origional 'shift' at this iteration
            origshift=ratio*(w1+w2)
            reflectshift=(1-ratio)*(w1-w2)
            newshift=origshift+reflectshift
            
            # apply
            UpdateReflect(toreflect, newshift)
            
            # update not yet reflected for next iteration
            toreflectall[toreflectall]=np.bitwise_not(val)
        
        
        if ret:
            return self.shift    
        
      
    def __UpdateSpinsPosMultiReflection(self, vx, vy, vz, ret=False):
            
        def Update(shift):
            
            # valid if wihtin image domain
            newpos=self.pos_protons+shift
            valid1=np.less_equal(newpos, np.array(self.map_size)) 
            valid1=valid1[:,0]*valid1[:,1]*valid1[:,2]
            
            # valid if within vessles boundary
            ind=tuple([newpos[valid1][:, i].astype(int) for i in [0,1,2]])
            valid2=self.binary_image[ind]>0 
    
            valid=valid1.copy()
            valid[valid]=valid2  
            
            # protons that need reflectance
            toreflect=valid1.copy()
            toreflect[toreflect]=np.bitwise_not(valid2)
        
            # update valide positions 
            self.pos_protons[valid]+=shift[valid] 
            
            return valid, toreflect
            
        def UpdateReflect(toreflect, newshift):
            
            # apply reflection if within image domain
            newpos=self.pos_protons[toreflect]+newshift
            valid1=np.less_equal(newpos, np.array(self.map_size)) 
            valid1=valid1[:,0]*valid1[:,1]*valid1[:,2]
            
            # check if within boundary
            ind=tuple([newpos[valid1][:, i].astype(int) for i in [0,1,2]])
            valid2=self.binary_image[ind]>0 
            valid=valid1.copy()
            valid[valid]=valid2  
            
            notreflected=valid1.copy()
            notreflected[notreflected]=np.bitwise_not(valid2) 
            
            # if within bounday
            self.pos_protons[toreflect][valid] += newshift[valid]
                 
            #if not: 
            #       1. update shifts to be used in next while iteration
            #       2. update notreflected to be used also in the next while iterartion
            self.shift[toreflect][notreflected]=newshift[notreflected] 
            self.notreflected[toreflect]=np.bitwise_or(self.notreflected[toreflect], notreflected)
            
            return 
     
        # dt should be in sec
        self.shift=np.array([vx, vy, vz]).T*self.dt*1e-3 # shift vectors 
        valid, toreflectall=Update(self.shift)
        
        # this to be used in while loop
        self.notreflected=np.zeros(len(toreflectall)).astype(bool) 
        
        # reflectance if hits vessels boundary
        cont=1
        itr=0
        
        while(cont>0):
        # this while loop to do multiple relections
        
            itr+=1
            substeps=10.0
            ds=self.shift/10.0 # sub shifts
            subshift=np.zeros_like(self.shift)   
            
            for i in range(int(substeps)):
                # the for loop to check when the spin that hits the vessels wall
                
                subshift[toreflectall]=subshift[toreflectall]+ds[toreflectall] # small increase of shift 
                newpos=self.pos_protons[toreflectall]+subshift[toreflectall]
                
                # check when to reflect (when reaches the boundary)
                idx=tuple([newpos[:,i].astype(int) for i in [0,1,2]])
                val=self.boundary[idx]==1 
                
                #print('Number of spins that hit vessels wall' , len(val)) 
                
                idxx=(idx[0][val], idx[1][val], idx[2][val])
                grad=np.array([self.grad_image_boundary[0][idxx], 
                               self.grad_image_boundary[1][idxx], 
                               self.grad_image_boundary[2][idxx]]).T
    
                toreflect=toreflectall.copy()
                toreflect[toreflect]=val
        
                # decompose shift vetor into prepend and prarallel to grad_norm vectors            
                w1=(np.sum(self.shift[toreflect]*grad, axis=1)/\
                    np.linalg.norm(self.shift[toreflect], axis=1))[:, None]*self.shift[toreflect] # parallel
                w2=self.shift[toreflect]-w1 # prepend
                
                # compute new shift
                ratio=(i+1)/substeps # ratio of the origional 'shift' at this iteration
                origshift=ratio*(w1+w2)
                reflectshift=(1-ratio)*(w1-w2)
                newshift=origshift+reflectshift
                
                # apply
                UpdateReflect(toreflect, newshift)
                
                # update not reached boundary for next iteration
                toreflectall[toreflectall]=np.bitwise_not(val)
                
            ###### update not reflected due exceeding boundary in the previous reflection    
            toreflectall=self.notreflected 
            
            #print('Iteration of reflection: ', itr)
            #print('Number of spins to be still reflected: ', np.sum(toreflectall))
            
            if np.sum(toreflectall)==0:
                cont=0
        if ret:
            return self.shift           
            
    
    def __CaptureOrientation2D(self):
        
            '''
            Capture snapshot of the orientation of 
            velocity vectors assigneed to protons.
            This is done by computing the angle between a velocity vector 
            and that emeging from the center of tha image.
            '''                
            posx, posy = [self.pos_protons[:, 0], 
                          self.pos_protons[:, 1]]
            posx=self.center[0]-posx
            posy=self.center[1]-posy
            
            vx, vy= self.shift[:,0], self.shift[:,1]
            
            uv=(posx*vx)+(posy*vy)
            u_v=np.sqrt(posx**2+posy**2)*np.sqrt(vx**2+vy**2)
            
            
            cosang=uv/u_v
            ang=np.arccos(cosang) 
            
            return ang
    
    
    def __CaptureOrientation(self):
        
            '''
            Capture snapshot of the orientation of 
            velocity vectors assigneed to protons.
            This is done by computing the angle between a velocity vector 
            and that emeging from the center of tha image.
            '''                
            posx, posy, posz = [self.pos_protons[:, 0], 
                                self.pos_protons[:, 1],
                                self.pos_protons[:, 2]]
            posx=self.center[0]-posx
            posy=self.center[1]-posy
            posz=self.center[2]-posz
            
            vx, vy, vz= self.shift[:,0], self.shift[:,1], self.shift[:,2]
            
            uv=(posx*vx)+(posy*vy)+(posz*vz)
            
            a1=np.sqrt(posx**2+posy**2+posz**2)
            a2=np.sqrt(vx**2+vy**2+vz**2)
            
            ind=a2!=0
            
            u_v=a1*a2
            
            cosang=uv[ind]/u_v[ind]
            ang=np.arccos(cosang) 
            
            return ang 
        
        
    def GetSignal(self, saveangles=False, 
                  savespin=False,
                  savepath=''):                                 
                                                            
        n_steps=self.sequence.n_steps
        t_steps=self.sequence.t_steps
        dephasing=self.sequence.dephasing
        gradient_on=self.sequence.gradient
    
            
        #### run    
        t0=time()
        print('\nCompute MRI signal ...')
        
        if savespin:
            hf = h5py.File(savepath+'spinspos_'+self.name+'_b'+str(self.b_value)+'.h5', 'w')
            
        phase=np.zeros(self.n_protons) # phase at each time step
        signal=np.zeros(self.n_protons) # signal value at each time step
        mean_phase=[]
        total_signal=[]
    
        print('Number of time steps: %s' %n_steps) 
        for idx, st, sign, g_on in tqdm(zip(range(n_steps), t_steps, 
                                                    dephasing, gradient_on)):
            
            ###### apply diffusion od spins ######
            
            if self.apply_diffusion:
                m=0#self.diff_sigma*np.random.rand(self.n_protons, 3) # movements
                valid_pos=np.less_equal(self.pos_protons+m, np.array(self.map_size)) # check if new pos do not exceed space borders
                self.pos_protons+=m*valid_pos
            
            ###### apply advection (blood flow) ######
            ind=tuple([self.pos_protons[:,i].astype(int) for i in [0,1,2]])
            
            vx=self.vx_image[ind]
            vy=self.vy_image[ind]
            vz=self.vz_image[ind]

            # update positions
            self.__UpdateSpinsPos(vx, vy, vz)
            #self.__UpdateSpinsPosWithReplacement(vx, vy, vz)            #self.__UpdateSpinsPosNew(imageind=ind)
            #self.__UpdateSpinsPosOneReflection(vx, vy, vz)    
            #self.__UpdateSpinsPosMultiReflection(vx, vy, vz)    
            
            if saveangles:
                try:
                    an = self.__CaptureOrientation() 
                    ang.append(an)
                except:
                    ang=[]

            if savespin:
                hf.create_dataset(str(idx), data=self.pos_protons)
                 
            ######## compute phase
            ind=tuple([self.pos_protons[:,i].astype(int) for i in [0,1,2]])
            if self.T2on:
                a = -(self.dt/self.T2_image[ind]) # signal degradation  from T2
            else:
                a = 0
            
            # phase variation from delta_B from berturbed B and gradient field (T2* effect)
            b1 = 1j*self.gamma*(self.delta_B[ind])*self.dt 
            b2 = 1j*self.gamma*g_on*self.gradient[ind]*self.dt
            
            signal=signal+a # signal amplitude for each spin
            phase=phase+sign*(b1+b2); # phase for that spin is already initiated
            mean_phase.append(np.imag(np.mean(phase)))
            
            ######## compute signal
            # dt should be in sec
            if self.T1on:
                s=np.exp(signal)*np.exp(phase)*np.exp(-idx*self.dt/self.T1)
                total_signal.append(np.abs(np.mean(s)))
            else:
                total_signal.append(np.abs(np.mean(np.exp(signal)*np.exp(phase))))
    
    
        # save orientations
        if saveangles:
            sio.savemat(savepath+'orient_t'+str(round(st, 2))+'_'+self.name+'.mat', 
                        {'angles': ang})  
        if savespin:
            hf.close()
    
        print('Time for protons simulation: %s \n' %(time()-t0))
    
        self.signals.append(total_signal)
        self.phi_values.append(self.phi)
        self.theta_values.append(self.theta)
        self.b_values.append(self.b_value)
        self.mean_phase.append(mean_phase)
        
    def SaveExp(self, path):
        
        self.mat={'b': np.array(self.b_values), 'phi':np.array(self.phi_values), 'theta':np.array(self.theta_values),
            't_steps':np.array(self.sequence.t_steps), 'dephasing':np.array(self.sequence.dephasing),
             'gradient': self.sequence.gradient,
             's':np.array(self.signals),
             'TE':self.TE,
             'delta_big':self.delta_big,
             'delta_small':self.delta_small,
             'name':self.name,
             'labels':self.labels}     
        
        sio.savemat(path+self.name+'.mat', self.mat)
        
    def AppendLabel(self, x):
        self.labels.append(x)
        
        
        
        
        