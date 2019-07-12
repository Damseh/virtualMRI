#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 13:50:23 2019

@author: rdamseh
"""



import util as util
import SynthAngio1 as synth
from time import time
from VascGraph.Tools.VisTools import visVolume, visG
import skimage.io as skio
from matplotlib import pyplot as plt
import numpy as np

import scipy as sp
from tqdm import tqdm


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
        x=sp.integrate.cumtrapz(x)-1
        x=np.repeat(x[:, None], repeats=self.shape[1], axis=1)
        
        y=np.ones(self.shape[1]+1)*dy
        y=sp.integrate.cumtrapz(y)-1
        y=np.repeat(y[None, :], repeats=self.shape[0], axis=0)
        
        xy=x+y
        xy=np.repeat(xy[:, :, None], repeats=self.shape[2], axis=2)
        
        
        z=np.ones(self.shape[2]+1)*dz
        z=sp.integrate.cumtrapz(z)-1
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


class MRISequence:
    
    def __init__(self,
                Type='FSE',
                T1 = 1590.0, #ms
                TE = 2.0, #ms
                echo_spacing = 1.0, #ms,
                dt=.1, #ms
                delta_big=.5, #ms
                delta_small=.25, #ms
                echo_number = [6, 12]):

        self.Type=Type
        self.T1=T1
        self.TE=TE
        self.echo_spacing=echo_spacing
        self.dt=dt
        self.delta_big=delta_big 
        self.delta_small=delta_small
        self.echo_number=echo_number
        
        self.UpdateSequence(Type=self.Type)
        
        
    def check_deltas(self):
        
        echo_number=self.echo_number
        try: 
            len(echo_number)
        except:
            echo_number=[int(echo_number/4.0), int(3*echo_number/4.0)]
                
        T180=self.TE+echo_number[0]*self.echo_spacing
        end1= T180 - ((self.delta_big-self.delta_small)/2.0)
        start2= T180 + ((self.delta_big-self.delta_small)/2.0)
        start1= end1-self.delta_small
        end2=start2+self.delta_small
        
        if start1>0:
            
            self.end1= end1
            self.start2=start2
            self.start1= start1
            self.end2=end2
            
            return True
        
        else:
            
            return False
        
    def UpdateSequence(self, Type=''):
       
        ''' 
       This Get the t-steps with rephasing direction (right(+), left(-)) for a given MRI sequence 
       Input:
           Type: string for the name of the sequence
           delta_big: used in case of DiffSE (diffusion spin echo) sequence
           delta_small: used in case of DiffSE (diffusion spin echo) sequence
        ''' 
       
        if Type=='GESFIDE':
            self.Sequence=self.GESFIDE(self.TE,
                                self.echo_spacing,
                                self.dt,
                                self.echo_number)
        elif Type=='FSE':
            self.Sequence=self.FSE(self.TE,
                                self.echo_spacing,
                                self.dt,
                                self.echo_number)
            
        elif Type=='DiffSE':
            
            check=self.check_deltas()
            if not check:
                print('values of delta_big and delta_small are not valid!')
                return
            else:
                self.Sequence=self.DiffSE(self.TE,
                                    self.echo_spacing,
                                    self.dt,
                                    self.echo_number,
                                    self.start1,
                                    self.end1,
                                    self.start2,
                                    self.end2)            
        else:
            self.Sequence=self.FID(self.TE,
                                self.echo_spacing,
                                self.dt,
                                self.echo_number)            
             
    class FID:
        
        def __init__(self,
                    TE, 
                    echo_spacing,
                    dt,
                    echo_number):
            
            echo_number=np.sum(echo_number)
            self.n_steps = int(np.ceil((TE+ echo_spacing*echo_number)/dt)) # time steps
            self.t_steps=np.linspace(0, TE+ echo_spacing*echo_number, self.n_steps)
    
    
    class FSE:
        
        def __init__(self,
                    TE, 
                    echo_spacing,
                    dt,
                    echo_number):

        
            echo_number=np.sum(echo_number)
            self.n_steps = int(np.ceil((TE+ echo_spacing*echo_number)/dt)) # time steps
            self.t_steps=np.linspace(0, TE+ echo_spacing*echo_number, self.n_steps)
            self.dephasing=self.dephasing_sign(self.n_steps, TE, echo_spacing, echo_number)


        def pluses_train(self, t_steps, TE, echo_spacing):
               
            add_step=TE+(echo_spacing/2.0)
            echo_half=(echo_spacing/2.0)
            output=np.array(t_steps-add_step)
            output[output<0]=0
            output = (output*.5)%echo_spacing < echo_half
            
            return output
        
        def dephasing_sign(self, n_steps, TE, echo_spacing, echo_number):

            echo_half=echo_spacing/2.0
            t_steps=np.linspace(0, (TE+ echo_spacing*echo_number), n_steps)
            dephasing=self.pluses_train(t_steps, TE, echo_spacing)
            chck=self.pluses_train(TE+echo_half, TE, echo_spacing)
            dephasing[t_steps<(TE/2.0)]=chck
            dephasing[(t_steps>(TE/2.0))&(t_steps<(TE+echo_half))]=chck-1
            dephasing=dephasing*2-1 # binary to 1, -1
            if dephasing[0]==-1:
                dephasing*=-1
                
            return dephasing    


    class GESFIDE:
        
        def __init__(self,
                    TE, 
                    echo_spacing,
                    dt,
                    echo_number):
       
            try: 
                len(echo_number)
            except:
                echo_number=[int(echo_number/4.0), int(3*echo_number/4.0)]
                
            self.n_steps = int(np.ceil((TE+echo_spacing*np.sum(echo_number))/dt)) # time steps
            self.t_steps=np.linspace(0, TE+echo_spacing*np.sum(echo_number), self.n_steps)
            self.dephasing=self.dephasing_sign(self.n_steps, TE, echo_spacing, echo_number)

        def dephasing_sign(self, n_steps, TE, echo_spacing, echo_number):
            
            t_steps=np.linspace(0, (TE+ echo_spacing*np.sum(echo_number)), n_steps)
            dephasing=t_steps
            dephasing[dephasing<(TE+echo_spacing*echo_number[0])]=1
            dephasing[dephasing>(TE+echo_spacing*echo_number[0])]=-1
            
            return dephasing    
    
    class DiffSE:
        
        ''' 
       This class to create objects for diffusion spin echo sequence that includes the foloowing attributes:
           - t-steps: time steps in ms
           - repahsing: rephasing direction (right(+), left(-)) for a given MRI sequence 
           - gradient: one-zero signal to indicate the status of the gradient field (on-off) 
       
       '''
        def __init__(self, 
                    TE,
                    echo_spacing,
                    dt,
                    echo_number,
                    start1,
                    end1,
                    start2,
                    end2):
            
            try: 
                len(echo_number)
            except:
                echo_number=[int(echo_number/4.0), int(3*echo_number/4.0)]
                
            self.n_steps = int(np.ceil((TE+echo_spacing*np.sum(echo_number))/dt)) # time steps
            self.t_steps=np.linspace(0, TE+echo_spacing*np.sum(echo_number), self.n_steps)
            self.dephasing=self.dephasing_sign(self.n_steps, TE, echo_spacing, echo_number)
            self.gradient=self.gradient_on_off(self.n_steps, TE, echo_spacing, echo_number,
                                          start1, end1, start2, end2)

        def gradient_on_off(self, n_steps, TE, echo_spacing, echo_number,
                             start1, end1, start2, end2):
            
            t_steps=np.linspace(0, (TE+ echo_spacing*np.sum(echo_number)), n_steps)
            gg=np.zeros_like(t_steps)
            gg[(t_steps >start1)&(t_steps <end1)]=1
            gg[(t_steps >start2)&(t_steps <end2)]=1
            
            return gg
            
        def dephasing_sign(self, n_steps, TE, echo_spacing, echo_number):
            
            t_steps=np.linspace(0, (TE+ echo_spacing*np.sum(echo_number)), n_steps)
            dephasing=t_steps
            dephasing[dephasing<(TE+echo_spacing*echo_number[0])]=1
            dephasing[dephasing>(TE+echo_spacing*echo_number[0])]=-1
            
            return dephasing                
            

class MRISignalDiffusion:

    def __init__(self,
                binary_image,
                delta_B, 
                T2_image,
                vx_image, 
                vy_image, 
                vz_image,
                n_protons=int(1e6),
                n_protons_all=1,
                T1on = 0.0,
                TR = 1000.0,
                compute_phase = 0.0,
                compute_A = 0.0,
                dt = 0.2, #ms
                T1 = 1590.0, #ms
                TE = 2.0, #ms
                echo_spacing = 1.0,
                echo_number = [12,24],
                delta_big=2.0,
                delta_small=1.0,
                b_value=10.0,
                phi=45,
                theta=45,
                gamma=2.675e5, # rad/Tesla/msec (gyromagenatic ratio)
                diff_coeff=0.8,
                apply_spin_labeling=False,
                apply_diffusion=True): #Le Bihan 2013 Assumed isotropic; %Proton (water) diffusion coefficient(um2/msec)


        self.binary_image = binary_image
        self.delta_B = delta_B 
        self.T2_image = T2_image
        self.vx_image = vx_image
        self.vy_image = vy_image
        self.vz_image = vz_image
        self.n_protons = n_protons
        self.n_protons_init = n_protons
        self.n_protons_all = n_protons_all
        self.T1on = T1on
        self.TR = TR 
        self.compute_phase = compute_phase
        self.compute_A = compute_A 
        self.dt = dt 
        self.T1 = T1
        self.TE = TE
        self.echo_spacing = echo_spacing 
        self.echo_number = echo_number 
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
        self.sequence=MRISequence(Type='DiffSE', TE=TE, 
                      echo_number=echo_number, 
                      echo_spacing=echo_spacing, 
                      delta_big=delta_big,
                      delta_small=delta_small,
                      dt=dt).Sequence 
                          
        self.signals=[]
        self.mean_phase=[]
        
    def InitiateSpins(self):

                    
        if self.apply_spin_labeling:
            self.InitiateSpinsSL()
        else:
            np.random.seed(999)
            self.pos_protons=np.random.rand(self.n_protons_init,3)*np.array(self.map_size) # protons positions
        
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
    
    def GetSignal(self):                                 
                                                            
        n_steps=self.sequence.n_steps
        dephasing=self.sequence.dephasing
        gradient_on=self.sequence.gradient
    
        #### run    
        t0=time()
        print('\nCompute MRI signal ...')
    
            
        phase=np.zeros(self.n_protons) # phase at each time step
        signal=np.zeros(self.n_protons) # signal value at each time step
        mean_phase=[]
        total_signal=[]
    
        print('Number of time steps: %s' %n_steps) 
        for idx, sign, g_on in tqdm(zip(range(n_steps), dephasing, gradient_on)):
            
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
            m2=np.array([vx, vy, vz]).T*self.dt
            valid_pos=np.less_equal(self.pos_protons+m2, np.array(self.map_size)) # check if new pos do not exceed space borders
            self.pos_protons+=m2*valid_pos
            
            ######## compute phase
            ind=tuple([self.pos_protons[:,i].astype(int) for i in [0,1,2]])
            a = 0#-(self.dt/self.T2_image[ind]) # signal degradation  from T2
                
            
            # phase variation from delta_B from berturbed B and gradient field (T2* effect)
            b1 = 1j*self.gamma*(self.delta_B[ind])*self.dt 
            b2 = 1j*self.gamma*g_on*self.gradient[ind]*self.dt 
            
            signal=signal+a # signal amplitude for each spin
            phase=phase+sign*(b1+b2); # phase for that spin is already initiated
            mean_phase.append(np.imag(np.mean(phase)))
            
            ######## compute signal
            if self.T1:
                s=np.exp(signal)*np.exp(phase)*np.exp(-idx*self.dt/self.T1)
                total_signal.append(np.abs(np.mean(s)))
            else:
                total_signal.append(np.abs(np.mean(np.exp(signal)*np.exp(phase))))
    
        print('Time for protons simulation: %s \n' %(time()-t0))
    
        self.signals.append(total_signal)
        self.mean_phase.append(mean_phase)
            


if __name__=='__main__':
    

    synth_graph='normal'
    
    if synth_graph=='sphere':    
    
        ############ get spherical PO2 graph ###########
        s=synth.SynthVN(type=synth_graph,                     
                  length=[128],
                  n_vessels=[230],
                  vessel_radii=[4.0])
        g=s.GetCircularVN(sphere_radii=[28.0])
    
    elif synth_graph=='normal':
    
        ############ get noraml PO2 graph ###########
        s=synth.SynthVN(type=synth_graph,                     
                  length=[128],
                  n_vessels=[150],
                  vessel_radii=[4.0])
        g=s.GetNormalVN(x=352,y=356,z=228)

    elif synth_graph=='cylinder':
        ############ get cylindrical like PO2 graph ###########
        s=synth.SynthVN(type=synth_graph,
                  length=[128],
                  n_vessels=[240],
                  vessel_radii=[4.0])
        g=s.GetCylinderVN(sphere_radii=[50.0], hight=50.0)
    else:
        ############ get noraml PO2 graph ###########
        s=synth.SynthVN(type=synth_graph,                     
                  length=[128],
                  n_vessels=[150],
                  vessel_radii=[4.0])
        g=s.GetOrientedVN( x=312, y=312, z=312,
                    orientation=[45, 90])
    visG(g)
        
    ###################################### 
    ##### create 3d maps for anatomy, so2, hct and velocity #####
    t0=time()
    print('\nCreate anatomical, SatO2, Htc and velocity maps ...')
    binary_image, so2_image, hct_image, vx_image, vy_image, vz_image= util.CreateMappings(g, 
                                                                        interpolate=10.0,
                                                                        radius_scaling=0.5,
                                                                        hct_values=[0.44, 0.33, 0.44])
    
    print('Time : %s' %(time()-t0))
    
    vd=np.sum(binary_image)/np.sum(np.ones_like(binary_image))*100.0
    print('vascular density = %s%s' %(np.round(vd, decimals=4), ' %'))
    
    ###################################### 
    ##### compute 3d maps of T2 paramter and delta B ####
    t0=time()
    print('\nCreate T2 and delta B maps ...')
    T2_image, delta_B = util.ComputeT2andDeltaB(binary_image=binary_image, 
                                           so2_image=so2_image,
                                           hct_image=hct_image)
    del so2_image, hct_image
    print('Time : %s' %(time()-t0))   

    
    apply_spin_labeling=True
    apply_diffusion=True
    
    
    d=0.8
    
    
    TE=5.0
    dt=.5
    echo_spacing=1.0
    echo_number=[1, TE*echo_spacing*4] 
    delta_big=3.0
    delta_small=1.5
    TE_half=TE+echo_number[0]*echo_spacing
    TE_full=TE_half*2.0
    
                  
    ###### get MRI signal #######
    exp_name=synth_graph
    MRI_expirement = MRISignalDiffusion(binary_image=binary_image,
                                     delta_B=delta_B, 
                                     T2_image=T2_image,
                                     vx_image=vx_image,
                                     vy_image=vy_image,
                                     vz_image=vz_image,
                                     dt=dt,
                                     TE = TE, #ms
                                     T1=False,
                                     echo_spacing=echo_spacing,
                                     echo_number=echo_number,
                                     delta_big=delta_big,
                                     delta_small=delta_small,
                                     diff_coeff=d,
                                     n_protons=int(1e6),
                                     apply_diffusion=apply_diffusion,
                                     apply_spin_labeling=apply_spin_labeling,)
    
    ##### genrate signal attenuation based on different b_values and gradient orientations
        
    b_values=np.linspace(0, 10, 2)
    phi=[0, 45, 90]#np.linspace(0, 90, 3)
    theta=[45]#np.linspace(0, 360, 4)
    phi_theta=np.meshgrid(phi,theta)
    phi_theta=np.array([phi_theta[0].T.ravel(), phi_theta[1].T.ravel()]).T
    
    for b_value in b_values: 
        if b_value==0:
            MRI_expirement.InitiateSpins()
            MRI_expirement.InitiateGradient(b_value=b_value)
            MRI_expirement.GetSignal()  
        else:
            for i in phi_theta:
                MRI_expirement.InitiateSpins()
                MRI_expirement.InitiateGradient(b_value=b_value,
                                                phi=i[0],
                                                theta=i[1])
                MRI_expirement.GetSignal() 

    t_steps=MRI_expirement.sequence.t_steps
    
    # signals at all time steps
    s0_all=MRI_expirement.signals[0]
    signals_b1=MRI_expirement.signals[1:]
    
    plt.figure()
    plt.title(exp_name+': Signal')
    for i, j in zip(signals_b1, phi_theta): 
        plt.plot(t_steps, i, label=j)
    plt.legend()        
    plt.plot(t_steps, MRI_expirement.sequence.dephasing)
    plt.plot(t_steps, MRI_expirement.sequence.gradient) 
    plt.yticks(ticks=np.arange(-1,1.1,.1))


    sp.io.savemat('MyResults/'+exp_name+'_s0_all.mat', {'s0_all':np.array(s0_all)})
    sp.io.savemat('MyResults/'+exp_name+'_signals_b10.mat', {'signals_b10':np.array(signals_b1)})
    sp.io.savemat('MyResults/'+exp_name+'_t_steps.mat', {'t_steps':np.array(t_steps)})
    sp.io.savemat('MyResults/'+exp_name+'_phi_theta.mat', {'phi_theta':np.array(phi_theta)})

    s0_all=sp.io.loadmat('MyResults/'+exp_name+'_s0_all.mat')['s0_all'][0]
    signals_b1=sp.io.loadmat('MyResults/'+exp_name+'_signals_b10.mat')['signals_b10']

        
    # signals value at read out (spin echo)
    readout=np.where(np.array(t_steps).astype(int)==TE_full)[0]
    if len(readout>1): readout=readout[0]
    
    s0=s0_all[readout]
    s_b1=[i[readout] for i in signals_b1] # echo at various gradient directions when b1
    
    adc_b1=-np.log(np.array(s_b1)/s0)/b_values[1] 
    adc_b1_diff2=np.diff(np.diff(adc_b1))
    
    # s/s0
    plt.figure()
    plt.title(exp_name+': S/S0')
    plt.plot(np.array(s_b1)/s0)
    
    # appearant diffusion 
    plt.figure()
    plt.title(exp_name+': ADC')
    plt.plot(adc_b1)
    
    # appearant diffusion 
    plt.figure()
    plt.title(exp_name+': ADC/mean')
    plt.plot(adc_b1/adc_b1.mean())
    
    # second derivative of aperant diffusion
    plt.figure()
    plt.title(exp_name+': Second derivative of ADC')
    plt.plot(adc_b1_diff2)
#    
#
#        
        
    '''  
    g=DiffusionGradient(shape=(100,100,100), delta_big=3.0, delta_small=1.0)
    
    phi=[45]#np.linspace(0, 90, 3)
    theta=[0,45,90,135,180,225,270,360]#np.linspace(0, 360, 4)
    phi_theta=np.meshgrid(phi,theta)
    phi_theta=np.array([phi_theta[0].ravel(), phi_theta[1].ravel()]).T
    
    for i in phi_theta:     
        g_=g.GetGradient(i[0],i[1])
        print(i, g.length, '------',g_.max())   
        
    plt.imshow(g_[:,0,:])
        
    '''
        
        