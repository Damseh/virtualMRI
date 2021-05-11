#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 18:14:39 2020

@author: rdamseh
"""
import numpy as np

class Sequence:
    
    def __init__(self,
                Type='FSE',
                T1 = 1590.0, #ms
                TE = 2.0, #ms
                echo_spacing = 1.0, #ms,
                dt=.1, #ms
                delta_big=.5, #ms
                delta_small=.25, #ms
                echo_number = None):

        self.Type=Type
        self.T1=T1
        self.TE=TE
        self.echo_spacing=echo_spacing
        self.dt=dt
        self.delta_big=delta_big 
        self.delta_small=delta_small
        
        if echo_number is None :
            self.echo_number = [1, TE*echo_spacing*4] 
        else:
            self.echo_number=echo_number

        
        self.UpdateSequence(Type=self.Type)
        
      
    def check_deltas(self):
        
        '''
        thsi is used when returning diffusion mri sequence
        '''
                
        T180=self.TE
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
                self.Sequence=self.DiffSE2(self.TE,
                                    self.dt,
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
    
    
    
    class DiffSE2:
        
        ''' 
       This class to create objects for diffusion spin echo sequence that includes the foloowing attributes:
           - t-steps: time steps in ms
           - repahsing: rephasing direction (right(+), left(-)) for a given MRI sequence 
           - gradient: one-zero signal to indicate the status of the gradient field (on-off) 
       
       '''
        def __init__(self, 
                    TE,
                    dt,
                    start1,
                    end1,
                    start2,
                    end2):
            
                
            self.n_steps = int(np.ceil(TE*5/dt)) # time steps
            self.t_steps=np.linspace(0, TE*5, self.n_steps)
            self.dephasing=self.dephasing_sign(self.n_steps, TE)
            self.gradient=self.gradient_on_off(self.n_steps, TE, start1, end1, start2, end2)

        def gradient_on_off(self, n_steps, TE, start1, end1, start2, end2):
            
            t_steps=np.linspace(0, TE*5, self.n_steps)
            gg=np.zeros_like(t_steps)
            gg[(t_steps >start1)&(t_steps <end1)]=1
            gg[(t_steps >start2)&(t_steps <end2)]=1
            
            return gg
            
        def dephasing_sign(self, n_steps, TE):
            
            t_steps=np.linspace(0, TE*5, self.n_steps)
            dephasing=t_steps
            dephasing[dephasing<(TE)]=1
            dephasing[dephasing>(TE)]=-1
            
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
                
            self.n_steps = int(np.ceil((TE+np.sum(echo_number))/dt)) # time steps
            self.t_steps=np.linspace(0, TE+np.sum(echo_number), self.n_steps)
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