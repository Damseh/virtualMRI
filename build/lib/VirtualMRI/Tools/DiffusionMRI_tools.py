#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 18:11:51 2020

@author: rdamseh
"""

import numpy as np



import scipy.io as sio
import numpy as np 
from matplotlib import pyplot as plt

class LoadDiffusionExp:
    
    def __init__(self, Exp=None, path=None):
        
        # read mat file file hat containes the mri expirement 
        
        try:
            self.exp=sio.loadmat(path)
        except: pass
                
        try:
            self.b_values=self.exp['b'].ravel()
        except: pass
        
        try:
            self.phi_values = self.exp['phi'].ravel()
        except: pass
    
        try:
            self.theta_values = self.exp['theta'].ravel()
        except: pass
    
        try:
            self.t_steps = self.exp['t_steps'].ravel()
        except: pass
    
        try:
            self.dephasing = self.exp['dephasing'].ravel()
        except: pass
    
        try:
            self.gradient = self.exp['gradient'].ravel()
        except: pass
        
        try:
            self.signals = self.exp['s']
        except: pass
    
        try:
            self.TE = self.exp['TE'].ravel()
        except: pass
    
        try:
            self.delta_big  = self.exp['delta_big']
        except: pass
    
        try:
            self.delta_small = self.exp['delta_small']
        except: pass

        try:
            self.name = str(self.exp['name'])
        except: self.name='MRIexp'

        self.readout=np.where(np.array(self.t_steps).astype(int)==int(self.TE*2.0))[0][0]

        del self.exp
        
        
    def PlotSignals(self, show_gradient=True, savepath=None, legend=True):
        
        plt.figure()
        plt.title(self.name+'_signals_profile')
        
        if show_gradient:
            
            plt.plot(self.t_steps, self.dephasing)
            plt.plot(self.t_steps, self.gradient) 
            plt.yticks(ticks=np.arange(-1,1.1,.1))
            
        for i, j in zip(self.signals, zip(self.b_values, self.phi_values, self.theta_values)):
            
            l='b='+str(j[0])+', $\phi$='+str(j[1])+', $\\theta$='+str(j[2])
            plt.plot(self.t_steps.ravel(), i, label=l)
            
        if legend:
            plt.legend()
        
        if savepath is not None:
            plt.savefig(savepath+self.name+'_signals_profile.png')
            
            
            
    def PlotSignals_Ratio(self, to='phi0', show_gradient=True, savepath=None):
        
        if to=='phi0':
            
            ## ratio to signa with phi=0 ########
            
            # index where phi = 0
            phi0ind=np.where(self.phi_values==0)[0]
            if len(phi0ind)>1:
                phi0ind=phi0ind[0]
                
            # index where b =0
            b0ind=np.where(self.b_values==0)[0]
            if len(b0ind)>1:
                b0ind=b0ind[0]                
            
            # get signals with b!=0 and phi !=0
            phi0=self.signals[phi0ind]
            s=[a for ind, a in enumerate(self.signals) if ind!=phi0ind and ind!=b0ind]
            b_values=[a for ind, a in enumerate(self.b_values) if ind!=phi0ind and ind!=b0ind]
            theta_values=[a for ind, a in enumerate(self.theta_values) if ind!=phi0ind and ind!=b0ind]
            phi_values=[a for ind, a in enumerate(self.phi_values) if ind!=phi0ind and ind!=b0ind]
    
            ratios=[(np.array(a)/np.array(phi0)).ravel() for a in s]
            
            plt.figure()
            plt.title(self.name+'_ratio_signals_profile_to_phi0')        
            
            if show_gradient:
                
                plt.plot(self.t_steps, self.dephasing)
                plt.plot(self.t_steps, self.gradient) 
                
            for i, j in zip(ratios, zip(b_values, phi_values, theta_values)):
                
                l='b='+str(j[0])+', $\phi$='+str(j[1])+', $\\theta$='+str(j[2])
                plt.plot(self.t_steps, i, label=l)
            
            plt.legend()
            if savepath is not None:
                plt.savefig(savepath+self.name+'_ratio_phi0.png')
        
        else:
            
            ## ratio to signal with b=0 ########
            
            # index where b =0
            b0ind=np.where(self.b_values==0)[0]
            if len(b0ind)>1:
                b0ind=b0ind[0]
                       
            # get signals with b!=0 
            b0=self.signals[b0ind]
            s=[a for ind, a in enumerate(self.signals) if ind!=b0ind]
            b_values=[a for ind, a in enumerate(self.b_values) if ind!=b0ind]
            theta_values=[a for ind, a in enumerate(self.theta_values) if ind!=b0ind]
            phi_values=[a for ind, a in enumerate(self.phi_values) if ind!=b0ind]
    
            ratios=[(np.array(a)/np.array(b0)).ravel() for a in s]
            
            plt.figure()
            plt.title(self.name+'_ratio_signals_profile_to_b=0')        
            
            if show_gradient:
                
                plt.plot(self.t_steps, self.dephasing)
                plt.plot(self.t_steps, self.gradient) 
                
            for i, j in zip(ratios, zip(b_values, phi_values, theta_values)):
                
                l='b='+str(j[0])+', $\phi$='+str(j[1])+', $\\theta$='+str(j[2])
                plt.plot(self.t_steps, i, label=l)
            
            plt.legend()
            if savepath is not None:
                plt.savefig(savepath+self.name+'_ratio_b0.png')  
    
    def ReadoutRatiosPhi0(self):
    
        # index where phi = 0
        phi0ind=np.where(self.phi_values==0)[0]
        if len(phi0ind)>1:
            phi0ind=phi0ind[0]
            
        # index where b =0
        b0ind=np.where(self.b_values==0)[0]
        if len(b0ind)>1:
            b0ind=b0ind[0]                
        
        # get signals with b!=0 and phi !=0
        phi0=self.signals[phi0ind]
        s=[a for ind, a in enumerate(self.signals) if ind!=phi0ind and ind!=b0ind]
        b_values=[a for ind, a in enumerate(self.b_values) if ind!=phi0ind and ind!=b0ind]
        theta_values=[a for ind, a in enumerate(self.theta_values) if ind!=phi0ind and ind!=b0ind]
        phi_values=[a for ind, a in enumerate(self.phi_values) if ind!=phi0ind and ind!=b0ind]

        ratios=[(np.array(a)/np.array(phi0)).ravel() for a in s]  
        r=[i[self.readout] for i in ratios]
        
        return r
        
      
    def ReadoutRatio_phi0_to_b0(self):
    
        # index where phi = 0
        phi0ind=np.where(self.phi_values==0)[0]
        if len(phi0ind)>1:
            phi0ind=phi0ind[0]
            
        # index where b =0
        b0ind=np.where(self.b_values==0)[0]
        if len(b0ind)>1:
            b0ind=b0ind[0]                
        
        # get signals with b!=0 and phi !=0
        phi0=self.signals[phi0ind][0]
        b0=self.signals[b0ind][0]
        
        r=phi0[self.readout]/b0[self.readout]
        
        return r  


def get_marker(Exp):
    
    '''
    Exp is a class of DiffusionSignal: VirtualMRI.MRI.DiffusionSignal
    
    max(s/s0)-min(s/s0)
    '''

    s=get_s_readout(Exp)
    s0=s[0]
    s_s0=[k/s0 for k in s[1:]]

    # biomarker
    marker=max(s_s0)-min(s_s0)

    return marker

def get_marker2(Exp):
    
    '''
    Exp is a class of DiffusionSignal: VirtualMRI.MRI.DiffusionSignal
    
    max(s/s0)-min(s/s0)
    '''

    s=get_s_readout(Exp)
    s0=s[0]
    s_s0=[s0-k for k in s[1:]]

    # biomarker
    marker=min(s_s0)

    return marker


def get_s_readout(Exp):
    
    '''
    Exp is a class of DiffusionSignal: VirtualMRI.MRI.DiffusionSignal
    '''
    
    # extract signals at differet gradients
    signals = Exp.signals
    labels = Exp.labels
    labels=[str(k) for k in labels]
    t_steps=Exp.sequence.t_steps
    TE=Exp.TE
    
    # s/s0 at read out 
    ind=np.where(t_steps.astype(int)==int(TE*2))[0][0]
    s=[s[ind] for s in signals] # signal value at read out
    
    return s


def plot_signals_profile(Exp, with_markers=False, grid=True, legend=False, linewidth=3):
    
    '''
    Exp is a class of DiffusionSignal: VirtualMRI.MRI.DiffusionSignal
    '''
    
    # extract signals at differet gradients
    t=Exp.sequence.t_steps
    gr=Exp.sequence.gradient
    labels=Exp.labels
    
    if with_markers:
        plot_markers=['','o','v','^','<','>',
                      '1','2','3','4','8','s','p',
                      'P','*','h','H','+',
                      'x','X','d','D']
    else:
        plot_markers=['' for i in range(len(labels))]
    
    fig, axx = plt.subplots(1,2)
    axx[0].plot(t, gr, linewidth=linewidth, marker='', markevery=5, markersize=15)
    ss=Exp.signals
    for s, l, k in zip(ss, labels, plot_markers):
        axx[0].plot(t, s,  marker=k, markevery=5, linewidth=linewidth, markersize=15, label='$\\theta$1='+str(l[1])+', $\\theta$2='+str(l[2]))    
    if legend:
        axx[0].legend()
    if grid:
        axx[0].grid()

    axx[1].plot(t, gr, linewidth=linewidth, marker='', markevery=5, markersize=15)
    ss=np.array(Exp.signals)
    ss=ss/ss[0][None,:]
    for s, l, k in zip(ss, labels , plot_markers):
        axx[1].plot(t, s, marker=k, linewidth=linewidth, markevery=5, markersize=15, label='$\\theta$1='+str(l[1])+', $\\theta$2='+str(l[2]))   
    if legend:
        axx[1].legend()
    if grid:
        axx[1].grid()        
    
    
    return s







