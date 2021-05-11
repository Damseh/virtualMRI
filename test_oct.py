#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 11:52:26 2020

@author: rdamseh
"""


import VirtualMRI as vmri
from time import time
import numpy as np 
import configparser
import ast
from VascGraph.GraphIO import ReadPajek
import VascGraph as vg
from mayavi import mlab
import networkx as nx

import os
import scipy.stats as stat

from matplotlib import pyplot as plt

def get_signal(g, 
            B0=None, 
            TE=None, 
            delta_small=None, 
            delta_big=None, 
            ASL=None, 
            b=None):
    
    '''
    extract diffusion spin echo signals
    '''
    
    configpath='config.txt'
    exp=vmri.MRI.DiffusionExp(configpath=configpath, n_protons=1e5)
    
    if not B0 is None:
        B0=str(B0)
        exp.config.set('MRI','B0', B0)
    
    if not TE is None:
        TE=str(TE) 
        exp.config.set('MRI','TE', TE)
        
    if not delta_small is None:
        delta_small=str(delta_small) 
        exp.config.set('MRI','delta_small', delta_small)
   
    if not delta_big is None:
        delta_big=str(delta_big) 
        exp.config.set('MRI','delta_big', delta_big)
        
    if not ASL is None:
        if ASL:
            ASL='yes'
        else:
            ASL='no'        
        exp.config.set('MRI','with_spin_labeling', ASL)
        
    if not b is None:
        b_values='0, '+str(b) 
        exp.config.set('Gradients','b_values', b_values)
        
    exp.Run(g)
    
    return exp.Exp

def test_vis(g):
    
    gplot=vg.GraphLab.GraphPlot(new_engine=True)
    gplot.Update(g)
    gplot.SetGylphSize(.05)
    gplot.SetTubeRadius(1)
    gplot.SetTubeRadiusByScale(True)
    gplot.SetTubeRadiusByColor(True)

def preprocess_graph(g, scale=0.5):
    
    '''
    preprocessing of oct vascular graphs
    '''
    
    print('--Refining radius...')
    g.RefineRadiusOnSegments(rad_mode='median')    
    rad=np.array(g.GetRadii())
    rad[np.isnan(rad)]=2.0
    for i, r in zip(g.GetNodes(), rad):
        g.node[i]['r']=r
    
    # flip and fix minus node positions
    pos=np.array(g.GetNodesPos())
    minp = np.min(pos, axis=0)
    minp[minp<0]=minp[minp<0]*-1
    minp[minp<1]=minp[minp<1]+1
    for i, p in zip(g.GetNodes(), pos):
        g.node[i]['pos']=np.array([p[1],p[2],p[0]])+minp
    
    # scaling the domain
    pos=np.array(g.GetNodesPos())
    pos=pos*scale
    g.SetNodesPos(pos)
    
    #crop # sphere # isotropic
    pos=np.array(g.GetNodesPos())
    maxp = np.max(pos, axis=0)
    remove=[]
    for i, p in zip(g.GetNodes(), pos):
        if p[2]>maxp[2]*0.66:
            remove.append(i)  
    remove=list(set(remove))
    g.remove_nodes_from(remove)
    g=vg.Tools.CalcTools.fixG(g)    

    return g

if __name__=='__main__':
    
    '''
    compute diffusion spin echo response based on oct vascular graphs in 
    'data_oct'
    
    coefficients of the mri expirement are extracted fom 'config.txt'; however, new values
    can be inserted in the code 
    '''
    
    path='data_oct/'
    animal='K3/'
        
    B0=7
    b=500

    animalpath=path+animal
    animalgraph=os.listdir(animalpath)[0]
                                
    
    # read vascular graph
    graphpath=path+animal+animalgraph
    print('--Reading graph from: '+graphpath)
    g=ReadPajek(graphpath, mode='di').GetOutput()        
        
    # preprocess
    g=preprocess_graph(g)
    
    # assign oxygen quantities
    print('--Assign PO2/SO2...')
    oxyg=vmri.Graphing.OxyGraph(g)
    oxyg.Update()
    g=oxyg.GetOuput()
    g=vg.Tools.CalcTools.fixG(g)
                    
    #mri
    print('--Simulate MRI ...')
    '''
    this simulates the diffusion mri response(s) based on different values of 
    b, phi and theta. Check the 'config.txt' file
    '''
    exp = get_signal(g, B0=B0, b=b)
                    
    
    # labels of the set of signals generated [b, phi, theta]
    labels=exp.labels
            
    # output signals
    signals=exp.signals

    # plot    
    from matplotlib import pyplot as plt
    plt.figure()
    gr=exp.sequence.gradient
    t=exp.sequence.t_steps   
    plt.plot(t, gr, label='Gradiant (on/off)')
    ss=exp.signals
    for s, l in zip(ss, labels):
        plt.plot(t, s, label='b='+str(l[0])+', phi='+str(l[1])+', theta='+str(l[2])) 
    plt.xlabel('Time (ms)')
    plt.ylabel('Signal loss')
    plt.title('Diffusion signal')
    plt.legend()
    plt.show()             
                
            
            
            
            
        
    