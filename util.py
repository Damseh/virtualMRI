#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 13:28:13 2019

@author: rdamseh
"""


import numpy as np
from scipy.linalg import norm 
from scipy.spatial.transform import Rotation as R
import networkx as nx
from VascGraph.Tools import ExtraTools as et

from VascGraph.GeomGraph.DiGraph import DiGraph
from VascGraph.GeomGraph.Graph import Graph

from VascGraph.Tools.VisTools import visG, visStack, visVolume
from VascGraph.Tools.CalcTools import fixG, numpy_fill

import VascGraph as vg
import scipy as sp

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from mayavi import mlab
from numpy.linalg import inv
from tqdm import tqdm
from numba import jit
from time import time

import skimage.io as skio
from numpy import fft as fourier

from math import sin, cos

import SynthAngio as synth

class ImageFromGraph():
    
    def __init__(self, graph):
        self.graph=graph
  
def BuildO2GraphFromMat(folder, scaling=False):
    
    alpha=1.27e-15 # Bunsen solubility coefficient


    m=sp.io.loadmat(folder+'mesh.mat')
    m=m['im2'][0,0]
    
    p=m['nodePos_um']
    e=m['nodeEdges']
    diam=m['nodeDiam'].T
    
    seg_types=np.squeeze(m['segVesType'])
    seg_number=np.squeeze(m['nodeSegN'])-1
    types=seg_types[seg_number]-1
        
    if scaling:
        scale=m['Hvox'][0]
    else:
        scale=1.0


    # read o2 content per mml
    ref=sp.io.loadmat(folder+'ref.mat')
    o2_content=ref['cg1']
    
    # get po2 
    po2=o2_content/alpha
    
    g=DiGraph()
    nodes=range(1, len(p)+1)
    g.add_nodes_from(nodes)
    for i, j, d, k, t in zip(g.GetNodes(), p, diam, po2, types):
        
        g.node[i]['pos']=j/scale
        g.node[i]['d']=float(d)
        g.node[i]['po2']=float(k)
        g.node[i]['type']=int(t)
    
    g.add_edges_from(e)
    g=fixG(g, copy=False)
    return g
    
def RefineGraphDiam(g, n=5):
    
    def neighborhood(g, node, n):
        path_edges = nx.bfs_edges(g1, node, depth_limit=n)
        return [e[1] for e in path_edges]
    
    g1=g.copy()
    g1=g1.to_undirected()
    nbrs=[neighborhood(g1, i, n) for i in g1.GetNodes()]
    diam=[[g1.node[i]['d'] for i in j] for j in nbrs]
    
    
    #diam, mask=numpy_fill(diam)
    diam_new=[np.mean(i) for i in diam]
    
    for i, j in zip(g.GetNodes(), diam_new):
        g.node[i]['d']=j
    
    return g


def RefineGraphAttribute(g, attr='d', n=5):
    
    def neighborhood(g, node, n):
        path_edges = nx.bfs_edges(g1, node, depth_limit=n)
        return [e[1] for e in path_edges]
    
    g1=g.copy()
    g1=g1.to_undirected()
    nbrs=[neighborhood(g1, i, n) for i in g1.GetNodes()]
    diam=[[g1.node[i][attr] for i in j] for j in nbrs]
    
    
    #diam, mask=numpy_fill(diam)
    diam_new=[np.mean(i) for i in diam]
    
    for i, j in zip(g.GetNodes(), diam_new):
        g.node[i][attr]=j
    
    return g


def UpSampleGraph(g):
        
    
    # nodes
    e=g.GetEdges()
    p_new=np.array([(g.node[i[0]]['pos']+g.node[i[1]]['pos'])/2.0 for i in e])
    r_new=np.array([(g.node[i[0]]['d']+g.node[i[1]]['d'])/2.0 for i in e])
    po2_new=np.array([(g.node[i[0]]['po2']+g.node[i[1]]['po2'])/2.0 for i in e])
    
    nodes_new=range(len(g), len(g)+len(p_new))
    g.add_nodes_from(nodes_new)
    for i,j,k,l in zip(nodes_new, p_new, r_new, po2_new):
        g.node[i]['pos']=j
        g.node[i]['d']=k
        g.node[i]['po2']=l


    #edges
    e1_new=[(i,j[0]) for i,j in zip(nodes_new, e)]
    e2_new=[(i,j[1]) for i,j in zip(nodes_new, e)]
    g.add_edges_from(e1_new)
    g.add_edges_from(e2_new)

    g=fixG(g)
    
    return g
    

def CreateMappings(g, 
                   interpolate=5.0, 
                   radius_scaling=None,
                   hct_values=[0.44, 0.33, 0.44]):
    '''
    
    This funtion create 3D maps based on spheres built at each graph edge 
    
    Input:
        interpolate: This in the number of points interplated at each graph edge
        radius_scaling: This factor used to increase/decrease the overll radius size
        hct_values: A list in the format [hct_in_arteriols, hct_in_cappilaries, hct_in_venules]
    '''
    def get_spheres_infos():
        
        e=g.GetEdges()
        
        pos1=np.array([g.node[i[0]]['pos'] for i in e])
        pos2=np.array([g.node[i[1]]['pos'] for i in e])
        
        radius1=np.array([g.node[i[0]]['d'] for i in e])
        radius2=np.array([g.node[i[1]]['d'] for i in e]) 
        radius=(radius1+radius2)/2.0# diameter
        radius*=0.5 # diameter to radius
        if radius_scaling is not None:
            radius*=radius_scaling
       
        so21=np.array([g.node[i[0]]['so2'] for i in e])
        so22=np.array([g.node[i[1]]['so2'] for i in e]) 
        
        types=np.array([g.node[i[0]]['type'] for i in e])
        
        velocity=np.array([g.node[i[0]]['velocity'] for i in e])
        
        vx=np.array([g.node[i[0]]['dx'] for i in e])
        vy=np.array([g.node[i[0]]['dy'] for i in e])
        vz=np.array([g.node[i[0]]['dz'] for i in e])

        return pos1, pos2, radius, so21, so22, types, velocity, vx, vy, vz
    
    pos1, pos2, radius, so21, so22, types, velocity, vx, vy, vz = get_spheres_infos()
    
    hct_values=np.array(hct_values)
    hct=hct_values[types]
    
    real_s=np.array([np.max(pos1[:,i], axis=0) for i in [0,1,2]]) # real image size
    new_s=real_s
    maxr=np.max(radius)
    new_s=tuple((np.ceil(new_s+2*maxr)).astype(int)) # image size after padding
    binary_image=np.zeros(new_s) 
    so2_image=np.zeros(new_s) 
    hct_image=np.zeros(new_s) 
    vx_image=np.zeros(new_s) 
    vy_image=np.zeros(new_s) 
    vz_image=np.zeros(new_s) 

    for p1, p2, r, s1, s2, h, velo, dx, dy, dz  in tqdm(zip( pos1, 
                                               pos2, 
                                               radius, 
                                               so21, 
                                               so22, 
                                               hct, 
                                               velocity,
                                               vx,
                                               vy,
                                               vz)):
        
        xrange, yrange, zrange = np.meshgrid(np.arange(-r, r+1, .5),
                                             np.arange(-r, r+1, .5),
                                             np.arange(-r, r+1, .5))
                                             
        sphere = (xrange**2 + yrange**2 + zrange**2) <= r**2
        x, y, z = np.where(sphere)
        
        p=(p2-p1)/interpolate # (sphere position) to fill gaps between graph nodes
        s=(s2-s1)/interpolate # interpolate so2 values between two graph nodes

        x = x+p1[0]
        y = y+p1[1]
        z = z+p1[2]
        
        ss=s1
        
        
        
        for i in range(int(interpolate)):
             
            c=(np.round(x).astype(int), np.round(y).astype(int), np.round(z).astype(int))
            
            try:
                binary_image[c]=1
                so2_image[c]=ss
                hct_image[c]=h
                vx_image[c]=velo*dx
                vy_image[c]=velo*dy
                vz_image[c]=velo*dz

            except:
                pass
            
            x += p[0]
            y += p[1]
            z += p[2]
            ss += s
        
    real_s=real_s.astype(int)
    ends=((new_s-real_s)/2.0).astype(int)

    # cropping to recover origional image size
    binary_image=binary_image[ends[0]:-ends[0], 
                              ends[1]:-ends[1], 
                              ends[2]:-ends[2]]
    
    so2_image=so2_image[ends[0]:-ends[0], 
                        ends[1]:-ends[1], 
                        ends[2]:-ends[2]]
    
    hct_image=hct_image[ends[0]:-ends[0], 
                        ends[1]:-ends[1], 
                        ends[2]:-ends[2]]    
    
    vx_image=vx_image[ends[0]:-ends[0], 
                        ends[1]:-ends[1], 
                        ends[2]:-ends[2]]  
    
    vy_image=vy_image[ends[0]:-ends[0], 
                        ends[1]:-ends[1], 
                        ends[2]:-ends[2]]   

    vz_image=vz_image[ends[0]:-ends[0], 
                        ends[1]:-ends[1], 
                        ends[2]:-ends[2]]   
        
    return binary_image.astype(int),\
            so2_image.astype('float32'),\
            hct_image.astype('float32'),\
            vx_image.astype('float32'),\
            vy_image.astype('float32'),\
            vz_image.astype('float32')


def so2FROMpo2(po2, model='mouse'):
    
    '''
    Converts the partial pressure of oxygen into o2 saturation
    based on disassociation curve
    '''
    
    # assumptions : pco2=40; temp= 37 
    if model=='mouse':
        
        # C57BL/6 mice, Uchida K. et al., Zoological Science 15, 703-706, 1997
        
        n=2.59 # hill curve
        ph=7.4
        p50=40.2 # for single Beta-globin type
    
      
    elif model=='rat':
        # Cartheuser, Comp Biochem Physiol Comp Physiol, 1993
        
        n=2.6
        ph=7.4
        p50=36
     
    po2s=po2*(10**(0.61*(ph-7.4)))
    so2=po2s**n/(po2s**n+p50**n)        
    return so2


def BuildSO2Graph(g):
    
    po2=np.array([g.node[i]['po2'] for i in g.GetNodes()])
    so2=so2FROMpo2(po2)
    
    for i,j in zip(g.GetNodes(), so2):
        g.node[i]['so2']=float(j)

    return g

def BuildRegModel(datafile, savefile='RegModel'):
    
    '''
    build and save random forest regression model
    
    Input:
        -datafile: An xlcx file with sample in rows;
                    last column in the sheet is used as the prediction, 
                    whereas the left columns are used as features    
    '''
    
    import pandas as pd
    data = pd.read_excel(datafile)

    X=[[j for j in data[i]] for i in data]
    X=np.array(X).T
    indx=np.logical_not(np.isnan(X[:,-1]))
    X=X[indx]
    
    x=X[:, 0:-1]
    y=X[:, -1]
   
    from sklearn.ensemble import RandomForestRegressor as RegModel
    
    model=RegModel(n_estimators=100, 
                   max_depth=10,
                   bootstrap=True,
                   random_state=1)
    model.fit(x,y)

    import pickle
    filename = savefile+'.sav'
    pickle.dump(model, open(filename, 'wb'))
    
    return model

def BuildVelocityGraph(g):
    
    '''
    Build random forests regression mode to infer velocity values based on:
        -diamter      -pO2      -type     -brancing order      -depth
        
        unit: mm/s = um/ms
    '''
    
    ####### build the regressino model based on existing measurments #######
    reg_model=BuildRegModel(datafile='Measurments/Data.xlsx') 
   
    
    ####### collect information from graph ########
    diameter=np.array(g.GetRadii())*2.0 # diameter
    depth=np.array(g.GetNodesPos())[:,2] # depth
    po2=np.array([g.node[i]['po2'] for i in g.GetNodes()]) # po2
    types=np.array([g.node[i]['type'] for i in g.GetNodes()]) # type
    types+=1 # to transform from 0,1,2 to 1,2,3
    
    # set branchings (to match with data I had from M.Moeini)
    from VascGraph.Tools.CalcTools import getBranches
    
    b=getBranches(g.to_undirected())
    degree=np.array([max([g.degree(i[-1]), g.degree(i[0]), 2]) for i in b])
    for i, deg in zip(b, degree):
        for j in i:
           g.node[j]['deg']=deg
    for i in g.GetNodes():
        try:
            g.node[i]['deg']
        except:
            g.node[i]['deg']=2
    
    branching_order= np.array([g.node[i]['deg'] for i in g.GetNodes()]) # branching_order
    
   
    ######### Regression #########
    x_=[[a1, a2, a3, a4, a5] for a1, a2, a3, a4, a5 in 
                             zip(diameter, po2, types, branching_order, depth)]
    
    # infer flow values and convert to velocities
    flows=reg_model.predict(x_) # pL/s
    velocities=flows/(np.pi*(diameter/2.0)**2) # mm/s
    velocities[velocities>5.0]=5.0
    
    for i, v in zip(g.GetNodes(), velocities):
        g.node[i]['velocity']=v

    return g

def BuildDirectionsGraph(g):
    
    '''
    Construct vlelocity unit vectors at each node in the graph .
    '''
    ### non entry nodes
    nodes=[i for i in g.GetNodes() if len(g.GetPredecessors(i))!=0]
    p1=[g.node[i]['pos'] for i in nodes] # starting pos
    pred=[g.GetPredecessors(i) for i in nodes]
    p0=[[g.node[j]['pos'] for j in i] for i in pred]
    p0=np.array([np.mean(i, axis=0) for i in p0]) # ending pos
    d=(p1-p0)/np.linalg.norm((p1-p0), axis=1)[:,None]          
    d[np.isnan(d)]=0
    for i, j in zip(nodes,d):
        g.node[i]['dx']=j[0]
        g.node[i]['dy']=j[1]
        g.node[i]['dz']=j[2]
        
        
    ###### entry nodes
    nodes=[i for i in g.GetNodes() if len(g.GetPredecessors(i))==0]
    p0=[g.node[i]['pos'] for i in nodes]
    succ=[g.GetSuccessors(i) for i in nodes]
    p1=[[g.node[j]['pos'] for j in i] for i in succ]
    p1=np.array([np.mean(i, axis=0) for i in p1])
    d=(p1-p0)/np.linalg.norm((p1-p0), axis=1)[:,None]          
    d[np.isnan(d)]=0
    for i, j in zip(nodes,d):
        g.node[i]['dx']=j[0]
        g.node[i]['dy']=j[1]
        g.node[i]['dz']=j[2]
        
    return g

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


def ComputeT2andDeltaB(binary_image, so2_image, hct_image,
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


def MRISignalFID(delta_B, T2_image,
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
                TE = 20.0, #ms
                echo_spacing = 0.280,
                echo_number = 64,
                gamma=2.675e5, # rad/Tesla/msec (gyromagenatic ratio)
                diff_coeff=0.8): #Le Bihan 2013 Assumed isotropic; %Proton (water) diffusion coefficient(um2/msec)

    t0=time()
    print('\nCompute MRI signal ...')

    diff_sigma=np.sqrt(2*diff_coeff*dt)
    map_size=T2_image.shape
    
    n_steps = int(np.ceil((TE+ echo_spacing*echo_number/2.0)/dt)) # time steps
    movements=np.array([diff_sigma*np.random.rand(n_protons,3) for i in range(n_steps)]) # movements in each step
    pos_protons=np.random.rand(n_protons,3)*np.array(map_size) # protons positions
    
    phase=np.zeros(n_protons) # if not, initaite
    mean_phase=[]
    signal=[]
    print('Number of iterations: %s' %n_steps)
    
    for idx, m1 in tqdm(enumerate(movements)):
        
        ###### apply diffusion ######
        valid_pos=np.less_equal(pos_protons+m1, np.array(map_size)) # check if new pos do not exceed space borders
        pos_protons+=m1*valid_pos
        
        ###### apply advection (blood flow) ######
        ind=tuple([pos_protons[:,i].astype(int) for i in [0,1,2]])
        vx=vx_image[ind]
        vy=vy_image[ind]
        vz=vz_image[ind]
        m2=np.array([vx, vy, vz]).T*dt
        valid_pos=np.less_equal(pos_protons+m2, np.array(map_size)) # check if new pos do not exceed space borders
        pos_protons+=m2*valid_pos
        
        ######## compute phase
        ind=tuple([pos_protons[:,i].astype(int) for i in [0,1,2]])
        a = -(1.0/T2_image[ind])*dt # phase degradation  from T2
        b = 1j*gamma*delta_B[ind]*dt # phase variation from delta_B
    
        phase=phase+a+b; # phase for that proton is already initiated
        mean_phase.append(np.imag(np.mean(phase)))
        signal.append(np.abs(np.mean(np.exp(phase))))

    print('Time for protons simulation: %s' %(time()-t0))


    return signal



def MRISignalSE(delta_B, T2_image,
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
                TE = 20.0, #ms
                echo_spacing = 0.280,
                echo_number = 64,
                gamma=2.675e5, # rad/Tesla/msec (gyromagenatic ratio)
                diff_coeff=0.8): #Le Bihan 2013 Assumed isotropic; %Proton (water) diffusion coefficient(um2/msec)

    def pluses_train(t_steps, TE, echo_spacing):
        add_step=TE+(echo_spacing/2.0)
        echo_half=(echo_spacing/2.0)
        output=np.array(t_steps-add_step)
        output[output<0]=0
        output = (output*.5)%echo_spacing < echo_half
        return output
        
    def dephasing_sign(n_steps, TE, echo_spacing, echo_number):
        
        echo_half=echo_spacing/2.0
        t_steps=np.linspace(0, (TE+ echo_spacing*echo_number), n_steps)
        dephasing=pluses_train(t_steps, TE, echo_spacing)
        chck=pluses_train(TE+echo_half, TE, echo_spacing)
        dephasing[t_steps<(TE/2.0)]=chck
        dephasing[(t_steps>(TE/2.0))&(t_steps<(TE+echo_half))]=chck-1
        dephasing=dephasing*2-1 # binary to 1, -1
        if dephasing[0]==-1:
            dephasing*=-1
        return dephasing    
    
    t0=time()
    print('\nCompute MRI signal ...')

    diff_sigma=np.sqrt(2*diff_coeff*dt)
    map_size=T2_image.shape
    n_steps = int(np.ceil((TE+ echo_spacing*echo_number)/dt)) # time steps
    t_steps=np.linspace(0, TE+ echo_spacing*echo_number, n_steps)
    
    pos_protons=np.random.rand(n_protons,3)*np.array(map_size) # protons positions
    
    dephasing=dephasing_sign(n_steps, TE, echo_spacing, echo_number)
    phase=np.zeros(n_protons) # if not, initaite
    signal=np.zeros(n_protons) # if not, initaite
    mean_phase=[]
    total_signal=[]

    print('Number of time steps: %s' %n_steps)    
    for idx, sign in tqdm(zip(range(n_steps), dephasing)):
        
        ###### apply diffusion ######
        m=diff_sigma*np.random.rand(n_protons,3) # movements
        valid_pos=np.less_equal(pos_protons+m, np.array(map_size)) # check if new pos do not exceed space borders
        pos_protons+=m*valid_pos
        
        ###### apply advection (blood flow) ######
        ind=tuple([pos_protons[:,i].astype(int) for i in [0,1,2]])
        vx=vx_image[ind]
        vy=vy_image[ind]
        vz=vz_image[ind]
        m2=np.array([vx, vy, vz]).T*dt
        valid_pos=np.less_equal(pos_protons+m2, np.array(map_size)) # check if new pos do not exceed space borders
        pos_protons+=m2*valid_pos
        
        ######## compute phase
        ind=tuple([pos_protons[:,i].astype(int) for i in [0,1,2]])
        a = -(1.0/T2_image[ind])*dt # phase degradation  from T2
        b = 1j*gamma*delta_B[ind]*dt # phase variation from delta_B
    
        signal=signal+a # signal amplitude for each proton
        phase=phase+(sign*b); # phase for that proton is already initiated
        mean_phase.append(np.imag(np.mean(phase)))
        
        ######## compute signal
        if T1:
            s=np.exp(signal)*np.exp(phase)*np.exp(-idx*dt/T1)
            total_signal.append(np.abs(np.mean(s)))
        else:
            total_signal.append(np.abs(np.mean(np.exp(signal)*np.exp(phase))))

    print('Time for protons simulation: %s' %(time()-t0))


    return t_steps, total_signal


def MRISignalGESFIDE(delta_B, T2_image,
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
                echo_spacing = 0.25,
                echo_number = [12,24],
                gamma=2.675e5, # rad/Tesla/msec (gyromagenatic ratio)
                diff_coeff=0.8): #Le Bihan 2013 Assumed isotropic; %Proton (water) diffusion coefficient(um2/msec)

        
    def dephasing_sign(n_steps, TE, echo_spacing, echo_number):
        
        t_steps=np.linspace(0, (TE+ echo_spacing*np.sum(echo_number)), n_steps)
        dephasing=t_steps
        dephasing[dephasing<(TE+echo_spacing*echo_number[0])]=1
        dephasing[dephasing>(TE+echo_spacing*echo_number[0])]=-1
        
        return dephasing    
    
    t0=time()
    print('\nCompute MRI signal ...')

    diff_sigma=np.sqrt(2*diff_coeff*dt)
    map_size=T2_image.shape
    n_steps = int(np.ceil((TE+echo_spacing*np.sum(echo_number))/dt)) # time steps
    t_steps=np.linspace(0, TE+echo_spacing*np.sum(echo_number), n_steps)
    
    pos_protons=np.random.rand(n_protons,3)*np.array(map_size) # protons positions
    
    dephasing=dephasing_sign(n_steps, TE, echo_spacing, echo_number)
    phase=np.zeros(n_protons) # if not, initaite
    signal=np.zeros(n_protons) # if not, initaite
    mean_phase=[]
    total_signal=[]

    print('Number of time steps: %s' %n_steps)    
    for idx, sign in tqdm(zip(range(n_steps), dephasing)):
        
        ###### apply diffusion ######
        m=diff_sigma*np.random.rand(n_protons, 3) # movements
        valid_pos=np.less_equal(pos_protons+m, np.array(map_size)) # check if new pos do not exceed space borders
        pos_protons+=m*valid_pos
        
        ###### apply advection (blood flow) ######
        ind=tuple([pos_protons[:,i].astype(int) for i in [0,1,2]])
        vx=vx_image[ind]
        vy=vy_image[ind]
        vz=vz_image[ind]
        m2=np.array([vx, vy, vz]).T*dt
        valid_pos=np.less_equal(pos_protons+m2, np.array(map_size)) # check if new pos do not exceed space borders
        pos_protons+=m2*valid_pos
        
        ######## compute phase
        ind=tuple([pos_protons[:,i].astype(int) for i in [0,1,2]])
        a = -(dt/T2_image[ind]) # signal degradation  from T2
        b = 1j*gamma*delta_B[ind]*dt # phase variation from delta_B, T2* effect
        
        signal=signal+a # signal amplitude for each proton
        phase=phase+(sign*b); # phase for that proton is already initiated
        mean_phase.append(np.imag(np.mean(phase)))
        
        ######## compute signal
        if T1:
            s=np.exp(signal)*np.exp(phase)*np.exp(-idx*dt/T1)
            total_signal.append(np.abs(np.mean(s)))
        else:
            total_signal.append(np.abs(np.mean(np.exp(signal)*np.exp(phase))))

    print('Time for protons simulation: %s' %(time()-t0))


    return t_steps, total_signal, mean_phase


def test():
    
    # plus or negative dephasing
    n_steps=10
    TE=5
    echo_spacing=2
    echo_number=3
    dephasing=dephasing_sign(n_steps, TE, echo_spacing, echo_number)
    t_steps=np.linspace(0, (TE+ echo_spacing*echo_number), n_steps)
    plt.figure()
    plt.plot(t_steps, dephasing)
    print(TE, echo_spacing)
    plt.figure()
    plt.plot(t_steps, pluses_train(t_steps, TE, echo_spacing))
    
    
#### save velocity image as (rgb) channels for x,y,z components
#    im=np.array([vx_image.T, vy_image.T, vz_image.T]).astype('uint16')
#    im=np.rollaxis(im, 3)
#    im=np.rollaxis(im, 3)
#    im=np.rollaxis(im, 3)
#    skio.imsave('velo_image.tif', im)
    #plt.hist(velocity_image[velocity_image>0].ravel(), bins=1000)    
    

if __name__=='__main__':

    mouse_folder='DATA_2PH/mouse_20120626/'
    
    #### create so2 graph ####
    print('\nRead graph with pO2 ...')
    g=BuildO2GraphFromMat(mouse_folder, scaling=False)
    
    #### add so2 to graph ####
    print('\nCompute so2 ...')
    g=BuildSO2Graph(g)
    
    #### add directions, dx, dy, dz, to graph ####
    print('\nCompute dx, dy, dz ...')
    g=BuildDirectionsGraph(g)

    #### add velocity to graph ####
    print('\nPredict velocity ...\n')
    g=BuildVelocityGraph(g)
    
    ##### create 3d maps for anatomy, so2, hct and velocity #####
    t0=time()
    print('\nCreate anatomical, SatO2, Htc and velocity maps ...')
    binary_image, so2_image, hct_image, vx_image, vy_image, vz_image= CreateMappings(g, 
                                                                        interpolate=10.0,
                                                                        radius_scaling=0.5,
                                                                        hct_values=[0.44, 0.33, 0.44])
    
    print('Time : %s' %(time()-t0))
    
  
    ##### compute 3d maps of T2 paramter and delta B ####
    t0=time()
    print('\nCreate T2 and delta B maps ...')
    T2_image, delta_B = ComputeT2andDeltaB(binary_image=binary_image, 
                                           so2_image=so2_image,
                                           hct_image=hct_image)
    del binary_image, so2_image, hct_image
    print('Time : %s' %(time()-t0))
    
'''

    ###### get MRI signal #######
    d1=0.8
    d2=0.8
    dt=.005
    TE=.2
    echo_spacing=.1
    echo_number=[12, 24]
    t, signal1, mean_phase1 = MRISignalGESFIDE(delta_B=delta_B, T2_image=T2_image,
                        vx_image=vx_image*0,
                        vy_image=vy_image*0,
                        vz_image=vz_image*0,
                        dt=dt,
                        TE = TE, #ms
                        T1=False,
                        echo_spacing=echo_spacing,
                        echo_number=echo_number,
                        diff_coeff=d1)
    
    _,signal2, mean_phase2  = MRISignalGESFIDE(delta_B=delta_B, T2_image=T2_image,
                        vx_image=vx_image,
                        vy_image=vy_image,
                        vz_image=vz_image,
                        dt=dt,
                        TE = TE, #ms
                        T1=False,
                        echo_spacing=echo_spacing,
                        echo_number=echo_number,
                        diff_coeff=d2)
    
    plt.figure()         
    plt.plot(t, signal1, label='D = '+str(d1)+', without adv')
    plt.plot(t, signal2, label='D = '+str(d2)+', with adv')
    plt.legend(loc='upper right')

    plt.figure()         
    plt.plot(t, np.array(signal2)/signal1, label='signal ratio')



    signal11=sp.io.loadmat('S_1_0.8.mat')
    signal11=signal11['S'][0]
    signal11=signal11['s'][0]
    signal11=signal11[0]
    signal22=sp.io.loadmat('S_1_5.8.mat')
    signal22=signal22['S'][0]
    signal22=signal22['s'][0]
    signal22=signal22[0]

    plt.figure()         
    plt.plot(signal11, label='D = '+str(d1))
    plt.plot(signal22, label='D = '+str(d2))
    plt.legend(loc='upper right')

    
    


# importance of features
plt.figure()
names=[i for i in data]
plt.bar(names[:-1], model.feature_importances_)

plt.figure()
plt.scatter(x[:,3], y,label='True')
plt.scatter(x[:,3], model.predict(x), label='Pred')


    

#
#visStack(binary_image.astype(int))
#mlab.figure()
#visVolume(so2_image*255)
#



#t2mask=sp.io.loadmat('T2mask.mat')['T2mask']
#mlab.figure()
#visVolume(t2mask)
#
#mlab.figure()
#visVolume(T2_image)

#
#
#skio.imsave('my_so2_map.tif', (so2_image*255).astype('uint8'))
#
#po2=np.array([g.node[i]['po2'] for i in g.GetNodes()])
#so2=np.array([g.node[i]['so2'] for i in g.GetNodes()])
#
#
#
#plt.imshow(so2_image[120])
#plt.colorbar()
#
#
#sato2=sp.io.loadmat('sato2mask.mat')['SatO2mask']
#mlab.figure()
#visVolume(sato2*255)
#a=sato2[:,:,300]
#plt.figure()
#plt.imshow(a)
#


#deltab=sp.io.loadmat('test_DeltaB.mat')['delta_B']
#plt.figure()
#plt.imshow(deltab[:,:,100], cmap='Greys')
#bb=(deltab-deltab.min())/(deltab.max()-deltab.min())
#del deltab
#mlab.figure()
#visVolume(bb*255)




#
#
#a=sp.io.loadmat('RAW_MASK_BIN1.mat')
#
#visStack((sato2>0).astype(int))
#mlab.figure()
#visVolume(sato2)
#
'''
