#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 09:58:10 2019

@author: rdamseh
"""


import numpy as np
from VascGraph.GeomGraph.DiGraph import DiGraph

from VascGraph.Tools.CalcTools import fixG
from VascGraph.Tools.VisTools import visG

class SynthVN:
    
    def __init__(self, 
                 n_vessels=[100],
                 vessel_radii=[5.0],
                 length=[100],
                 type='circular', 
                 fixed_o2=True):
       
        self.type=type
        self.fixed_o2=fixed_o2
        self.n_vessels=n_vessels
        self.vessel_radii=vessel_radii
        self.length=length
        
    def init_seed(self): 
        np.random.seed(99999)
                 
                 
    def GetCircularVN(self, sphere_radii=[25.0]):
        self.init_seed()
        g=self.CircularRandomVN(sphere_radii=sphere_radii,
                                n_vessels=self.n_vessels,
                                length=self.length,
                                vessel_radii=self.vessel_radii)
        return self.AddO2(g)
    
    def GetCylinderVN(self, sphere_radii=[25.0], hight=10.0):
        self.init_seed()
        g=self.CylinderRandomVN(sphere_radii=sphere_radii,
                                hight=hight,
                                n_vessels=self.n_vessels,
                                length=self.length,
                                vessel_radii=self.vessel_radii)
        return self.AddO2(g)

    
    def GetNormalVN(self, x=512, y=512, z=512):
        self.init_seed()
        g=self.NormalRandomVN(x=x, y=y, z=z,
                            n_vessels=self.n_vessels,
                            length=self.length,
                            vessel_radii=self.vessel_radii)
        return self.AddO2(g)    
    
    def GetOrientedVN(self, x=512, y=512, z=512,
                    orientation=[0, 0]):
        self.init_seed()
        g=self.NormalRandomOrientedVN(x=x, y=y, z=z, orientation=orientation,
                                    n_vessels=self.n_vessels,
                                    length=self.length,
                                    vessel_radii=self.vessel_radii)
        return self.AddO2(g)    

    def AddO2(self, g):
         
        if self.fixed_o2:
            g=self.BuildFixedO2Graph(g)
        else:
            g=self.BuildFixedO2Graph(g)
            
        g=self.BuildSO2Graph(g)
        g=self.BuildDirectionsGraph(g)         
        
        return g         

    def CircularRandomVN(self, sphere_radii=[50, 100, 200], # *center------)rad1-------)rad2--------)radn
            length=[50, 75, 100], # vessel length generated on the surface of shpheres with various radii
            interpolate=25,
            n_vessels=[75, 75, 15],
            vessel_radii=[2.5, 5.0, 10.0], 
            exclude_depth=False,
            factor=1.0):
            
        g=DiGraph() # initiate a directed graph
        
        
        factor=factor #this is to reduce complexity #
        
        for r, l, n, rad in zip(sphere_radii, length, n_vessels, vessel_radii):
            
            # this is compensated later, it is done to reduce complexity #
            r= r/factor
            l=l/factor
            #
            
            xrange, yrange, zrange = np.meshgrid(np.arange(-r, r+1, 1),
                                             np.arange(-r, r+1, 1),
                                             np.arange(-r, r+1, 1))
           
            sphere = ((xrange**2 + yrange**2 + zrange**2) < (r**2)+1.0)\
                    |((xrange**2 + yrange**2 + zrange**2) > (r**2)-1.0)
            
            pos_ind = np.where(sphere)
            del sphere
            
            x=xrange[pos_ind].ravel()
            y=yrange[pos_ind].ravel()
            z=zrange[pos_ind].ravel()
        
            del xrange, yrange, zrange
            
            ind=np.array(range(len(x)))
            np.random.shuffle(ind)
        
            ind=ind[0:n]
            if len(ind)<n :
                print('Number of vessels: %s' %(len(ind)))
                print('Cannot generate enough vessles!')
                print('To solve: 1. decrease factor or 2. decrease number of vessels.')
                return 
    
            
            x=x[ind] 
            y=y[ind]
            z=z[ind]
            
    
                    
            p1=np.array([x,y,z]).T # starting points in each segment
            p1_norm=p1/np.linalg.norm(p1, axis=1)[:,None]
            lenn=l #(l+l/2.0*np.random.rand(len(p1))) # randomly vary length
            p2=p1+(p1_norm)*lenn #p1+(p1_norm)*lenn[:, None] # ending point in each segmet
            
                
            ###### interpolate points along each segment ##########
            pos_x=[]
            pos_y=[]
            pos_z=[]
            
            for i1, i2 in zip(p1, p2):
                pos_x.append(np.linspace(i1[0], i2[0], interpolate))
                pos_y.append(np.linspace(i1[1], i2[1], interpolate))
                pos_z.append(np.linspace(i1[2], i2[2], interpolate))
                
            pos_x=np.array(pos_x).ravel()*factor
            pos_y=np.array(pos_y).ravel()*factor
            pos_z=np.array(pos_z).ravel()*factor
            ##########################################################
        
        
            ######### create randomly varying radii #########################
            radii=rad*np.ones(n) #rad+(rad/2.0)*np.random.rand(n) # radius in each segment
            radii=np.repeat(radii.T[:, None], axis=1, repeats=interpolate) 
            shape=radii.shape # this shape info is used to build edges below
            radii=radii.ravel()
            ################################################
            
            n_nodes=g.number_of_nodes()
            nodes=np.array(range(len(pos_x)))+n_nodes
            g.add_nodes_from(nodes)
            
            # add pos and radius to graph
            for node, ix, iy, iz, rr in zip(nodes, pos_x, pos_y, pos_z, radii):
                g.node[node]['pos']=np.array([ix, iy, iz])
                g.node[node]['r']=rr
                g.node[node]['d']=rr*2.0
    
            # add edges to graph
            ed=np.linspace(interpolate*n, 1, interpolate*n)+n_nodes-1
            ed=ed.reshape(shape).astype(int)
            ed1=ed[:,:-1]
            ed2=ed[:,1:]
            edges=[[i1,j1] for i, j in zip(ed1, ed2) for i1, j1 in zip(i,j)]
            
            del ed, ed1, ed2
            g.add_edges_from(edges)
            
        max_pos=float(np.max(g.GetNodesPos()))
        for i in g.GetNodes():
            g.node[i]['pos']=g.node[i]['pos']+max_pos
            
            
        # add types
        def label_path(g, source):
            t=g.node[source]['type']
            i=source
            while len(g.GetSuccessors(i))>0:
                i=g.GetSuccessors(i)[0]
                g.node[i]['type']=t
            return g
            
        pos=g.GetNodesPos()
        pos_max=np.max(pos, axis=0)[0]
        pos_thr=pos_max/2.0
        sources=[i for i in g.GetNodes() if len(g.GetPredecessors(i))==0]
        radius_thr=5.0
        
    #    for i in sources:
    #        if g.node[i]['pos'][0]<=pos_thr and g.node[i]['r']>radius_thr:
    #            g.node[i]['type']=0 # artery
    #            g=label_path(g, i)
    #            
    #        elif g.node[i]['pos'][0]>pos_thr and g.node[i]['r']>radius_thr:
    #            g.node[i]['type']=2 # venule
    #            g=label_path(g, i)
    #            
    #        else:
    #            g.node[i]['type']=1 # cappilary 
    #            g=label_path(g, i)
        
        for i in g.GetNodes():
            g.node[i]['type']=0
                
        # remove arteries and veins exceeding certain depth
    #    if exclude_depth:
    #        depth_thr=300.0
    #        for i in g.GetNodes():
    #            if g.node[i]['type']==0 or g.node[i]['type']==2:
    #                if g.node[i]['pos'][2]>depth_thr:
    #                    g.remove_node(i)
    #                   
    #        g=fixG(g, copy=True)
            
        return g
    
    def CylinderRandomVN(self, sphere_radii=[50, 100, 200], # *center------)rad1-------)rad2--------)radn
            hight=10.0,
            length=[50, 75, 100], # vessel length generated on the surface of shpheres with various radii
            interpolate=25,
            n_vessels=[75, 75, 15],
            vessel_radii=[2.5, 5.0, 10.0], 
            exclude_depth=False,
            factor=1.0):
            
        g=DiGraph() # initiate a directed graph
        
        
        factor=factor #this is to reduce complexity #
        
        for r, l, n, rad in zip(sphere_radii, length, n_vessels, vessel_radii):
            
            # this is compensated later, it is done to reduce complexity #
            r= r/factor
            l=l/factor
            #
            
            rh=hight/2.0
            
            xrange, yrange, zrange = np.meshgrid(np.arange(-r, r+1, 1),
                                             np.arange(-r, r+1, 1),
                                             np.arange(-rh, rh+1, 1))
           
            sphere = ((xrange**2 + yrange**2) < (r**2))\
                    |((xrange**2 + yrange**2) > (r**2))
            
            sphere = sphere & (np.abs(zrange)<=hight)
            pos_ind = np.where(sphere)
            del sphere
            
            x=xrange[pos_ind].ravel()
            y=yrange[pos_ind].ravel()
            z=zrange[pos_ind].ravel()
        
            del xrange, yrange, zrange
            
            ind=np.array(range(len(x)))
            np.random.shuffle(ind)
        
            ind=ind[0:n]
            if len(ind)<n :
                print('Number of vessels: %s' %(len(ind)))
                print('Cannot generate enough vessles!')
                print('To solve: 1. decrease factor or 2. decrease number of vessels.')
                return 
    
            
            x=x[ind] 
            y=y[ind]
            z=z[ind]
            
            p1=np.array([x,y,z]).T
                    
            p1_xy=np.array([x,y]).T # starting points in each segment
            p1_norm_xy=p1_xy/np.linalg.norm(p1_xy, axis=1)[:, None]
            lenn=l #(l+l/2.0*np.random.rand(len(p1))) # randomly vary length
            p2_xy=p1_xy+(p1_norm_xy)*lenn #p1+(p1_norm)*lenn[:, None] # ending point in each segmet
            
            p2=p1.copy()
            p2[:, 0:-1]=p2_xy
            
                
            ###### interpolate points along each segment ##########
            pos_x=[]
            pos_y=[]
            pos_z=[]
            
            for i1, i2 in zip(p1, p2):
                pos_x.append(np.linspace(i1[0], i2[0], interpolate))
                pos_y.append(np.linspace(i1[1], i2[1], interpolate))
                pos_z.append(np.linspace(i1[2], i2[2], interpolate))
                
            pos_x=np.array(pos_x).ravel()*factor
            pos_y=np.array(pos_y).ravel()*factor
            pos_z=np.array(pos_z).ravel()*factor
            ##########################################################
        
        
            ######### create randomly varying radii #########################
            radii=rad*np.ones(n) #rad+(rad/2.0)*np.random.rand(n) # radius in each segment
            radii=np.repeat(radii.T[:, None], axis=1, repeats=interpolate) 
            shape=radii.shape # this shape info is used to build edges below
            radii=radii.ravel()
            ################################################
            
            n_nodes=g.number_of_nodes()
            nodes=np.array(range(len(pos_x)))+n_nodes
            g.add_nodes_from(nodes)
            
            # add pos and radius to graph
            for node, ix, iy, iz, rr in zip(nodes, pos_x, pos_y, pos_z, radii):
                g.node[node]['pos']=np.array([ix, iy, iz])
                g.node[node]['r']=rr
                g.node[node]['d']=rr*2.0
    
            # add edges to graph
            ed=np.linspace(interpolate*n, 1, interpolate*n)+n_nodes-1
            ed=ed.reshape(shape).astype(int)
            ed1=ed[:,:-1]
            ed2=ed[:,1:]
            edges=[[i1,j1] for i, j in zip(ed1, ed2) for i1, j1 in zip(i,j)]
            
            del ed, ed1, ed2
            g.add_edges_from(edges)
            
        max_pos=float(np.max(g.GetNodesPos()))
        for i in g.GetNodes():
            g.node[i]['pos']=g.node[i]['pos']+max_pos
            
            
        # add types
        def label_path(g, source):
            t=g.node[source]['type']
            i=source
            while len(g.GetSuccessors(i))>0:
                i=g.GetSuccessors(i)[0]
                g.node[i]['type']=t
            return g
            
        pos=g.GetNodesPos()
        pos_max=np.max(pos, axis=0)[0]
        pos_thr=pos_max/2.0
        sources=[i for i in g.GetNodes() if len(g.GetPredecessors(i))==0]
        radius_thr=5.0
        
    #    for i in sources:
    #        if g.node[i]['pos'][0]<=pos_thr and g.node[i]['r']>radius_thr:
    #            g.node[i]['type']=0 # artery
    #            g=label_path(g, i)
    #            
    #        elif g.node[i]['pos'][0]>pos_thr and g.node[i]['r']>radius_thr:
    #            g.node[i]['type']=2 # venule
    #            g=label_path(g, i)
    #            
    #        else:
    #            g.node[i]['type']=1 # cappilary 
    #            g=label_path(g, i)
        
        for i in g.GetNodes():
            g.node[i]['type']=0
                
        # remove arteries and veins exceeding certain depth
    #    if exclude_depth:
    #        depth_thr=300.0
    #        for i in g.GetNodes():
    #            if g.node[i]['type']==0 or g.node[i]['type']==2:
    #                if g.node[i]['pos'][2]>depth_thr:
    #                    g.remove_node(i)
    #                   
    #        g=fixG(g, copy=True)
            
        return g    
    
    def NormalRandomVN(self, x=512,
                y=512,
                z=512,
                length=[200, 200, 152], # vessel length generated on the surface of shpheres with various radii
                interpolate=25,
                n_vessels=[10, 50, 100],
                vessel_radii=[10.0, 5, 2.5]):
        
        g=DiGraph() # initiate a directed graph
    
        for l, n, r in zip( length, n_vessels, vessel_radii):
            
            # random positions
            px1=x*np.random.rand(n)
            py1=y*np.random.rand(n)
            pz1=z*np.random.rand(n)
            p1=np.array([px1, py1, pz1]).T
           
            px2=x*np.random.rand(n)
            py2=y*np.random.rand(n)
            pz2=z*np.random.rand(n)
            p2=np.array([px2, py2, pz2]).T
    
            
            ###### interpolate points along each segment ##########
            pos_x=[]
            pos_y=[]
            pos_z=[]
            
            for i1, i2 in zip(p1, p2):
                pos_x.append(np.linspace(i1[0], i2[0], interpolate))
                pos_y.append(np.linspace(i1[1], i2[1], interpolate))
                pos_z.append(np.linspace(i1[2], i2[2], interpolate))
                
            pos_x=np.array(pos_x).ravel()
            pos_y=np.array(pos_y).ravel()
            pos_z=np.array(pos_z).ravel()
            ##########################################################
        
        
            ######### create randomly varying radii #########################
            radii=r*np.ones(n) #r+r*np.random.rand(n)*0.25 # radius in each segment
            radii=np.repeat(radii.T[:, None], axis=1, repeats=interpolate) 
            shape=radii.shape # this shape info is used to build edges below
            radii=radii.ravel()
            ################################################
    
    
            n_nodes=g.number_of_nodes()
            nodes=np.array(range(len(pos_x)))+n_nodes
            g.add_nodes_from(nodes)
            
            # add pos and radius to graph
            for node, ix, iy, iz, rr in zip(nodes, pos_x, pos_y, pos_z, radii):
                g.node[node]['pos']=np.array([ix, iy, iz])
                g.node[node]['r']=rr
                g.node[node]['d']=rr*2.0
                
            # add edges to graph
            ed=np.linspace(1, interpolate*n, interpolate*n)+n_nodes-1
            ed=ed.reshape(shape).astype(int)
            ed1=ed[:,:-1]
            ed2=ed[:,1:]
            edges=[[i1,j1] for i, j in zip(ed1, ed2) for i1, j1 in zip(i,j)]
            
            del ed, ed1, ed2
            g.add_edges_from(edges)
        
          
        # add types
        def label_path(g, source):
            t=g.node[source]['type']
            i=source
            while len(g.GetSuccessors(i))>0:
                i=g.GetSuccessors(i)[0]
                g.node[i]['type']=t
            return g
            
        pos=g.GetNodesPos()
        pos_max=np.max(pos, axis=0)[0]
        pos_thr=pos_max/2.0
        sources=[i for i in g.GetNodes() if len(g.GetPredecessors(i))==0]
        radius_thr=5.0
        
        for i in g.GetNodes():
            g.node[i]['type']=0    
        
    #    for i in sources:
    #        if g.node[i]['pos'][0]<=pos_thr and g.node[i]['r']>radius_thr:
    #            g.node[i]['type']=0 # artery
    #            g=label_path(g, i)
    #            
    #        elif g.node[i]['pos'][0]>pos_thr and g.node[i]['r']>radius_thr:
    #            g.node[i]['type']=2 # venule
    #            g=label_path(g, i)
    #            
    #        else:
    #            g.node[i]['type']=1 # cappilary 
    #            g=label_path(g, i)
    
    
        return g
    
    
    
    
    def NormalOrientedVN(self, x=512,
                y=512,
                z=512,
                orientation=[0, 0],
                length=[200, 200, 152], # vessel length generated on the surface of shpheres with various radii
                interpolate=25,
                n_vessels=[10, 50, 100],
                vessel_radii=[10.0, 5, 2.5]):
        
        orientation=(np.array(orientation)*np.pi)/180
        
        g=DiGraph() # initiate a directed graph
    
        for l, n, r in zip( length, n_vessels, vessel_radii):
            
            # random positions
            px1=x*np.random.rand(n)
            py1=y*np.random.rand(n)
            pz1=np.zeros(n)
            p1=np.array([px1, py1, pz1]).T
           
     
            pxx=(l*np.sin(orientation[0])*np.cos(orientation[0]))*np.ones(n)
            px2=px1+pxx
            pyy=(l*np.sin(orientation[0])*np.sin(orientation[0]))*np.ones(n)
            py2=py1+pyy  
            pz2=np.cos(orientation[0])*l*np.ones(n)
            p2=np.array([px2, py2, pz2]).T
    
            
            ###### interpolate points along each segment ##########
            pos_x=[]
            pos_y=[]
            pos_z=[]
            
            for i1, i2 in zip(p1, p2):
                pos_x.append(np.linspace(i1[0], i2[0], interpolate))
                pos_y.append(np.linspace(i1[1], i2[1], interpolate))
                pos_z.append(np.linspace(i1[2], i2[2], interpolate))
                
            pos_x=np.array(pos_x).ravel()
            pos_y=np.array(pos_y).ravel()
            pos_z=np.array(pos_z).ravel()
            ##########################################################
        
        
            ######### create randomly varying radii #########################
            radii=r*np.ones(n) #r+r*np.random.rand(n)*0.25 # radius in each segment
            radii=np.repeat(radii.T[:, None], axis=1, repeats=interpolate) 
            shape=radii.shape # this shape info is used to build edges below
            radii=radii.ravel()
            ################################################
    
    
            n_nodes=g.number_of_nodes()
            nodes=np.array(range(len(pos_x)))+n_nodes
            g.add_nodes_from(nodes)
            
            # add pos and radius to graph
            for node, ix, iy, iz, rr in zip(nodes, pos_x, pos_y, pos_z, radii):
                g.node[node]['pos']=np.array([ix, iy, iz])
                g.node[node]['r']=rr
                g.node[node]['d']=rr*2.0
                
            # add edges to graph
            ed=np.linspace(1, interpolate*n, interpolate*n)+n_nodes-1
            ed=ed.reshape(shape).astype(int)
            ed1=ed[:,:-1]
            ed2=ed[:,1:]
            edges=[[i1,j1] for i, j in zip(ed1, ed2) for i1, j1 in zip(i,j)]
            
            del ed, ed1, ed2
            g.add_edges_from(edges)
        
          
        # add types
        def label_path(g, source):
            t=g.node[source]['type']
            i=source
            while len(g.GetSuccessors(i))>0:
                i=g.GetSuccessors(i)[0]
                g.node[i]['type']=t
            return g
            
        pos=g.GetNodesPos()
        pos_max=np.max(pos, axis=0)[0]
        pos_thr=pos_max/2.0
        sources=[i for i in g.GetNodes() if len(g.GetPredecessors(i))==0]
        radius_thr=5.0
        
        for i in g.GetNodes():
            g.node[i]['type']=0    
        
    #    for i in sources:
    #        if g.node[i]['pos'][0]<=pos_thr and g.node[i]['r']>radius_thr:
    #            g.node[i]['type']=0 # artery
    #            g=label_path(g, i)
    #            
    #        elif g.node[i]['pos'][0]>pos_thr and g.node[i]['r']>radius_thr:
    #            g.node[i]['type']=2 # venule
    #            g=label_path(g, i)
    #            
    #        else:
    #            g.node[i]['type']=1 # cappilary 
    #            g=label_path(g, i)
    
        return g


    def BuildRegModel(self, datafile='Measurments/Data.xlsx'):
        
        '''
        build and save random forest regression models
        
        Input:
            -datafile: An xlcx file with sample in rows;
                        one column in the sheet is used as the prediction, 
                        whereas the left columns are used as features    
        '''
        
        # read measurments data
        import pandas as pd
        data = pd.read_excel(datafile)
        from sklearn.ensemble import RandomForestRegressor as RegModel
            
        def BuildModel(data, x_labels=[], y_labels=[]):
            
            X=[[j for j in data[i]] for i in x_labels]
            X=np.array(X).T
            y=[[j for j in data[i]] for i in y_labels][0]
            y=np.array(y)
            indx=np.logical_not(np.isnan(y))
            x=X[indx]
            y=y[indx]
            model=RegModel(n_estimators=100, 
                           max_depth=10,
                           bootstrap=True,
                           random_state=1)
            model.fit(x,y) 
            
            return model
        
        # po2 model
        model_PO2=BuildModel(data=data, x_labels=['Diameter (um)', 'type(A-C-V)', 'depth (um)'], 
                                                  y_labels=['pO2 (mmHg)'])
        # flow model
        model_flow=BuildModel(data=data, x_labels=['Diameter (um)', 'type(A-C-V)', 'depth (um)'], 
                                                           y_labels=['flow (pl/s)'])
        
    
        import pickle
        
        savefile='RegModel_PO2'
        filename = savefile+'.sav'
        pickle.dump(model_PO2, open(filename, 'wb'))
        
        savefile='RegModel_flow'
        filename = savefile+'.sav'
        pickle.dump(model_flow, open(filename, 'wb')) 
        
        return model_PO2, model_flow
    
    
    def BuildO2Graph(self, g, datafile='Measurments/Data.xlsx'):
        
        '''
        Build random forests regression mode to infer velocity values based on:
            -diamter      -pO2      -type     -brancing order      -depth
            
            unit: mm/s = um/ms
        '''
        
        ####### build the regressino model based on existing measurments #######
        reg_model_PO2, reg_model_flow=self.BuildRegModel(datafile=datafile) 
           
        ####### collect information from graph ########
        diameter=np.array(g.GetRadii())*2.0 # diameter
        types=np.array([g.node[i]['type'] for i in g.GetNodes()]) # type
        types+=1 # to transform from 0,1,2 to 1,2,3
        depth=np.array(g.GetNodesPos())[:,2] # depth
    
        
       
        ######### Regression #########
        x_=[[a1, a2, a3] for a1, a2, a3 in zip(diameter,  types,  depth)]
        
        # infer po2 values
        values_PO2=reg_model_PO2.predict(x_) # pL/s
        for i, po2 in zip(g.GetNodes(), values_PO2):
            g.node[i]['po2']=po2
        
        # infer flow values and convert to velocities
        flows=reg_model_flow.predict(x_) # pL/s
        velocities=flows/(np.pi*(diameter/2.0)**2) # mm/s
        velocities[velocities>5.0]=5.0
        
        for i, v in zip(g.GetNodes(), velocities):
            g.node[i]['velocity']=v
    
        return g
    
    def BuildFixedO2Graph(self, g):
        
        for i in g.GetNodes():
            g.node[i]['po2']=60.0
            g.node[i]['velocity']=5.0
            
        return g

    def BuildDirectionsGraph(self, g):
        
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
    
    
    def so2FROMpo2(self, po2, model='mouse'):
        
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
    
    
    def BuildSO2Graph(self, g):
        
        po2=np.array([g.node[i]['po2'] for i in g.GetNodes()])
        so2=self.so2FROMpo2(po2)
        
        for i,j in zip(g.GetNodes(), so2):
            g.node[i]['so2']=float(j)
    
        return g    

if __name__=='__main__':
    
    

        g=CircularVN()
        g=BuildO2Graph(g)
        visG(g, gylph_r=.01, diam=True)











