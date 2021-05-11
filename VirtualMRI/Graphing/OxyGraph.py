#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 14:35:46 2020

@author: rdamseh
"""

import numpy as np
from VascGraph.GeomGraph.DiGraph import DiGraph

from VascGraph.Tools.CalcTools import fixG
from VascGraph.Tools.VisTools import visG


class OxyGraph:
    
    def __init__(self, g,
                 with_velocity=True,
                 fixed_o2=False):
        '''
        Inputs: 
            g: networkx directed graph generated using VascGraph package
                https://www.github.com/Damseh/VascGraph
        '''
       
        self.g=g
        self.fixed_o2=fixed_o2
        self.with_velocity=with_velocity
        
    def init_seed(self): 
        np.random.seed(99999)
                 

    def AddO2(self, g):
         
        if self.fixed_o2:
            g=self.BuildFixedO2Graph(g)
        else:
            g=self.BuildO2Graph(g)
            
        g=self.BuildSO2Graph(g)
        
        if self.with_velocity:
            g=self.BuildDirectionsGraph(g)         
        
        return g         
    
    def AddType(self, threshold=10, art_vein=False):
        '''
        add type to graph nodes
        
        1: art, 2: vein, 3: capp 
        '''
        
        def setT(g, nodes, t):
            for i in nodes:
                g.node[i]['type']=t

        if not art_vein:
            r=np.array(self.g.GetRadii())
            types=np.ones_like(r).astype(int)
            types[types<threshold]=3
            self.g.SetTypes(types)
            
        else:
            print('--Assign types to branches, this might take a while ...')
            branches=self.g.GetPathesDict()
            if type(branches)==zip:
                branches=list(branches)  
            art_vein=np.random.choice([1,2], size=len(branches))
            for b, ty in zip(branches, art_vein):
                nodes=b[1]
                rad=[self.g.node[i]['r'] for i in nodes]
                meanrad=np.mean(rad)
                if meanrad<threshold:
                    setT(self.g, nodes, 3)
                else:
                    setT(self.g, nodes, ty)

    def AddVelocityFromFlow(self, scale_flow=1e-7, scale_diam=1e-3):
        '''
        Ouput flow should be in mm3/s
        '''
        flows=np.array(self.g.GetFlows())*scale_flow # mm3/s
        radii=np.array(self.g.GetRadii())*scale_diam # mm
        velocities=flows/(np.pi*(radii)**2) # mm/s
        
        # fix
        velocities[velocities>3.0]=3.0
        velocities[velocities<0.5]=0.5        
        velocities=velocities*1000 # um/s      
        
        for i, v in zip(self.g.GetNodes(), velocities):
            self.g.node[i]['velocity']=v

        for i, f in zip(self.g.GetNodes(), flows):
            self.g.node[i]['flow']=f
            
    def AddSo2FromPo2(self):
        po2=np.array([self.g.node[i]['po2'] for i in self.g.GetNodes()])
        so2=self.so2FROMpo2(po2)
        for i, s in zip(self.g.GetNodes(), so2):
            self.g.node[i]['so2']=s

            
    def BuildRegModel(self, datafile=None):
        
        '''
        build and save random forest regression models
        
        Input:
            -datafile: An xlcx file with sample in rows;
                        one column in the sheet is used as the prediction, 
                        whereas the left columns are used as features    
        '''
        
        if datafile is None:
            import os
            path=os.path.dirname(__file__)
            datafile=path+'/Measurments/Data.xlsx'
        
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
        
        # from A-V-C to A-C-V
        avc2acv={1:1, 2:3, 3:2}
        
        ####### build the regressino model based on existing measurments #######
        reg_model_PO2, reg_model_flow=self.BuildRegModel(datafile=datafile) 
           
        ####### collect information from graph ########
        diameter=np.array(g.GetRadii())*2.0 # diameter #micrometer
        types=np.array([g.node[i]['type'] for i in g.GetNodes()]) # type
        types=np.array([avc2acv[i] for i in types])
        depth=np.array(g.GetNodesPos())[:, 2] # depth    
       
        # if nan values appear
        diameter[np.isnan(diameter)]=4.0
        types[np.isnan(types)]=3
        depth[np.isnan(depth)]=100.0
        
        ######### Regression #########
        x_=[[a1, a2, a3] for a1, a2, a3 in zip(diameter,  types,  depth)]
        
        # infer po2 values
        values_PO2=reg_model_PO2.predict(x_) # pL/s
        for i, po2 in zip(g.GetNodes(), values_PO2):
            g.node[i]['po2']=po2
        
        # infer flow values and convert to velocities
        flows=reg_model_flow.predict(x_) # pL/s
        flows=flows*1e-7 # mm3/s
        diameter=diameter*1e-3 # mm
        velocities=flows/(np.pi*(diameter/2.0)**2) # mm/s
        
        # fix
        velocities[velocities>3.0]=3.0
        velocities[velocities<0.5]=0.5        
        velocities=velocities*1000 # um/s      
        
        for i, v in zip(g.GetNodes(), velocities):
            g.node[i]['velocity']=v
    
        for i, f in zip(g.GetNodes(), flows):
            g.node[i]['flow']=f
            
        return g
    
    def BuildFixedO2Graph(self, g):
        
        for i in g.GetNodes():
            g.node[i]['po2']=60.0
            g.node[i]['velocity']=5.0
            
        return g

    def BuildDirectionsGraph(self, g=None):
        
        '''
        Construct vlelocity unit vectors at each node in the graph .
        '''
        ret=1
        if g is None:
            g=self.g
            ret=0
        
        if not g.is_directed():
            print('-- Input grpah should be directed!')
        
        nodes=[i for i in g.GetNodes() if len(g.GetSuccessors(i))>0]
        p = np.array([g.node[i]['pos'] for i in nodes])
        x, y, z = p[:, 0], p[:, 1], p[:, 2] 
        
        nxt=[g.GetSuccessors(i) for i in nodes] 
        nxtp=np.array([np.mean([g.node[j]['pos'] for j in i], axis=0) for i in nxt])
        nxtx, nxty, nxtz = nxtp[:, 0], nxtp[:, 1], nxtp[:, 2] 
        
        dx=nxtx-x
        dy=nxty-y
        dz=nxtz-z
        
        for i, i1, i2, i3 in zip(nodes, dx, dy, dz):
            g.node[i]['dx']=i1
            g.node[i]['dy']=i2
            g.node[i]['dz']=i3
            
        #################################################
        nodes=[i for i in g.GetNodes() if len(g.GetSuccessors(i))==0 and len(g.GetPredecessors(i))>0]
        p = np.array([g.node[i]['pos'] for i in nodes])
        x, y, z = p[:, 0], p[:, 1], p[:, 2] 
        
        nxt=[g.GetPredecessors(i) for i in nodes] 
        nxtp=np.array([np.mean([g.node[j]['pos'] for j in i], axis=0) for i in nxt])
        nxtx, nxty, nxtz = nxtp[:, 0], nxtp[:, 1], nxtp[:, 2] 
        
        dx=x-nxtx
        dy=y-nxty
        dz=z-nxtz
        
        for i, i1, i2, i3 in zip(nodes, dx, dy, dz):
            g.node[i]['dx']=i1
            g.node[i]['dy']=i2
            g.node[i]['dz']=i3
        
        #################################################
        nodes=[i for i in g.GetNodes() if len(g.GetSuccessors(i))==0 and len(g.GetPredecessors(i))==0]
        for i in nodes:
            g.node[i]['dx']=0
            g.node[i]['dy']=0
            g.node[i]['dz']=0
            
        if ret:
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

    def Update(self):
        self.g=self.AddO2(self.g)
        
    def GetOuput(self):
        return self.g