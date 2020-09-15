# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 10:55:28 2020

@author: valero
"""
import matplotlib.pyplot as plt
        
import numpy as np
import os
import subprocess
import sys
import math
from numpy import loadtxt

import cv2
import pandas as pd
from scipy.signal import savgol_filter
from shapely.geometry import Point,Polygon
from geopandas import GeoDataFrame
import shapely.wkt
import pandas as pd
import sys
# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
sys.path.append(r'C:\Users\valero\Documents\Yeltsin 2.0\traffictongsim')
from Scripts import processing_yvc as yvc

ind_dir=r"C:/Users/valero/Documents/Yeltsin 2.0/ind-dataset/drone-dataset-tools-master/data"

data_ind=pd.read_csv(ind_dir+"/00_tracks.csv")

col_names=['Vehicle_ID','Frame_ID','Total_Frames','Global_Time','Local_X','Local_Y','Global_X','Global_Y','v_Length','v_Width','v_Class','v_Vel','v_Acc','Lane_ID','Preceeding','Following','Space_Hdwy','Time_Hdwy']

#Vehicle_ID	Frame_ID	Local_X	Local_Y	v_Vel	v_Acc	v_Length	v_Width	Lane_ID	Following	Preceeding	left_follower_ID	left_leader_ID	right_follower_ID	right_leader_ID	Vehicle_ID_Following	Local_X_Flowwing	Local_Y_Following	v_Vel_Following	v_Acc_Following	v_Length_Following	v_Width_Following	Lane_ID_Following	Vehicle_ID_Preceeding	Local_X_Preceeding	Local_Y_Preceeding	v_Vel_Preceeding	v_Acc_Preceeding	v_Length_Preceeding	v_Width_Precdeing	Lane_ID_Preceeding	Vehicle_ID_LeftLeader	Local_X_LeftLeader	Local_Y_LeftLeader	v_Vel_LeftLeader	v_Acc_LeftLeader	v_Length_LeftLeader	v_Width_LeftLeader	Lane_ID_LeftLeader	Vehicle_ID_LeftFollower	Local_X_LeftFollower	Local_Y_LeftFollower	v_Vel_LeftFollower	v_Acc_LeftFollower	v_Length_LeftFollower	v_Width_LeftFollower	Lane_ID_LeftFollower	Vehicle_ID_RightLeader	Local_X_RightLeader	Local_Y_RightLeader	v_Vel_RightLeader	v_Acc_RightLeader	v_Length_RightLeader	v_Width_RightLeader	Lane_ID_RightLeader	Vehicle_ID_RightFollower	Local_X_RightFollower	Local_Y_RightFollower	v_Vel_RightFollower	v_Acc_RightFollower	v_Length_RightFollower	v_Width_RightFollower	Lane_ID_RightFollower"	



#Geometry
geomet=pd.DataFrame()
yvc.getGeometry(geomet,ind_dir+'/00_background.png')


geomet=pd.read_csv('geometry.csv')
poligons=[]
k=0

geomet=geomet.T
geomet = geomet[geomet[0] != 0]
geo=[yvc.Geometry(indx) for indx in geomet.index]

for objgeo in geo:
    objgeo.polygon=shapely.wkt.loads(geomet[0][objgeo.name])

data_ngsim=pd.DataFrame() 
data_ngsim['Vehicle_ID']=data_ind['trackId']
data_ngsim['Frame_ID']=data_ind['frame']
data_ngsim['Local_X']=data_ind['xCenter']
data_ngsim['Local_Y']=data_ind['yCenter']
data_ngsim['v_Length']=data_ind['length']
data_ngsim['v_Width']=data_ind['width']
#data_ngsim['v_Class']=data_ind['length']
data_ngsim['v_Vel']=data_ind.apply(lambda x: (x['xVelocity']**2+x['yVelocity']**2)**.5, axis=1)
data_ngsim['v_Acc']=data_ind.apply(lambda x: (x['xAcceleration']**2+x['yAcceleration']**2)**.5, axis=1)

points=[Point(xy) for xy in zip(data_ngsim['Local_X'], -data_ngsim['Local_Y'])]
for objgeo in geo:
    objgeo.isLane=[]
    for point in points:
        objgeo.isLane.append(point.within(objgeo.polygon))
    data_ngsim[objgeo.name]=objgeo.isLane


