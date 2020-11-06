# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 11:04:45 2020

@author: a022927
"""

import os
import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
from scipy import signal

# import created modules 
#from calibration import *
#from display import *

# global variables
DATA_DIR = r'D:\LocalData\a022927\OneDrive - Alliance\projet_Lu\1st_work\data_NGSIM\data_101'

out_put_path = r''
factor_foot_2_meter = 0.3048

def smoothing_NGSIM_data():
    id_file = 0
    file_names = os.listdir(DATA_DIR)
    file_name = r'trajectories-' + file_names[id_file] + '.csv'
    file_path = DATA_DIR + '\\' + file_names[id_file] + '\\' + file_name
    df0= pd.read_csv(file_path)
    df0[['Local_X','Local_Y','v_Length','v_Width','v_Vel','v_Acc']] = df0[['Local_X','Local_Y','v_Length','v_Width','v_Vel','v_Acc']]*factor_foot_2_meter
    df_new = pd.DataFrame()
    for ID in df0.Vehicle_ID.unique():
        vehicle_data = df0.loc[df0.Vehicle_ID == ID]
        vehicle_data['Local_Y'] =  signal.savgol_filter(np.array(vehicle_data['Local_Y']),11, 1) #window size used for filtering, order of fitted polynomial
        vehicle_data['v_Vel'] = vehicle_data['Local_Y'].diff(periods=1)*10
        vehicle_data['Local_X'] =  signal.savgol_filter(np.array(vehicle_data['Local_X']),11, 1)
        df_new= df_new.append(vehicle_data.drop(vehicle_data.index[0]))
    df_new.to_csv(r'E:\code_for_Anna\NGSIM_smoothing_data\smoothing_first_15mins.csv', index_label=False, sep= ';' )
    

if __name__ == "__main__":
    
    # smoothing the position x,y and recalculate speed x, y 
    smoothing_NGSIM_data()
    
    # prepare all cars data and find all surounding vehicles
    df = pd.read_csv(r'E:\code_for_Anna\NGSIM_smoothing_data\smoothing_first_15mins.csv', sep=';')
    # choose all vehicles of class 2 : auto
    df = df.loc[df['Lane_ID'] != 6]
    df = df.loc[df['Lane_ID'] != 7]
    df = df.loc[df['Lane_ID'] != 8]
    df['Lane_ID'] = df['Lane_ID'].replace([5, 4, 3, 2, 1], [0, 1, 2, 3, 4])
    
    dataset_new = pd.DataFrame()
    
    dataset_new = pd.read_csv(r'E:\code_for_Anna\NGSIM_smoothing_data\all_cars_first_15mins_temps.csv', sep= ';' )
    
    for ID in df.loc[df['v_Class'] == 2].Vehicle_ID.unique()[355:]: 
        print(ID)
        print(list(df.loc[df['v_Class'] == 2].Vehicle_ID.unique()).index(ID))
        data_vehicle = df.loc[df['Vehicle_ID'] == ID]
        data_vehicle_new = pd.DataFrame()
        for i in np.arange(len(data_vehicle)): 
            row = data_vehicle.iloc[i]
            car_lane = row.Lane_ID
            Frame_ID = row.Frame_ID
            car_position_Y = row.Local_Y
            car_position_X = row.Local_X
            if car_lane != 0 or car_lane != 4 : 
                all_vehicle_Frame_ID = df.loc[df.Frame_ID == Frame_ID]
                all_vehicle_Frame_ID['relative_dis'] = all_vehicle_Frame_ID.Local_Y - car_position_Y
                
                # left lane vehciles
                all_vehicle_Frame_ID_left_lane = all_vehicle_Frame_ID.loc[all_vehicle_Frame_ID.Lane_ID == car_lane + 1]
                all_leaders_left = all_vehicle_Frame_ID_left_lane.loc[all_vehicle_Frame_ID_left_lane['relative_dis']>0]
                
                try : 
                    row['left_leader_ID'] = all_leaders_left.sort_values('relative_dis').iloc[0].Vehicle_ID
                except: 
                    row['left_leader_ID'] = None
                
    
                all_followers_left = all_vehicle_Frame_ID_left_lane.loc[all_vehicle_Frame_ID_left_lane['relative_dis']<0]
                
                try: 
                    row['left_follower_ID'] = all_followers_left.sort_values('relative_dis').iloc[-1].Vehicle_ID
                except: 
                    row['left_follower_ID'] = None
                
                # right lane vehciles
                all_vehicle_Frame_ID_right_lane = all_vehicle_Frame_ID.loc[all_vehicle_Frame_ID.Lane_ID == car_lane - 1]
                all_leaders = all_vehicle_Frame_ID_right_lane.loc[all_vehicle_Frame_ID_right_lane['relative_dis']>0]
                
                try : 
                    row['right_leader_ID'] = all_leaders.sort_values('relative_dis').iloc[0].Vehicle_ID
                except: 
                    row['right_leader_ID'] = None
                    
                all_followers = all_vehicle_Frame_ID_right_lane.loc[all_vehicle_Frame_ID_right_lane['relative_dis']<0]
                
                try : 
                    
                    row['right_follower_ID'] = all_followers.sort_values('relative_dis').iloc[-1].Vehicle_ID
                except: 
                    
                    row['right_follower_ID'] = None
                
                
                
            elif car_lane == 0: 
                all_vehicle_Frame_ID = df.loc[df.Frame_ID == Frame_ID]
                all_vehicle_Frame_ID['relative_dis'] = all_vehicle_Frame_ID.Local_Y - car_position_Y
                
                # left lane vehciles
                all_vehicle_Frame_ID_left_lane = all_vehicle_Frame_ID.loc[all_vehicle_Frame_ID.Lane_ID == car_lane + 1]
                
                 
                all_leaders_left = all_vehicle_Frame_ID_left_lane.loc[all_vehicle_Frame_ID_left_lane['relative_dis']>0]
                
                try : 
                    row['left_leader_ID'] = all_leaders_left.sort_values('relative_dis').iloc[0].Vehicle_ID
                
                except: 
                    row['left_leader_ID'] = None
                
    
                all_followers_left = all_vehicle_Frame_ID_left_lane.loc[all_vehicle_Frame_ID_left_lane['relative_dis']<0]
                
                try: 
                    row['left_follower_ID'] = all_followers_left.sort_values('relative_dis').iloc[-1].Vehicle_ID
                except: 
                    
                    row['left_follower_ID'] = None
                    
                all_vehicle_Frame_ID_right_lane = None
                row['right_leader_ID'] = None
                row['right_follower_ID'] = None
                
            elif car_lane == 4: 
                all_vehicle_Frame_ID = df.loc[df.Frame_ID == Frame_ID]
                all_vehicle_Frame_ID['relative_dis'] = all_vehicle_Frame_ID.Local_Y - car_position_Y
                
                # left lane vehciles
                all_vehicle_Frame_ID_left_lane = None
                row['left_leader_ID'] = None
                row['left_follower_ID'] = None
                
                # right lane vehciles
                all_vehicle_Frame_ID_right_lane = all_vehicle_Frame_ID.loc[all_vehicle_Frame_ID.Lane_ID == car_lane - 1]
                
                all_leaders = all_vehicle_Frame_ID_right_lane.loc[all_vehicle_Frame_ID_right_lane['relative_dis']>0]
                
                try : 
                    row['right_leader_ID'] = all_leaders.sort_values('relative_dis').iloc[0].Vehicle_ID
                except: 
                    row['right_leader_ID'] = None
                    
                all_followers = all_vehicle_Frame_ID_right_lane.loc[all_vehicle_Frame_ID_right_lane['relative_dis']<0]
                
                try : 
                    
                    row['right_follower_ID'] = all_followers.sort_values('relative_dis').iloc[-1].Vehicle_ID
                except: 
                    row['right_follower_ID'] = None
        
        
            data_vehicle_new = data_vehicle_new.append(row, ignore_index=True)
            
        dataset_new = dataset_new.append(data_vehicle_new)
        dataset_new.to_csv(r'E:\code_for_Anna\NGSIM_smoothing_data\all_cars_first_15mins_temps2.csv', index_label=False, sep= ';' )
        
    dataset_new.to_csv(r'E:\code_for_Anna\NGSIM_smoothing_data\all_cars_first_15mins.csv', index_label=False, sep= ';' )
    
    df_V_target= dataset_new[['Vehicle_ID', 'Frame_ID', 'Local_X', 'Local_Y', 'v_Vel', 'v_Acc', 
                              'v_Length', 'v_Width', 'Lane_ID', 
                              'Following', 'Preceeding', 
                              'left_follower_ID','left_leader_ID', 'right_follower_ID', 'right_leader_ID',]].copy()

    df_V_following = df[['Vehicle_ID', 'Frame_ID', 'Local_X', 'Local_Y', 'v_Vel', 'v_Acc', 
                              'v_Length', 'v_Width', 'Lane_ID', ]].copy()
    df_V_following.columns = ['Vehicle_ID_Following', 'Frame_ID', 'Local_X_Flowwing', 'Local_Y_Following', 'v_Vel_Following', 'v_Acc_Following', 
                              'v_Length_Following', 'v_Width_Following', 'Lane_ID_Following' ]
 
    df_V_Preceeding = df[['Vehicle_ID', 'Frame_ID', 'Local_X', 'Local_Y', 'v_Vel', 'v_Acc', 
                              'v_Length', 'v_Width', 'Lane_ID', ]].copy()
    df_V_Preceeding.columns = ['Vehicle_ID_Preceeding', 'Frame_ID', 'Local_X_Preceeding', 'Local_Y_Preceeding', 'v_Vel_Preceeding', 'v_Acc_Preceeding', 
                              'v_Length_Preceeding', 'v_Width_Precdeing', 'Lane_ID_Preceeding' ]

    df_V_LeftLeader = df[['Vehicle_ID', 'Frame_ID', 'Local_X', 'Local_Y', 'v_Vel', 'v_Acc', 
                              'v_Length', 'v_Width', 'Lane_ID', ]].copy()
    df_V_LeftLeader.columns = ['Vehicle_ID_LeftLeader', 'Frame_ID', 'Local_X_LeftLeader', 'Local_Y_LeftLeader',
                               'v_Vel_LeftLeader', 'v_Acc_LeftLeader', 
                              'v_Length_LeftLeader', 'v_Width_LeftLeader', 'Lane_ID_LeftLeader' ]    
    df_V_LeftFollower = df[['Vehicle_ID', 'Frame_ID', 'Local_X', 'Local_Y', 'v_Vel', 'v_Acc', 
                              'v_Length', 'v_Width', 'Lane_ID', ]].copy()
    df_V_LeftFollower.columns = ['Vehicle_ID_LeftFollower', 'Frame_ID', 'Local_X_LeftFollower', 'Local_Y_LeftFollower', 
                                 'v_Vel_LeftFollower', 'v_Acc_LeftFollower', 
                              'v_Length_LeftFollower', 'v_Width_LeftFollower', 'Lane_ID_LeftFollower' ]   
    df_V_RightLeader = df[['Vehicle_ID', 'Frame_ID', 'Local_X', 'Local_Y', 'v_Vel', 'v_Acc', 
                              'v_Length', 'v_Width', 'Lane_ID', ]].copy()
    df_V_RightLeader.columns = ['Vehicle_ID_RightLeader', 'Frame_ID', 'Local_X_RightLeader', 'Local_Y_RightLeader',
                               'v_Vel_RightLeader', 'v_Acc_RightLeader', 
                              'v_Length_RightLeader', 'v_Width_RightLeader', 'Lane_ID_RightLeader' ]  
    df_V_RightFollower = df[['Vehicle_ID', 'Frame_ID', 'Local_X', 'Local_Y', 'v_Vel', 'v_Acc', 
                              'v_Length', 'v_Width', 'Lane_ID', ]].copy()
    df_V_RightFollower.columns = ['Vehicle_ID_RightFollower', 'Frame_ID', 'Local_X_RightFollower', 'Local_Y_RightFollower', 
                                 'v_Vel_RightFollower', 'v_Acc_RightFollower', 
                              'v_Length_RightFollower', 'v_Width_RightFollower', 'Lane_ID_RightFollower' ]
    
    df_V_target = df_V_target.merge(df_V_following, left_on=['Frame_ID', 'Following'], right_on=['Frame_ID', 'Vehicle_ID_Following'], how= 'left')
    
    df_V_target = df_V_target.merge(df_V_Preceeding, left_on=['Frame_ID', 'Preceeding'], right_on=['Frame_ID', 'Vehicle_ID_Preceeding'], how= 'left')
    
    df_V_target = df_V_target.merge(df_V_LeftLeader, left_on=['Frame_ID', 'left_leader_ID'], right_on=['Frame_ID', 'Vehicle_ID_LeftLeader'], how= 'left')
    
    df_V_target = df_V_target.merge(df_V_LeftFollower, left_on=['Frame_ID', 'left_follower_ID'], right_on=['Frame_ID', 'Vehicle_ID_LeftFollower'], how= 'left')
    
    df_V_target = df_V_target.merge(df_V_RightLeader, left_on=['Frame_ID', 'right_leader_ID'], right_on=['Frame_ID', 'Vehicle_ID_RightLeader'], how= 'left')
    
    df_V_target = df_V_target.merge(df_V_RightFollower, left_on=['Frame_ID', 'right_follower_ID'], right_on=['Frame_ID', 'Vehicle_ID_RightFollower'], how= 'left')
    
    df_V_target.to_csv(r'E:\code_for_Anna\NGSIM_smoothing_data\all_cars_first_15mins_all_info.csv', index_label=False, sep= ';' )
#
    