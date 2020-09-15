# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 17:44:50 2020

@author: a022927
"""

import os
import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
from IPython.display import clear_output

# import created modules 
#from calibration import *
#from display import *

# global variables
DATA_DIR = r'D:\LocalData\a022927\OneDrive - Alliance\projet_Lu\1st_work\data_NGSIM\data_101'

out_put_path = r''
factor_foot_2_meter = 0.3048

def read_data(id_file):
    file_names = os.listdir(DATA_DIR)
    file_name = r'trajectories-' + file_names[id_file] + '.csv'
    file_path = DATA_DIR + '\\' + file_names[id_file] + '\\' + file_name
    df0= pd.read_csv(file_path)
    name_vehs = ['Vehicle_ID','Frame_ID','Total_Frames','Global_Time','Local_X','Local_Y','Global_X','Global_Y',
                   'v_Length','v_Width','v_Class','v_Vel','v_Acc','Lane_ID','Preceeding','Following','Space_Hdwy','Time_Hdwy']
    
    df0[['Local_X','Local_Y','v_Length','v_Width','v_Vel','v_Acc']] = df0[['Local_X','Local_Y','v_Length','v_Width','v_Vel','v_Acc']]*factor_foot_2_meter

    # choose all vehicles of class 2 : auto
    df_auto = df0.loc[df0['v_Class'] == 2].copy()
    df_auto = df_auto.loc[df_auto['Lane_ID'] != 6]
    df_auto = df_auto.loc[df_auto['Lane_ID'] != 7]
    df_auto = df_auto.loc[df_auto['Lane_ID'] != 8]
    
    df_V2= df_auto[['Vehicle_ID', 'Frame_ID', 'Local_X', 'Local_Y', 'v_Vel', 'v_Acc', 'Space_Hdwy', 'Time_Hdwy', 'v_Length', 'v_Width', 'Lane_ID']].copy()
    df_V2.columns = ['Vehicle_ID_V2', 'Frame_ID', 'Local_X_V2', 'Local_Y_V2', 'v_Vel_V2', 'v_Acc_V2', 'dis_Headway_V2', 'THW_V2', 'v_Length_V2', 'v_Width_V2', 'Lane_ID_V2']
    
    df_V1 = df_auto[['Vehicle_ID', 'Frame_ID', 'Following', 'Local_X', 'Local_Y', 'v_Vel', 'v_Acc', 'v_Length', 'v_Width',  'Lane_ID']].copy()
    df_V1.columns = ['Vehicle_ID_V1', 'Frame_ID', 'Following', 'Local_X_V1', 'Local_Y_V1', 'v_Vel_V1', 'v_Acc_V1', 'v_Length_V1','v_Width_V1', 'Lane_ID_V1']
    
    df_V1V2 = df_V1.merge(df_V2, left_on=['Frame_ID', 'Following'], right_on=['Frame_ID', 'Vehicle_ID_V2'], how= 'inner')
    
    dataset = df_V1V2[['Frame_ID', 'Vehicle_ID_V2', 'Vehicle_ID_V1', 'Local_X_V1', 'Local_Y_V1','Local_X_V2', 'Local_Y_V2', 'v_Vel_V1',
                       'v_Length_V1', 'v_Width_V1', 'v_Length_V2', 'v_Width_V2', 'v_Vel_V2', 'v_Acc_V2', 'THW_V2', 'Lane_ID_V2', 'dis_Headway_V2', 'Lane_ID_V1']]
    dataset.columns = ['Frame_ID', 'Vehicle_ID_V2', 'Vehicle_ID_V1', 'Local_X_V1','Local_Y_V1','Local_X_V2', 'Local_Y_V2', 'v_Vel_V1', 
                       'v_Length_V1', 'v_Width_V1', 'v_Length_V2', 'v_Width_V2', 'v_Vel_V2', 'v_Acc_V2', 'THW', 'Lane_ID_V2', 'dis_Headway_V2', 'Lane_ID_V1']
    
    dataset['delta_V'] = df_V1V2['v_Vel_V2'] - df_V1V2['v_Vel_V1']
    dataset['delta_Y'] = df_V1V2['Local_Y_V1'] - df_V1V2['Local_Y_V2'] - df_V1V2['v_Length_V1']
    
    dataset['delta_Y_2'] = df_V1V2['dis_Headway_V2'] - df_V1V2['v_Length_V1']
    dataset['delta_X'] = df_V1V2['Local_X_V1'] - df_V1V2['Local_X_V2']
    dataset['TTC']= dataset.apply(lambda x: None if (x['delta_V']< 0 or x['delta_V']==0) else x['delta_Y']/x['delta_V'], axis=1)
    
    dataset['TTCi']= dataset.apply(lambda x: None if (x['delta_V']==0) else x['delta_V']/x['delta_Y'], axis=1)
    dataset['TTCi_positif']= dataset.apply(lambda x: None if (x['delta_V']==0 or x['delta_V'] < 0) else x['delta_V']/x['delta_Y'], axis=1)
    
    dataset['THW_time'] = dataset.apply(lambda x: None if  x['v_Vel_V2']==0 else x['delta_Y']/x['v_Vel_V2'], axis=1)    
    return dataset, df0

def driver_indicators(dataset): 

    g = dataset[["Frame_ID", "v_Vel_V2"]].groupby('Frame_ID').mean()
    g.columns = ['mean_vel']
    dataset_merge = dataset.merge(g, on='Frame_ID')
    dataset_merge['diff_vel_mean'] = dataset_merge['v_Vel_V2'] - dataset_merge['mean_vel']
    
    vehicles = dataset_merge [['Vehicle_ID_V2', 'v_Acc_V2', 'THW_time',  'TTC', 'diff_vel_mean', 'v_Vel_V2','delta_Y', 'TTCi', 'TTCi_positif' ]].groupby('Vehicle_ID_V2').mean()
    vehicles.columns = ['v_Acc_V2_mean', 'THW_mean',  'TTC_mean', 'mean_diff_vel_mean', 'v_Vel_V2_mean', 'distance_mean', 'TTCi_mean', 'TTCi_positif_mean']
    
    vehicles['v__abs_Acc_V2_mean'] = abs(dataset_merge[[ 'Vehicle_ID_V2', 'v_Acc_V2'] ]).groupby('Vehicle_ID_V2').mean()
    
    std_values = dataset_merge [['Vehicle_ID_V2','v_Vel_V2', 'v_Acc_V2', 'THW_time',  'TTC', 'diff_vel_mean', 'TTCi', 'TTCi_positif']].groupby('Vehicle_ID_V2').std()
    std_values.columns = [ 'v_Vel_V2_std', 'v_abs_Acc_V2_std', 'THW_std',  'TTC_std', 'mean_diff_vel_std', 'TTCi_std', 'TTCi_positif_std']
    vehicles = vehicles.merge(std_values, on='Vehicle_ID_V2')
    
    nb_lanes = dataset.groupby('Vehicle_ID_V2')['Lane_ID_V2'].nunique().to_frame() - 1
    nb_lanes.columns = ['nb_lane_change']
    vehicles = vehicles.merge(nb_lanes, on='Vehicle_ID_V2')
    
    g = dataset[["Vehicle_ID_V2", "v_Vel_V2"]].groupby('Vehicle_ID_V2').max()
    g.columns = ['v_Vel_V2_max']
    vehicles = vehicles.merge(g, on='Vehicle_ID_V2')
    
    g = dataset[["Vehicle_ID_V2", "v_Vel_V2"]].groupby('Vehicle_ID_V2').min()
    g.columns = ['v_Vel_V2_min']
    vehicles = vehicles.merge(g, on='Vehicle_ID_V2')
    
    g = dataset[["Vehicle_ID_V2", "TTC"]].groupby('Vehicle_ID_V2').max()
    g.columns = ['TTC_max']
    vehicles = vehicles.merge(g, on='Vehicle_ID_V2')
    
    g = dataset[["Vehicle_ID_V2", "TTC"]].groupby('Vehicle_ID_V2').min()
    g.columns = ['TTC_min']
    vehicles = vehicles.merge(g, on='Vehicle_ID_V2')
    
    g = dataset[["Vehicle_ID_V2", "TTCi"]].groupby('Vehicle_ID_V2').max()
    g.columns = ['TTCi_max']
    vehicles = vehicles.merge(g, on='Vehicle_ID_V2')
    
    g = dataset[["Vehicle_ID_V2", "TTCi"]].groupby('Vehicle_ID_V2').min()
    g.columns = ['TTCi_min']
    vehicles = vehicles.merge(g, on='Vehicle_ID_V2')
    
    g = dataset[["Vehicle_ID_V2", "TTCi_positif"]].groupby('Vehicle_ID_V2').max()
    g.columns = ['TTCi_positif_max']
    vehicles = vehicles.merge(g, on='Vehicle_ID_V2')
    
    g = dataset[["Vehicle_ID_V2", "TTCi_positif"]].groupby('Vehicle_ID_V2').min()
    g.columns = ['TTCi_positif_min']
    vehicles = vehicles.merge(g, on='Vehicle_ID_V2')
    
    ratio_presence = dataset.groupby('Vehicle_ID_V2')['Lane_ID_V2'].sum()/dataset.groupby('Vehicle_ID_V2')['Lane_ID_V2'].count()/5 # most left lane = 1 
    ratio_presence.name = 'ratio_presence'
    vehicles = vehicles.join(ratio_presence, on='Vehicle_ID_V2')
    
    return vehicles

def OD_matrix_SUMO(dataset):
    df = dataset.copy()
    df['Lane_ID_V2'] = df['Lane_ID_V2'].replace([5, 4, 3, 2, 1], [0, 1, 2, 3, 4])
    df['Lane_ID_V1'] = df['Lane_ID_V1'].replace([5, 4, 3, 2, 1], [0, 1, 2, 3, 4])
    g = df[[ "Local_Y_V2", 'Vehicle_ID_V2']].groupby('Vehicle_ID_V2').min()
    g.columns = ['origin_Y']
    df_merged = df.merge(g, on='Vehicle_ID_V2')
    
    g = df[["Local_Y_V2", 'Vehicle_ID_V2']].groupby('Vehicle_ID_V2').max()
    g.columns = ['destination_Y']
    df_merged = df_merged.merge(g, on='Vehicle_ID_V2')
    
    g = df[[ "Local_X_V2", 'Vehicle_ID_V2']].groupby('Vehicle_ID_V2').min()
    g.columns = ['min_X']
    df_merged = df_merged.merge(g, on='Vehicle_ID_V2')
    
    g = df[["Local_X_V2", 'Vehicle_ID_V2']].groupby('Vehicle_ID_V2').max()
    g.columns = ['max_X']
    df_merged = df_merged.merge(g, on='Vehicle_ID_V2')
    
    g = df[["Frame_ID", "Vehicle_ID_V2"]].groupby('Vehicle_ID_V2').min()
    g.columns = ['in_time']
    df_merged = df_merged.merge(g, on='Vehicle_ID_V2')
    
    df_merged = df_merged[[ "Vehicle_ID_V2", 'in_time', 'origin_Y', 'destination_Y', 'min_X', 'max_X']].groupby('Vehicle_ID_V2').mean()
    
    trajectories = df.sort_values('Frame_ID').groupby('Vehicle_ID_V2')['Lane_ID_V2'].unique()
    trajectories.name = 'traj'
        
    df_merged = df_merged.join(trajectories, on='Vehicle_ID_V2')
    df_merged['in_time'] = df_merged['in_time']/10
    
    speed_depart = df.sort_values('Frame_ID').groupby('Vehicle_ID_V2')['v_Vel_V2'].first()
    speed_depart.name = 'speed_depart'
    speed_depart_merge_calib_maxspeed = result_ga.join(speed_depart, on='Vehicle_ID_V2')
    
    df_merged = df_merged.join(speed_depart, on='Vehicle_ID_V2')
    df_merged_copied = df_merged.copy()
    df_merged_copied['speed_depart'] = np.min(speed_depart_merge_calib_maxspeed[['v0', 'speed_depart']].values, axis=1)
    
    path = r'D:\LocalData\a022927\OneDrive - Alliance\projet_Lu\1st_work\NGSIM_simulation\data\data_processing\OD_matric_new.csv'
    df_merged_copied.to_csv(path)
    
    
def OD_matrix_SUMO_NEW(df_auto):
    df = df_auto.copy()
    df['Lane_ID'] = df['Lane_ID'].replace([5, 4, 3, 2, 1], [0, 1, 2, 3, 4])
    g = df[[ "Local_Y", 'Vehicle_ID']].groupby('Vehicle_ID').min()
    g.columns = ['origin_Y']
    df_merged = df.merge(g, on='Vehicle_ID')
    
    g = df[["Local_Y", 'Vehicle_ID']].groupby('Vehicle_ID').max()
    g.columns = ['destination_Y']
    df_merged = df_merged.merge(g, on='Vehicle_ID')
    
    g = df[[ "Local_X", 'Vehicle_ID']].groupby('Vehicle_ID').min()
    g.columns = ['min_X']
    df_merged = df_merged.merge(g, on='Vehicle_ID')
    
    g = df[["Local_X", 'Vehicle_ID']].groupby('Vehicle_ID').max()
    g.columns = ['max_X']
    df_merged = df_merged.merge(g, on='Vehicle_ID')
    
    g = df[["Frame_ID", "Vehicle_ID"]].groupby('Vehicle_ID').min()
    g.columns = ['in_time']
    df_merged = df_merged.merge(g, on='Vehicle_ID')
    
    df_merged = df_merged[[ "Vehicle_ID", 'in_time', 'origin_Y', 'destination_Y', 'min_X', 'max_X']].groupby('Vehicle_ID').mean()
    
    trajectories = df.sort_values('Frame_ID').groupby('Vehicle_ID')['Lane_ID'].unique()
    trajectories.name = 'traj'
        
    df_merged = df_merged.join(trajectories, on='Vehicle_ID')
    df_merged['in_time'] = df_merged['in_time']/10
    
    speed_depart = df.sort_values('Frame_ID').groupby('Vehicle_ID')['v_Vel'].first()
    speed_depart.name = 'speed_depart'
    df_merged = df_merged.join(speed_depart, on='Vehicle_ID')
#    speed_depart_merge_calib_maxspeed = result_ga.join(speed_depart, on='Vehicle_ID_V2')
#    
#    df_merged = df_merged.join(speed_depart, on='Vehicle_ID_V2')
#    df_merged_copied = df_merged.copy()
#    df_merged_copied['speed_depart'] = np.min(speed_depart_merge_calib_maxspeed[['v0', 'speed_depart']].values, axis=1)
    
    path = r'D:\LocalData\a022927\OneDrive - Alliance\projet_Lu\1st_work\NGSIM_simulation\data\data_processing\OD_matric_auto.csv'
    df_merged.to_csv(path)
    
    return df_merged
    
if __name__ == "__main__":
    
    i = 0
    dataset, df0 = read_data(i)
    data_vehicle = df0.loc[df0['Vehicle_ID'] == 35]
    # choose all vehicles of class 2 : auto
    df_auto = df0.loc[df0['v_Class'] == 2].copy()
    df_auto = df_auto.loc[df_auto['Lane_ID'] != 6]
    df_auto = df_auto.loc[df_auto['Lane_ID'] != 7]
    df_auto = df_auto.loc[df_auto['Lane_ID'] != 8]
    
    df = df_auto.copy()
    df['Lane_ID'] = df['Lane_ID'].replace([5, 4, 3, 2, 1], [0, 1, 2, 3, 4])
    g = df[[ "Local_Y", 'Vehicle_ID']].groupby('Vehicle_ID').min()
    g.columns = ['origin_Y']
    df_merged = df.merge(g, on='Vehicle_ID')
    
    g = df[["Local_Y", 'Vehicle_ID']].groupby('Vehicle_ID').max()
    g.columns = ['destination_Y']
    df_merged = df_merged.merge(g, on='Vehicle_ID')
    
    g = df[[ "Local_X", 'Vehicle_ID']].groupby('Vehicle_ID').min()
    g.columns = ['min_X']
    df_merged = df_merged.merge(g, on='Vehicle_ID')
    
    g = df[["Local_X", 'Vehicle_ID']].groupby('Vehicle_ID').max()
    g.columns = ['max_X']
    df_merged = df_merged.merge(g, on='Vehicle_ID')
    
    g = df[["Frame_ID", "Vehicle_ID"]].groupby('Vehicle_ID').min()
    g.columns = ['in_time']
    df_merged = df_merged.merge(g, on='Vehicle_ID')
    
    g = df[["Frame_ID", "Vehicle_ID"]].groupby('Vehicle_ID').max()
    g.columns = ['out_time']
    df_merged = df_merged.merge(g, on='Vehicle_ID')
    
    df_merged = df_merged[[ "Vehicle_ID", 'in_time', 'out_time','origin_Y', 'destination_Y', 'min_X', 'max_X']].groupby('Vehicle_ID').mean()
    
    trajectories = df.sort_values('Frame_ID').groupby('Vehicle_ID')['Lane_ID'].unique()
    trajectories.name = 'traj'
        
    df_merged = df_merged.join(trajectories, on='Vehicle_ID')
    df_merged['in_time'] = df_merged['in_time']/10
    
    df_merged['out_time'] = df_merged['out_time']/10
    
    speed_depart = df.sort_values('Frame_ID').groupby('Vehicle_ID')['v_Vel'].first()
    speed_depart.name = 'speed_depart'
    df_merged = df_merged.join(speed_depart, on='Vehicle_ID')
    
    speed_arrival = df.sort_values('Frame_ID').groupby('Vehicle_ID')['v_Vel'].last()
    speed_arrival.name = 'speed_arrival'
    df_merged = df_merged.join(speed_arrival, on='Vehicle_ID')
    
    speed_max = df.sort_values('Frame_ID').groupby('Vehicle_ID')['v_Vel'].max()
    speed_max.name = 'speed_max'
    df_merged = df_merged.join(speed_max, on='Vehicle_ID')
    df_merged['lane_changes_nb'] = df_merged.apply(lambda x: len(x['traj'])-1, axis=1)
    
    list_vehicle= []
    list_lane_change = []
    list_lane_change_frameID = []
    list_lane_change_position_longi =[]
    list_lane_change_position_lat =[]
    lane_change_info_2 = pd.DataFrame()
    
    for ID in df.Vehicle_ID.unique(): 
        data_vehicle = df.loc[df['Vehicle_ID'] == ID]
        lane_change_mask = data_vehicle.loc[data_vehicle['Lane_ID'].diff(periods=-1).fillna(0).abs()!=0]
        if not lane_change_mask.Lane_ID.empty : 
            list_vehicle.append(ID)
            lane_change_path = list(lane_change_mask.Lane_ID)
            list_lane_change_frameID.append(list(lane_change_mask.Frame_ID))
            list_lane_change.append(lane_change_path)
            list_lane_change_position_longi.append(list(lane_change_mask.Local_Y))
            list_lane_change_position_lat.append(list(lane_change_mask.Local_X))
            lane_change_info_2 = lane_change_info_2.append(lane_change_mask)
            
    lane_change_info = pd.DataFrame()
    lane_change_info['Vehicle_ID'] = list_vehicle
    lane_change_info['Lane_change_path']=list_lane_change
    lane_change_info['Lane_change_FrameID'] = list_lane_change_frameID
    lane_change_info['Lane_change_Position_longi'] = list_lane_change_position_longi
    lane_change_info['Lane_change_Position_lat'] = list_lane_change_position_lat
    lane_change_info['lane_changes_nb'] = lane_change_info.apply(lambda x: len(x['Lane_change_path'])-1, axis=1)

    time_cl_start = pd.DataFrame()
    time_cl_end = pd.DataFrame()
    for ID in lane_change_info_2.Vehicle_ID.unique():
        for i in np.arange(len(lane_change_info_2.loc[lane_change_info_2.Vehicle_ID == ID])): 
            lane_change_tau_data = lane_change_info_2.loc[lane_change_info_2.Vehicle_ID == ID].iloc[i]
            tau_lc = lane_change_tau_data.Frame_ID
            pos_lat_lc = lane_change_tau_data.Local_X
            data_vehicle = df.loc[df['Vehicle_ID'] == ID]
            data_vehicle['time'] = data_vehicle['Frame_ID'] - tau_lc
            data_vehicle['Local_X_relative'] = data_vehicle['Local_X'] - pos_lat_lc
            lane_change_data = data_vehicle.loc[data_vehicle['time'] > - 60 ]
            lane_change_data = lane_change_data.loc[data_vehicle['time'] < 60 ]
            traj_lc = np.array(lane_change_data.Local_X_relative)
            time = np.array(lane_change_data['time'])
            if lane_change_data.iloc[0].Local_X_relative > 0 : 
                time_cl_start_frame = lane_change_data.loc[lane_change_data.Local_X_relative - lane_change_data.v_Width/2.0 < 0].sort_values('Frame_ID').head(1)
                time_cl_start_frame['change_lane_direction'] = 'right'
                time_cl_start = time_cl_start.append(time_cl_start_frame)
                time_cl_end_frame = lane_change_data.loc[lane_change_data.Local_X_relative + lane_change_data.v_Width/2.0 > 0].sort_values('Frame_ID').tail(1)
                time_cl_end_frame['change_lane_direction'] = 'right'
                time_cl_end = time_cl_end.append(time_cl_end_frame)
            else: 
                time_cl_start_frame = lane_change_data.loc[lane_change_data.Local_X_relative + lane_change_data.v_Width/2.0 > 0].sort_values('Frame_ID').head(1)
                time_cl_start_frame['change_lane_direction'] = 'left'
                time_cl_start = time_cl_start.append(time_cl_start_frame)
                time_cl_end_frame = lane_change_data.loc[lane_change_data.Local_X_relative - lane_change_data.v_Width/2.0 < 0].sort_values('Frame_ID').tail(1)
                time_cl_end_frame['change_lane_direction'] = 'left'
                time_cl_end = time_cl_end.append(time_cl_end_frame)
    
    reward =  np.array(time_cl_end.v_Vel) - np.array(time_cl_start.v_Vel)
    reward_left = np.array(time_cl_end.loc[time_cl_end.change_lane_direction == 'left'].v_Vel) - np.array(time_cl_start.loc[time_cl_start.change_lane_direction == 'left'].v_Vel)
    reward_right = np.array(time_cl_end.loc[time_cl_end.change_lane_direction == 'right'].v_Vel) - np.array(time_cl_start.loc[time_cl_start.change_lane_direction == 'right'].v_Vel)
    
    
    
        
#    plt.hist(reward, alpha = 0.5, label= 'all', bins = 50)
#    plt.hist(reward_left, label = 'left_change', alpha = 0.5, bins=50)
#    plt.hist(reward_right, label = 'right_change', alpha = 0.5, bins=50)
#    plt.legend()
#    plt.title('Distribution speed gain (m/s) of lane changes')
    
    
    
#    has_lc_vehicles = lane_change_info.loc[lane_change_info['lane_changes_nb']>=0]
#    dataset_new = pd.DataFrame()
#    for ID in has_lc_vehicles.Vehicle_ID.unique():
#        if ID in [3,11,14,24,7,6,2,1]: 
#            pass 
#        else : 
#            data_vehicle = df.loc[df['Vehicle_ID'] == ID]
#            data_vehicle_new = pd.DataFrame()
#            print(ID)
#            for i in np.arange(len(data_vehicle)): 
#                row = data_vehicle.iloc[i]
#                car_lane = row.Lane_ID
#                Frame_ID = row.Frame_ID
#                car_position_Y = row.Local_Y
#                car_position_X = row.Local_X
#                
#                if car_lane != 0 or car_lane != 4 : 
#                    all_vehicle_Frame_ID = df_auto.loc[df_auto.Frame_ID == Frame_ID]
#                    all_vehicle_Frame_ID['relative_dis'] = all_vehicle_Frame_ID.Local_Y - car_position_Y
#                    
#                    # left lane vehciles
#                    all_vehicle_Frame_ID_left_lane = all_vehicle_Frame_ID.loc[all_vehicle_Frame_ID.Lane_ID == car_lane + 1]
#                    all_leaders_left = all_vehicle_Frame_ID_left_lane.loc[all_vehicle_Frame_ID_left_lane['relative_dis']>0]
#                    
#                    try : 
#                        row['left_leader_ID'] = all_leaders_left.sort_values('relative_dis').iloc[0].Vehicle_ID
#                    except: 
#                        row['left_leader_ID'] = None
#                    
#        
#                    all_followers_left = all_vehicle_Frame_ID_left_lane.loc[all_vehicle_Frame_ID_left_lane['relative_dis']<0]
#                    
#                    try: 
#                        row['left_follower_ID'] = all_followers_left.sort_values('relative_dis').iloc[-1].Vehicle_ID
#                    except: 
#                        row['left_follower_ID'] = None
#                    
#                    # right lane vehciles
#                    all_vehicle_Frame_ID_right_lane = all_vehicle_Frame_ID.loc[all_vehicle_Frame_ID.Lane_ID == car_lane - 1]
#                    
#         
#                    all_leaders = all_vehicle_Frame_ID_right_lane.loc[all_vehicle_Frame_ID_right_lane['relative_dis']>0]
#                    
#                    try : 
#                        row['right_leader_ID'] = all_leaders.sort_values('relative_dis').iloc[0].Vehicle_ID
#                    except: 
#                        row['right_leader_ID'] = None
#                        
#                    all_followers = all_vehicle_Frame_ID_right_lane.loc[all_vehicle_Frame_ID_right_lane['relative_dis']<0]
#                    
#                    try : 
#                        
#                        row['right_follower_ID'] = all_followers.sort_values('relative_dis').iloc[-1].Vehicle_ID
#                    except: 
#                        row['right_follower_ID'] = None
#                    
#                elif car_lane == 0: 
#                    all_vehicle_Frame_ID = df_auto.loc[df_auto.Frame_ID == Frame_ID]
#                    all_vehicle_Frame_ID['relative_dis'] = all_vehicle_Frame_ID.Local_Y - car_position_Y
#                    
#                    # left lane vehciles
#                    all_vehicle_Frame_ID_left_lane = all_vehicle_Frame_ID.loc[all_vehicle_Frame_ID.Lane_ID == car_lane + 1]
#                    
#                     
#                    all_leaders_left = all_vehicle_Frame_ID_left_lane.loc[all_vehicle_Frame_ID_left_lane['relative_dis']>0]
#                    
#                    try : 
#                        row['left_leader_ID'] = all_leaders_left.sort_values('relative_dis').iloc[0].Vehicle_ID
#                    
#                    except: 
#                        row['left_leader_ID'] = None
#                    
#        
#                    all_followers_left = all_vehicle_Frame_ID_left_lane.loc[all_vehicle_Frame_ID_left_lane['relative_dis']<0]
#                    
#                    try: 
#                        row['left_follower_ID'] = all_followers_left.sort_values('relative_dis').iloc[-1].Vehicle_ID
#                    except: 
#                        
#                        row['left_follower_ID'] = None
#                        
#                    all_vehicle_Frame_ID_right_lane = None
#                    row['right_leader_ID'] = None
#                    row['right_follower_ID'] = None
#                    
#                elif car_lane == 4: 
#                    all_vehicle_Frame_ID = df_auto.loc[df_auto.Frame_ID == Frame_ID]
#                    all_vehicle_Frame_ID['relative_dis'] = all_vehicle_Frame_ID.Local_Y - car_position_Y
#                    
#                    # left lane vehciles
#                    all_vehicle_Frame_ID_left_lane = None
#                    row['left_leader_ID'] = None
#                    row['left_follower_ID'] = None
#                    
#                    # right lane vehciles
#                    all_vehicle_Frame_ID_right_lane = all_vehicle_Frame_ID.loc[all_vehicle_Frame_ID.Lane_ID == car_lane - 1]
#                    
#         
#                    all_leaders = all_vehicle_Frame_ID_right_lane.loc[all_vehicle_Frame_ID_right_lane['relative_dis']>0]
#                    
#                    try : 
#                        row['right_leader_ID'] = all_leaders.sort_values('relative_dis').iloc[0].Vehicle_ID
#                    except: 
#                        row['right_leader_ID'] = None
#                        
#                    all_followers = all_vehicle_Frame_ID_right_lane.loc[all_vehicle_Frame_ID_right_lane['relative_dis']<0]
#                    
#                    try : 
#    
#                        row['right_follower_ID'] = all_followers.sort_values('relative_dis').iloc[-1].Vehicle_ID
#                    except: 
#                        row['right_follower_ID'] = None
#            
#            
#                data_vehicle_new = data_vehicle_new.append(row, ignore_index=True)
#                
#            dataset_new = dataset_new.append(data_vehicle_new)
#        
#    dataset_new.to_csv(r'C:\Users\A022927\OneDrive - Alliance\projet_Lu\7th_CL\data_lc_NGISM\all_cars_third_15mins.csv', index_label=False, sep= ';' )
#    
#    dataset_new = pd.read_csv(r'C:\Users\A022927\OneDrive - Alliance\projet_Lu\7th_CL\data_lc_NGISM\all_cars_third_15mins.csv', sep=';')
#    
#    df_V_target= dataset_new[['Vehicle_ID', 'Frame_ID', 'Local_X', 'Local_Y', 'v_Vel', 'v_Acc', 
#                              'v_Length', 'v_Width', 'Lane_ID', 
#                              'Following', 'Preceeding', 'left_follower_ID','left_leader_ID', 'right_follower_ID', 'right_leader_ID',]].copy()
#    
#    df_V_following = df[['Vehicle_ID', 'Frame_ID', 'Local_X', 'Local_Y', 'v_Vel', 'v_Acc', 
#                              'v_Length', 'v_Width', 'Lane_ID', ]].copy()
#    df_V_following.columns = ['Vehicle_ID_Following', 'Frame_ID', 'Local_X_Flowwing', 'Local_Y_Following', 'v_Vel_Following', 'v_Acc_Following', 
#                              'v_Length_Following', 'v_Width_Following', 'Lane_ID_Following' ]
# 
#    df_V_Preceeding = df[['Vehicle_ID', 'Frame_ID', 'Local_X', 'Local_Y', 'v_Vel', 'v_Acc', 
#                              'v_Length', 'v_Width', 'Lane_ID', ]].copy()
#    df_V_Preceeding.columns = ['Vehicle_ID_Preceeding', 'Frame_ID', 'Local_X_Preceeding', 'Local_Y_Preceeding', 'v_Vel_Preceeding', 'v_Acc_Preceeding', 
#                              'v_Length_Preceeding', 'v_Width_Precdeing', 'Lane_ID_Preceeding' ]
#
#    df_V_LeftLeader = df[['Vehicle_ID', 'Frame_ID', 'Local_X', 'Local_Y', 'v_Vel', 'v_Acc', 
#                              'v_Length', 'v_Width', 'Lane_ID', ]].copy()
#    df_V_LeftLeader.columns = ['Vehicle_ID_LeftLeader', 'Frame_ID', 'Local_X_LeftLeader', 'Local_Y_LeftLeader',
#                               'v_Vel_LeftLeader', 'v_Acc_LeftLeader', 
#                              'v_Length_LeftLeader', 'v_Width_LeftLeader', 'Lane_ID_LeftLeader' ]    
#    df_V_LeftFollower = df[['Vehicle_ID', 'Frame_ID', 'Local_X', 'Local_Y', 'v_Vel', 'v_Acc', 
#                              'v_Length', 'v_Width', 'Lane_ID', ]].copy()
#    df_V_LeftFollower.columns = ['Vehicle_ID_LeftFollower', 'Frame_ID', 'Local_X_LeftFollower', 'Local_Y_LeftFollower', 
#                                 'v_Vel_LeftFollower', 'v_Acc_LeftFollower', 
#                              'v_Length_LeftFollower', 'v_Width_LeftFollower', 'Lane_ID_LeftFollower' ]   
#    df_V_RightLeader = df[['Vehicle_ID', 'Frame_ID', 'Local_X', 'Local_Y', 'v_Vel', 'v_Acc', 
#                              'v_Length', 'v_Width', 'Lane_ID', ]].copy()
#    df_V_RightLeader.columns = ['Vehicle_ID_RightLeader', 'Frame_ID', 'Local_X_RightLeader', 'Local_Y_RightLeader',
#                               'v_Vel_RightLeader', 'v_Acc_RightLeader', 
#                              'v_Length_RightLeader', 'v_Width_RightLeader', 'Lane_ID_RightLeader' ]  
#    df_V_RightFollower = df[['Vehicle_ID', 'Frame_ID', 'Local_X', 'Local_Y', 'v_Vel', 'v_Acc', 
#                              'v_Length', 'v_Width', 'Lane_ID', ]].copy()
#    df_V_RightFollower.columns = ['Vehicle_ID_RightFollower', 'Frame_ID', 'Local_X_RightFollower', 'Local_Y_RightFollower', 
#                                 'v_Vel_RightFollower', 'v_Acc_RightFollower', 
#                              'v_Length_RightFollower', 'v_Width_RightFollower', 'Lane_ID_RightFollower' ]
#    
#    df_V_target = df_V_target.merge(df_V_following, left_on=['Frame_ID', 'Following'], right_on=['Frame_ID', 'Vehicle_ID_Following'], how= 'left')
#    
#    df_V_target = df_V_target.merge(df_V_Preceeding, left_on=['Frame_ID', 'Preceeding'], right_on=['Frame_ID', 'Vehicle_ID_Preceeding'], how= 'left')
#    
#    df_V_target = df_V_target.merge(df_V_LeftLeader, left_on=['Frame_ID', 'left_leader_ID'], right_on=['Frame_ID', 'Vehicle_ID_LeftLeader'], how= 'left')
#    
#    df_V_target = df_V_target.merge(df_V_LeftFollower, left_on=['Frame_ID', 'left_follower_ID'], right_on=['Frame_ID', 'Vehicle_ID_LeftFollower'], how= 'left')
#    
#    df_V_target = df_V_target.merge(df_V_RightLeader, left_on=['Frame_ID', 'right_leader_ID'], right_on=['Frame_ID', 'Vehicle_ID_RightLeader'], how= 'left')
#    
#    df_V_target = df_V_target.merge(df_V_RightFollower, left_on=['Frame_ID', 'right_follower_ID'], right_on=['Frame_ID', 'Vehicle_ID_RightFollower'], how= 'left')
#    
#    df_V_target.to_csv(r'C:\Users\A022927\OneDrive - Alliance\projet_Lu\7th_CL\data_lc_NGISM\all_cars_third_15mins_all_info.csv', index_label=False, sep= ';' )
    
#    dataset = dataset.sort_values(by=['Vehicle_ID_V2', 'Frame_ID'])
#    vehicles1 = driver_indicators(dataset)
#    vehicles2 = vehicles1.drop(dataset['Vehicle_ID_V2'].loc[dataset.delta_Y < 0].unique())
#    
#    g = dataset[[ "THW_time", 'Vehicle_ID_V2']].groupby('Vehicle_ID_V2').min()
#    g.columns = ['THW_min']
#    vehicles2 = vehicles2.merge(g, on='Vehicle_ID_V2')
#    
#    g = dataset[[ "THW_time", 'Vehicle_ID_V2']].groupby('Vehicle_ID_V2').max()
#    g.columns = ['THW_max']
#    vehicles2 = vehicles2.merge(g, on='Vehicle_ID_V2')
#    
#    g = dataset[[ "delta_Y", 'Vehicle_ID_V2']].groupby('Vehicle_ID_V2').min()
#    g.columns = ['distance_min']
#    vehicles2 = vehicles2.merge(g, on='Vehicle_ID_V2')
#    
#    g = dataset[[ "THW", 'Vehicle_ID_V2']].groupby('Vehicle_ID_V2').count() 
#    g.columns = ['nbr_frames']
#    veh2 = vehicles2.merge(g, on='Vehicle_ID_V2')
#    
#    g = dataset[[ "delta_Y", 'Vehicle_ID_V2']].groupby('Vehicle_ID_V2').min() 
#    g.columns = ['min_distance']
#    veh2 = veh2.merge(g, on='Vehicle_ID_V2')
#     
#    veh2_selec = veh2.loc[veh2.nbr_frames > 400] # 1533
#    veh2_selec.THW_min.hist(bins=50)
#    veh2_selec.hist(bins = 50)
#    
#    veh2_selec = veh2_selec.reset_index(level=0)
#    
#    veh2_selec['veh_meanTHW_stdTHW'] = veh2_selec['THW_mean'] - 1/2*veh2_selec['THW_std']
#    
#    veh_agressifs1= veh2_selec.sort_values('THW_mean').iloc[:50]  # <0.82
#    veh_agressifs2 =  veh2_selec.sort_values('THW_mean').iloc[50:100]  # <0.936
#
#    fig = plt.figure()
#    plt.scatter(veh2_selec.TTCi_positif_mean, veh2_selec.THW_mean)
#    plt.scatter(veh_agressifs1.TTCi_positif_mean, veh_agressifs1.THW_mean)
#    plt.scatter(veh_agressifs2.TTCi_positif_mean, veh_agressifs2.THW_mean)
#    plt.show()
#    
#    
#    veh_longs2 = veh2_selec.sort_values('THW_min').iloc[-100:-50] # > 1.656s 
#    veh_longs1 = veh2_selec.sort_values('THW_min').iloc[-50:] # > 1.84s 
#    
#    
#    veh_longs3 = veh2_selec.sort_values('THW_min').iloc[-150:-100] # > 1.518
#
#    
#    normal =  veh2_selec.drop(veh_agressifs1.index)
#    normal =  normal.drop(veh_agressifs2.index)
#    normal =  normal.drop(veh_longs1.index)
#    normal =  normal.drop(veh_longs2.index)
#
#    
#    fig = plt.figure()
#    plt.scatter(normal.THW_mean, normal.THW_min)
#    plt.scatter(veh_agressifs1.THW_mean, veh_agressifs1.THW_min, marker='d', label = 'ag1')
#    plt.scatter(veh_agressifs2.THW_mean, veh_agressifs2.THW_min, marker='^', label = 'ag2')
#    plt.scatter(veh_longs1.THW_mean, veh_longs1.THW_min, marker='*', label = 'inatt1' )
#    plt.scatter(veh_longs2.THW_mean, veh_longs2.THW_min, marker='P',label = 'inatt2')
#    plt.xlim ([0, 7])
#    plt.xlabel('THW_mean (s)', fontsize=15)
#    plt.ylabel('THW_min (s)', fontsize=15)
#    plt.legend()
#    plt.title('THW_mean THW_min distribution for all drivers', fontsize=15)
#    plt.show()
#    
#    
#    collision_info= pd.read_csv(r'E:\1_work\accidents_version2020\test_for_one_simulation\simulation_percentage5050\speed_info_collision_2.csv', sep=';')
#    fig = plt.figure()
#    plt.hist(collision_info.delta_speed*3.6, bins=20)
#    plt.xlabel('relative speed (km/h)')
#    plt.ylabel('number')
#    plt.title('144 crashes info : relative speed ')
#    plt.show()
    
    
    
    
#    path = r'E:\1_work\simulation_version14\calibration_longs\calibration_3rd_50_long'
#    
#    veh_longs3_dataset = dataset.merge( veh_longs3[['Vehicle_ID_V2']], on = 'Vehicle_ID_V2').sort_values(by=['Vehicle_ID_V2', 'Frame_ID'])
#    veh_longs3_dataset.to_csv(r'E:\1_work\calibration_data\long_100_150\veh_longs3_dataset.csv', index=False)

    
#    df_old100 = pd.read_csv(r'C:\Users\A022927\OneDrive - Alliance\projet_Lu\1st_work\NGSIM_simulation\simulation_initial_100 - Copie\data\result_cal_pos_100_agressifs.csv')    
#    df2_old = pd.read_csv(r'C:\Users\A022927\OneDrive - Alliance\projet_Lu\1st_work\NGSIM_simulation\simulation_initial_100 - Copie\data\result_ga_reaction_time6_agressifs.csv')
#    
#    df_old_merged = df_old100.merge(veh_agressifs, on='Vehicle_ID_V2', how='inner')  
#    
#    cal_speed = pd.read_csv(r'E:\cal_matlab_data\calibration_data\calibration_output\50ags_50longs_THW\cal_data\cal_pos.csv', header=None)
#    cal_speed.columns = ['Vehicle_ID_V2', 'v0', 'tau', 's0', 'action_time','accel','decel', 'goodness_speed', 'nb_iterations', 
#                         'rmse_pos', 'rmse_speed', 'rmpse_pos', 'rmpse_speed']
#    
#    
#    cal_pos_100100 = pd.read_csv(r'E:\cal_matlab_data\calibration_data\calibration_output\100ags_100longs_THW\cal_data\cal_position_end10.csv', header=None)
#    cal_pos_100100.columns = ['Vehicle_ID_V2', 'v0', 'tau', 's0', 'action_time','accel','decel', 'goodness_speed', 'nb_iterations', 
#                         'rmse_pos', 'rmse_speed', 'rmpse_pos', 'rmpse_speed']
#    
#    cal_pos_100100_1 = pd.read_csv(r'E:\cal_matlab_data\calibration_data\calibration_output\100ags_100longs_THW\cal_data\cal_position_end10_2.csv', header=None)
#    cal_pos_100100_1.columns = ['Vehicle_ID_V2', 'v0', 'tau', 's0', 'action_time','accel','decel', 'goodness_speed', 'nb_iterations', 
#                         'rmse_pos', 'rmse_speed', 'rmpse_pos', 'rmpse_speed']
    
#    cal_pos_100100 = pd.read_csv(r'E:\cal_matlab_data\calibration_data\calibration_output\100ags_100longs_THW\cal_data\cal_position_end10_3.csv', header=None)
#    cal_pos_100100.columns = ['Vehicle_ID_V2', 'v0', 'tau', 's0', 'action_time','accel','decel', 'goodness_speed', 'nb_iterations', 
#                         'rmse_pos', 'rmse_speed', 'rmpse_pos', 'rmpse_speed']
    
#    
#    cal_pos_100100 = pd.read_csv(r'E:\cal_matlab_data\calibration_data\calibration_long_time_reaction\cal_pos_long_ag_end3.csv', header=None)
#    cal_pos_100100.columns = ['Vehicle_ID_V2', 'v0', 'tau', 's0', 'action_time','accel','decel', 'goodness_speed', 'nb_iterations', 
#                         'rmse_pos', 'rmse_speed', 'rmpse_pos', 'rmpse_speed']
##    
#    veh_agressifs = veh2_selec.sort_values('THW_mean').iloc[50:100]  # <0.875s
#    veh_longs = veh2_selec.sort_values('THW_min').iloc[-100:-50] # > 1.84s 
#    veh_agressifs_dataset = dataset.merge( veh_agressifs[['Vehicle_ID_V2']], on = 'Vehicle_ID_V2').sort_values(by=['Vehicle_ID_V2', 'Frame_ID'])
#    veh_longs_dataset = dataset.merge( veh_longs[['Vehicle_ID_V2']], on = 'Vehicle_ID_V2').sort_values(by=['Vehicle_ID_V2', 'Frame_ID'])
#    veh_agressif_merged_speed = veh_agressifs[['Vehicle_ID_V2']].merge(cal_pos_100100, on = 'Vehicle_ID_V2')
#    path = r'E:\cal_matlab_data\calibration_data\calibration_long_time_reaction\50_100_new\ag_50_meanTHW_less.csv'
#    veh_agressif_merged_speed[['Vehicle_ID_V2', 'v0', 'tau', 's0', 'action_time','accel','decel',
#                               'rmse_pos', 'rmse_speed', 'rmpse_pos', 'rmpse_speed' ]].to_csv(path, sep=';',  index = False)
#    
#    
#    
#    cal_pos_long3 = pd.read_csv(r'E:\1_work\calibration_data\long_100_150\cal_pos_long_ag_end.csv', header=None)
#    veh_longs_merged_speed = veh_longs3[['Vehicle_ID_V2']].merge(cal_pos_long3, on = 'Vehicle_ID_V2')
#    path = r'E:\1_work\calibration_data\long_100_150\longs3_50_meanTHW_less.csv'
#    veh_longs_merged_speed[['Vehicle_ID_V2', 'v0', 'tau', 's0', 'action_time','accel','decel', 'rmse_pos', 
#                            'rmse_speed', 'rmpse_pos', 'rmpse_speed' ]].to_csv(path, sep=';', index = False)
#    
#    veh_longs_merged_speed[['action_time']] = np.round(veh_longs_merged_speed[['action_time' ]])/10
#    veh_longs_merged_speed[['rmpse_pos', 'rmpse_speed' ]] = veh_longs_merged_speed[['rmpse_pos', 'rmpse_speed' ]]*100
#    a = np.stack((veh_longs_merged_speed.mean().values, veh_longs_merged_speed.std().values, 
#                  veh_longs_merged_speed.median().values, veh_longs_merged_speed.quantile(0.25).values, veh_longs_merged_speed.quantile(0.75).values))
#    adf = pd.DataFrame(np.around(a, decimals=2).transpose())
#    adf.index = ['Vehicle_ID_V2', 'v0', 'tau', 's0', 'action_time','accel','decel', 'goodness_speed', 'nb_iterations', 
#                         'rmse_pos', 'rmse_speed', 'rmpse_pos', 'rmpse_speed']
#    adf.to_csv(r'E:\1_work\calibration_data\long_100_150\50long3_parameters.csv', sep=';')
    
#    veh_agressif_merged_speed[['action_time']] = np.round(veh_agressif_merged_speed[['action_time' ]])/10
#    veh_agressif_merged_speed[['rmpse_pos', 'rmpse_speed' ]] = veh_agressif_merged_speed[['rmpse_pos', 'rmpse_speed' ]]*100
#    a = np.stack((veh_agressif_merged_speed.mean().values, veh_agressif_merged_speed.std().values, 
#                  veh_agressif_merged_speed.median().values, veh_agressif_merged_speed.quantile(0.25).values, 
#                  veh_agressif_merged_speed.quantile(0.75).values))
#    adf = pd.DataFrame(np.around(a, decimals=2).transpose())
#    adf.index = ['Vehicle_ID_V2', 'v0', 'tau', 's0', 'action_time','accel','decel', 'goodness_speed', 'nb_iterations', 
#                         'rmse_pos', 'rmse_speed', 'rmpse_pos', 'rmpse_speed']
#    adf.to_csv(r'E:\cal_matlab_data\calibration_data\calibration_long_time_reaction\50_100_new\50ag_parameters_less.csv', sep=';')
##    
##    
##    
#    veh_agressifs2 = veh2_selec.sort_values('THW_mean').iloc[0:100]  # <0.875s
#    veh_longs2 = veh2_selec.sort_values('THW_min').iloc[-100:] # > 1.84s
#    veh_agressifs_dataset = dataset.merge( veh_agressifs2[['Vehicle_ID_V2']], on = 'Vehicle_ID_V2').sort_values(by=['Vehicle_ID_V2', 'Frame_ID'])
#    veh_longs_dataset = dataset.merge( veh_longs2[['Vehicle_ID_V2']], on = 'Vehicle_ID_V2').sort_values(by=['Vehicle_ID_V2', 'Frame_ID'])
#    veh_agressifs_dataset.to_csv(r'E:\cal_matlab_data\calibration_data\calibration_input\100longs_100ags_THW\veh_agressifs_dataset.csv', index=False)
#    veh_longs_dataset.to_csv(r'E:\cal_matlab_data\calibration_data\calibration_input\100longs_100ags_THW\veh_longs_dataset.csv', index=False)
#    
#    
#    veh_agressif_merged_speed = veh_agressifs2[['Vehicle_ID_V2']].merge(cal_pos_100100, on = 'Vehicle_ID_V2')
#    path = r'E:\cal_matlab_data\calibration_data\calibration_long_time_reaction\50_100_new\ag_100_meanTHW.csv'
#    veh_agressif_merged_speed[['Vehicle_ID_V2', 'v0', 'tau', 's0', 'action_time',
#                               'accel','decel', 'rmse_pos', 'rmse_speed', 'rmpse_pos', 'rmpse_speed' ]].to_csv(path, sep=';',  index = False)
#    veh_longs_merged_speed = veh_longs2[['Vehicle_ID_V2']].merge(cal_pos_100100, on = 'Vehicle_ID_V2')
#    path = r'E:\cal_matlab_data\calibration_data\calibration_long_time_reaction\50_100_new\longs_100_meanTHW.csv'
#    veh_longs_merged_speed[['Vehicle_ID_V2', 'v0', 'tau', 's0', 'action_time','accel','decel', 
#                            'rmse_pos', 'rmse_speed', 'rmpse_pos', 'rmpse_speed' ]].to_csv(path, sep=';', index = False)
#    
#    veh_longs_merged_speed[['action_time']] = np.round(veh_longs_merged_speed[['action_time' ]])/10
#    veh_longs_merged_speed[['rmpse_pos', 'rmpse_speed' ]] = veh_longs_merged_speed[['rmpse_pos', 'rmpse_speed' ]]*100
#    a = np.stack((veh_longs_merged_speed.mean().values, veh_longs_merged_speed.std().values, 
#                  veh_longs_merged_speed.median().values, veh_longs_merged_speed.quantile(0.25).values, veh_longs_merged_speed.quantile(0.75).values))
#    adf = pd.DataFrame(np.around(a, decimals=2).transpose())
#    adf.index = ['Vehicle_ID_V2', 'v0', 'tau', 's0', 'action_time','accel','decel', 'goodness_speed', 'nb_iterations', 
#                         'rmse_pos', 'rmse_speed', 'rmpse_pos', 'rmpse_speed']
#    adf.to_csv(r'E:\cal_matlab_data\calibration_data\calibration_long_time_reaction\50_100_new\100long_parameters.csv', sep=';')
#    
#    veh_agressif_merged_speed[['action_time']] = np.round(veh_agressif_merged_speed[['action_time' ]])/10
#    veh_agressif_merged_speed[['rmpse_pos', 'rmpse_speed' ]] = veh_agressif_merged_speed[['rmpse_pos', 'rmpse_speed' ]]*100
#    a = np.stack((veh_agressif_merged_speed.mean().values, veh_agressif_merged_speed.std().values, 
#                  veh_agressif_merged_speed.median().values, veh_agressif_merged_speed.quantile(0.25).values, 
#                  veh_agressif_merged_speed.quantile(0.75).values))
#    adf = pd.DataFrame(np.around(a, decimals=2).transpose())
#    adf.index = ['Vehicle_ID_V2', 'v0', 'tau', 's0', 'action_time','accel','decel', 'goodness_speed', 'nb_iterations', 
#                         'rmse_pos', 'rmse_speed', 'rmpse_pos', 'rmpse_speed']
#    
#    adf.to_csv(r'E:\cal_matlab_data\calibration_data\calibration_long_time_reaction\50_100_new\100ag_parameters.csv', sep=';')
#    
#    
#    
#    cal_100_ag = pd.read_csv(r'E:\cal_matlab_data\50_100_drivers\ag_100_meanTHW.csv', sep=';')
#    cal_50_ag = pd.read_csv(r'E:\cal_matlab_data\50_100_drivers\ag_50_meanTHW.csv', sep=';')
#    cal_50_ag_less = cal_100_ag.merge(cal_50_ag, how='outer')
#    
#    veh_agressifs_dataset = dataset.merge( veh_agressifs[['Vehicle_ID_V2']], on = 'Vehicle_ID_V2').sort_values(by=['Vehicle_ID_V2', 'Frame_ID'])
#    veh_longs_dataset = dataset.merge( veh_longs[['Vehicle_ID_V2']], on = 'Vehicle_ID_V2').sort_values(by=['Vehicle_ID_V2', 'Frame_ID'])
#    veh_agressif_merged_speed = veh_agressifs[['Vehicle_ID_V2']].merge(cal_100_ag, on = 'Vehicle_ID_V2')
#    path = r'E:\cal_matlab_data\calibration_data\calibration_long_time_reaction\50_100_new\ag_50_meanTHW_less.csv'
#    veh_agressif_merged_speed[['Vehicle_ID_V2', 'v0', 'tau', 's0', 'action_time','accel','decel',
#                               'rmse_pos', 'rmse_speed', 'rmpse_pos', 'rmpse_speed' ]].to_csv(path, sep=';',  index = False)
#    veh_agressif_merged_speed[['action_time']] = np.round(veh_agressif_merged_speed[['action_time' ]])/10
#    veh_agressif_merged_speed[['rmpse_pos', 'rmpse_speed' ]] = veh_agressif_merged_speed[['rmpse_pos', 'rmpse_speed' ]]*100
#    a = np.stack((veh_agressif_merged_speed.mean().values, veh_agressif_merged_speed.std().values, 
#                  veh_agressif_merged_speed.median().values, veh_agressif_merged_speed.quantile(0.25).values, 
#                  veh_agressif_merged_speed.quantile(0.75).values))
#    adf = pd.DataFrame(np.around(a, decimals=2).transpose())
#    adf.index = ['Vehicle_ID_V2', 'v0', 'tau', 's0', 'action_time','accel','decel',
#                         'rmse_pos', 'rmse_speed', 'rmpse_pos', 'rmpse_speed']
#    adf.to_csv(r'E:\cal_matlab_data\calibration_data\calibration_long_time_reaction\50_100_new\50ag_parameters_less.csv', sep=';')
##    
#    veh_agressifs2 = veh2_selec.sort_values('THW_mean').iloc[0:500].sort_values('veh_meanTHW_stdTHW').iloc[0:100]  # <0.875s
##    
#    veh_agressifs3 = veh2_selec.sort_values('THW_mean').iloc[0:500].sort_values('veh_meanTHW_stdTHW').iloc[0:50]  # <0.875s
    
#    cal_all_pos_rspme = pd.read_csv(r'E:\cal_matlab_data\calibration_data\calibration_output\all_drivers\cal_data\all_drivers_cal_pos_rmpse.csv', 
#                                    sep = ',', header=None)
#    cal_all_pos_rspme.columns = ['Vehicle_ID_V2', 'v0', 'tau', 's0', 'action_time','accel','decel', 'goodness_speed', 'nb_iterations', 
#                         'rmse_pos', 'rmse_speed', 'rmpse_pos', 'rmpse_speed']
#    
    
#    fig = plt.figure()
#    plt.scatter(veh2_selec.THW_mean, veh2_selec.THW_std)
#    plt.show()
#    
#    fig = plt.figure()
#    plt.scatter(veh2_selec.TTCi_positif_mean, veh2_selec.THW_min)
#    plt.scatter(veh_agressifs.TTCi_positif_mean, veh_agressifs.THW_min)
#    plt.scatter(veh_longs.TTCi_positif_mean, veh_longs.THW_min)
#    plt.show()
    
#    h = 2
#    w = 2
#    fig=plt.figure(figsize=(8, 8))
#    ax = fig.add_subplot(1, 1, 1)
#    data_vehicle = dataset.loc[dataset['Vehicle_ID_V2'] == 2750].sort_values('Frame_ID')
#    data_vehicle = data_vehicle.loc[data_vehicle.Frame_ID > 8300]
#    data_vehicle = data_vehicle.loc[data_vehicle.Frame_ID < 8400]
#    ax.set_title('Display of vehicle ' + str(2483) + ' time sequential data and the leading vehicule speed')
#    ax.axis('off')
#    fig.add_subplot(h, w, 1)
#    #plt.scatter( data_vehicle['Frame_ID'], data_vehicle['v_Vel_V1']*3.6,  s=5, c ='#1f77b4', label= 'speed V1')
#    plt.scatter(data_vehicle['Frame_ID'], data_vehicle['v_Vel_V2']*3.6, s=5, color='#ff7f0e', label= 'speed V2')
#    #plt.plot(data_vehicle['Frame_ID'], data_vehicle['v_Vel_V1']*3.6, c='#1f77b4', label= 'speed V1')
#    plt.plot(data_vehicle['Frame_ID'], data_vehicle['v_Vel_V2']*3.6,  color='#ff7f0e', label='speed V2')
#    plt.ylabel('speed (km/h)')
#    plt.legend()
#    
#    fig.add_subplot(h, w, 2)
#    plt.scatter(data_vehicle['Frame_ID'], data_vehicle['v_Acc_V2'],  s=5, label= 'acceleration')
#    plt.plot(data_vehicle['Frame_ID'], data_vehicle['v_Acc_V2'], label= 'acceleration')
#    plt.ylabel('accel (m/s2)')
#    plt.legend()
#    
#    fig.add_subplot(h, w, 3)
#    plt.scatter(data_vehicle['Frame_ID'], data_vehicle['delta_X'],  s=5)
#    plt.plot( data_vehicle['Frame_ID'], data_vehicle['delta_X'], label= 'delta_X')
#    plt.ylabel('distance X (m)')
#    plt.legend()
#    
#    fig.add_subplot(h, w, 4)
#    plt.scatter(data_vehicle['Frame_ID'], data_vehicle['THW_time'],  s=5, label= 'THW',  color='#ff7f0e')
#    plt.plot( data_vehicle['Frame_ID'], data_vehicle['THW_time'], label= 'THW',  color='#ff7f0e')
#    plt.ylabel('THW (s)')
#    plt.legend()
#    
#    
#    
#    fig.add_subplot(h, w, 3)
#    plt.scatter(data_vehicle['Frame_ID'], data_vehicle['Local_X_V2'],  s=5, label= 'position V2')
#    plt.scatter(data_vehicle['Frame_ID'], data_vehicle['Local_X_V1'],  s=5,  label= 'position V1')
    
#    plt.plot(data_vehicle['Frame_ID'], data_vehicle['Local_X_V2'])
#    plt.plot(data_vehicle['Frame_ID'], data_vehicle['Local_X_V1'])
#    plt.ylabel('leader V1 and follower V2 position X (m)')
#
#    fig.add_subplot(h, w, 2)
#    plt.scatter(data_vehicle['Frame_ID'], data_vehicle['v_Acc_V2'],  s=5, label= 'Accel V2', color='#ff7f0e')
#
#    plt.plot(data_vehicle['Frame_ID'], data_vehicle['v_Acc_V2'], label= 'Accel V2',  color='#ff7f0e')
#    plt.ylabel('follower acceleration (m/s2)')
#    
#    fig.add_subplot(h, w, 3)
#    plt.scatter(data_vehicle['Frame_ID'], data_vehicle['delta_Y'],  s=5, label= 'distance', color='#ff7f0e')
#    plt.plot( data_vehicle['Frame_ID'], data_vehicle['delta_Y'], label= 'distance',  color='#ff7f0e')
#    plt.ylabel('distance (m)')
#    plt.legend()
#    
#    fig.add_subplot(h, w, 5)
#    plt.scatter(data_vehicle['Frame_ID'], data_vehicle['delta_X'],  s=5)
#    plt.plot( data_vehicle['Frame_ID'], data_vehicle['delta_X'], label= 'delta_X')
#    plt.ylabel('distance X (m)')
#    plt.legend()
#    
#    
#
#    
    
    
#    dataVeh = dataset.loc[dataset.Vehicle_ID_V2 == 1216]
#    plt.plot(dataVeh.Frame_ID, dataVeh.v_Vel_V2)
#    plt.show()
#    
#    
#    dataVeh = dataset.loc[dataset.Vehicle_ID_V2 == 1216]
#    plt.plot(dataVeh.Frame_ID, dataVeh.delta_Y)
#    plt.show()
#    
#    
#    dataVeh = dataset.loc[dataset.Vehicle_ID_V2 == 1216]
#    plt.plot(dataVeh.Frame_ID, dataVeh.Local_Y_V1)
#    plt.show()
#    
#    
#    dataVeh = dataset.loc[dataset.Vehicle_ID_V2 == 1216]
#    plt.plot(dataVeh.Frame_ID, dataVeh.v_Vel_V1)
#    plt.show()
#    
#    
#    
#    dataVeh = dataset.loc[dataset.Vehicle_ID_V2 == 1216]
#    plt.plot(dataVeh.sort_values('Frame_ID').Frame_ID, dataVeh.sort_values('Frame_ID').THW_time)
#    plt.show()
#    
#    
#    dataVeh = dataset.loc[dataset.Vehicle_ID_V2 == 1216]
#    plt.plot(dataVeh.sort_values('Frame_ID').Frame_ID, dataVeh.sort_values('Frame_ID').THW_time)
#    plt.show()
    
    
#    cal_position = pd.read_csv(r'E:\1st_work_data_matlab\cal_position_all_15min.csv', header=None)
#    cal_position.columns = ['Vehicle_ID_V2', 'v0', 'tau', 's0', 'action_time','accel','decel', 'error_RMSE_pos', 'nb_iter', 'rmse_speed','rmpse_pos', 'rmpse_speed']
#    veh_agressif_merged_pos = veh_agressifs2[['Vehicle_ID_V2']].merge(cal_position, on = 'Vehicle_ID_V2' )
#
#    path = r'E:\1st_work_data_matlab\data_agressive\ag_100_meanTHW_0.5stdTHW.csv'
#    veh_agressif_merged_pos[['Vehicle_ID_V2', 'v0', 'tau', 's0', 'action_time','accel','decel', 'error_RMSE_pos' ]].to_csv(path, index = False)
#    
#    
#    veh_agressif_merged_pos2 = veh_agressifs3[['Vehicle_ID_V2']].merge(cal_position, on = 'Vehicle_ID_V2' )
#
#    path = r'E:\1st_work_data_matlab\data_agressive\ag_50_meanTHW_0.5stdTHW.csv'
#    veh_agressif_merged_pos2[['Vehicle_ID_V2', 'v0', 'tau', 's0', 'action_time','accel','decel', 'error_RMSE_pos' ]].to_csv(path, index = False)
#    
#        
#    dataVeh = dataset.loc[dataset.Vehicle_ID_V2 == 704]
#    fig = plt.figure()
#    plt.plot(dataVeh.sort_values('Frame_ID').Frame_ID, dataVeh.sort_values('Frame_ID').THW_time)
#    plt.show()
#    
#    dataVeh = dataset.loc[dataset.Vehicle_ID_V2 == 704]
#    fig = plt.figure()
#    plt.plot(dataVeh.sort_values('Frame_ID').Frame_ID, dataVeh.sort_values('Frame_ID').Local_X_V2)
#    plt.show()
#    
#    dataVeh = dataset.loc[dataset.Vehicle_ID_V2 == 699]
#    fig = plt.figure()
#    plt.plot(dataVeh.sort_values('Frame_ID').Frame_ID, dataVeh.sort_values('Frame_ID').THW_time)
#    plt.show()
    
    
#    veh_agressif_merged_pos = veh_agressifs[['Vehicle_ID_V2']].merge(cal_position, on = 'Vehicle_ID_V2' )
#    veh_longs_merged_pos = veh_longs[['Vehicle_ID_V2']].merge(cal_position, on = 'Vehicle_ID_V2' )
#    
#    path = r'D:\LocalData\a022927\OneDrive - Alliance\projet_Lu\1st_work\NGSIM_simulation\data\data_processing\result_ga_reaction_time6_agressifs.csv'
#    veh_agressif_merged_pos[['Vehicle_ID_V2', 'v0', 'tau', 's0', 'action_time','accel','decel', 'error_RMSE_pos' ]].to_csv(path, index = False)
#    
#    path = r'D:\LocalData\a022927\OneDrive - Alliance\projet_Lu\1st_work\NGSIM_simulation\data\data_processing\result_ga_reaction_time6_longs.csv'
#    veh_longs_merged_pos[['Vehicle_ID_V2', 'v0', 'tau', 's0', 'action_time','accel','decel', 'error_RMSE_pos',]].to_csv(path, index = False)
#    
#    veh_agressifs = veh2_selec.sort_values('THW_mean').iloc[0:50]  # <0.875s
#    
#    veh_longs = veh2_selec.sort_values('THW_min').iloc[-50:] # > 1.84s 
#    
#    veh_agressifs_dataset = dataset.merge( veh_agressifs[['Vehicle_ID_V2']], on = 'Vehicle_ID_V2').sort_values(by=['Vehicle_ID_V2', 'Frame_ID'])
#    
#    veh_longs_dataset = dataset.merge( veh_longs[['Vehicle_ID_V2']], on = 'Vehicle_ID_V2').sort_values(by=['Vehicle_ID_V2', 'Frame_ID'])
#    dataset.sort_values(by=['Vehicle_ID_V2', 'Frame_ID']).to_csv(r'E:\cal_matlab_data\dataset.csv')

#    nb_lane_change_1 = vehicles2['nb_lane_change'].loc[vehicles2['nb_lane_change'] == 1].count()
#    nb_lane_change_2 = vehicles2['nb_lane_change'].loc[vehicles2['nb_lane_change'] == 2].count()
#    nb_lane_change_3 = vehicles2['nb_lane_change'].loc[vehicles2['nb_lane_change'] == 3].count()
#    nb_lane_change_4 = vehicles2['nb_lane_change'].loc[vehicles2['nb_lane_change'] == 4].count()
#    
#    vehicles2_masked = vehicles2.mask(vehicles2.TTC_mean > 800)
#    
#    vehicles2_masked = vehicles2_masked.mask(vehicles2_masked.THW_mean > 10)
#    
#    
#    indices = ['v_Acc_V2_mean', 'THW_mean', 'TTC_mean', 'mean_diff_vel_mean',
#       'v_Vel_V2_mean', 'v_Vel_V2_std', 'v_Acc_V2_std', 'THW_std', 'TTC_std',
#       'mean_diff_vel_std', 'nb_lane_change', 'max_vel_V2', 'min_vel_V2',
#       'ratio_presence']
#    dataset_columns = ['Frame_ID', 'Vehicle_ID_V2', 'Vehicle_ID_V1', 'Local_X_V1',
#       'Local_Y_V1', 'Local_X_V2', 'Local_Y_V2', 'v_Vel_V1', 'v_Length_V1',
#       'v_Width_V1', 'v_Length_V2', 'v_Width_V2', 'v_Vel_V2', 'v_Acc_V2',
#       'THW', 'Lane_ID_V2', 'dis_Headway_V2', 'Lane_ID_V1', 'delta_V',
#       'delta_Y', 'delta_Y_2', 'delta_X', 'TTC']
#    
#    vehicles_ag_THW = vehicles2.loc[(vehicles2.THW_mean < 1.2 )] # 93
#    
#    all_vehicles_in_plot = dataset.loc[dataset['Vehicle_ID_V2'].isin(vehicles_ag_THW.index)]
##    all_vehicles_id = set(list(all_vehicles_in_plot['Vehicle_ID_V2'].unique()) + list(all_vehicles_in_plot['Vehicle_ID_V1'].unique()))
#    
#    vehicles_ag_TTC_diffVel_THW = vehicles2.loc[ (vehicles2.mean_diff_vel_mean > 2 ) & (vehicles2.THW_mean < 1.4 ) ] # 184
#    list_parameters = []
#    list_error = []
#    list_error_percent = []
#    list_veh_id = []
#    for i in  vehicles_ag_TTC_diffVel_THW.index: 
#        list_veh_id.append(i)
#        np.random.seed(28)
#        data_vehicle = dataset.loc[dataset['Vehicle_ID_V2'] == i].sort_values('Frame_ID')
#        path = r'D:\LocalData\a022927\OneDrive - Alliance\projet_Lu\1st_work\out_put_calibration\vehicles_ag_TTC_diffVel_THW2'
#
#        display_vehicle_traces(data_vehicle, i, path)
#        
#        X = np.array(data_vehicle[['v_Vel_V2', 'delta_V', 'delta_Y']])
#        Y = np.array(data_vehicle[['v_Acc_V2']]).flatten()
#
#        x0 = np.random.rand(1,3)[0] + 0.1
#        x0[0] = x0[0] + X[:, 0][0]
#        bounds = ([X[:, 0][0], 0.1, 0] , [40, 2.5, 10])
#        #        res = least_squares(error_3_parameters, x0=x0,  method='lm')
#        #        res = least_squares(error_3_parameters, x0, args=(Y, X), method='lm')
#        res = least_squares(error_3_parameters, x0=x0, args=(X, Y), bounds=bounds)
#        print (res.x, res.fun)
#        list_parameters.append(res.x)
#        list_error.append(res.fun)
#        a_simul = IDM_3_parameters(res.x[0], res.x[1], res.x[2], X)
##        print(a_simul)
#        print('error', np.mean(np.abs(Y - a_simul)) )
#        list_error_percent.append(np.mean(np.abs(Y - a_simul)))
#
#        fig = plt.figure()
#        plt.scatter( np.arange(len(X)), a_simul,  s=5)
#        plt.scatter( np.arange(len(X)), Y,  s=5)
#        plt.plot( a_simul , label= 'calibrated a by simulation')
#        plt.plot( np.array(Y), label= 'real accel')
#        
#        plt.ylabel('acceleration')
#        plt.legend()
#        file_name = path + '\\' + 'vehicle' + str(i) + '_accel_cal.png'
#        plt.savefig(file_name)
#        plt.close(fig)
#        
#        a_simul_cumsum = np.insert(np.cumsum(a_simul)/10, 0, 0.0)
#        a_simul_cumsum = np.insert(np.cumsum(a_simul)/10, 0, 0.0)
#        
#        fig = plt.figure()
#        plt.scatter( np.arange(len(X)), X[0][0] + a_simul_cumsum[:-1],  s=5)
#        plt.scatter( np.arange(len(X)), X[:, 0],  s=5)
#        plt.plot( X[0][0] + a_simul_cumsum[:-1] , label= 'calibrated a by simulation')
#        plt.plot( X[:, 0] , label= 'real speed')
#        plt.ylabel('speed')
#        plt.legend()
#        file_name = path + '\\' + 'vehicle' + str(i) + '_speed_cal.png'
#        plt.savefig(file_name)
#        plt.close(fig)
#        
#    a = pd.DataFrame(list_parameters)
#    a.index = list_veh_id
#    a.columns = ['v0', 'tau', 's0']
#    error = np.array(list_error)
#    a['error'] = error.flatten()
#    a.to_csv(r'D:\LocalData\a022927\OneDrive - Alliance\projet_Lu\1st_work\NGSIM_simulation\data\data_processing\vehicles_ag_TTC_diffVel_THW2.csv')
#        
#        X = np.array(dataset.loc[dataset['Vehicle_ID_V2'] == i].sort_values('Frame_ID')[['v_Vel_V2', 'delta_V', 'delta_Y']])
#        Y = dataset.loc[dataset['Vehicle_ID_V2'] == i].sort_values('Frame_ID')['v_Acc_V2']
    
#    vehicles_ag_TTC_Lanes_THW = vehicles.loc[ (vehicles.nb_lanes > 1 ) & (vehicles.THW < 1.5 )] # 41
#    
#    vehicles2.v_Acc_V2_mean.hist(bins=50)
#    vehicles2.THW_min.hist(bins=50)
#    result_ga = pd.read_csv(r'D:\LocalData\a022927\OneDrive - Alliance\projet_Lu\1st_work\src_matlab\data\result_ga_03_mse_10optimi.csv', header=None)
#    result_ga.columns = ['Vehicle_ID_V2', 'v0', 'tau', 's0', 'error_RMSE_Acel', 'nb_iterations']
#    b = result_ga[['v0', 'tau', 's0', 'error_RMSE_Acel', 'nb_iterations']]
#    b.columns = ['v0_max_speed', 'tau_reaction_time', 's0_min_distance', 'error_RMSE_Acel', 'nb_iterations']
#    b[['v0_max_speed', 'tau_reaction_time', 's0_min_distance', 'error_RMSE_Acel', 'nb_iterations']].hist(bins = 50) # 1866 
#    
#    result_valide_calibrarion = result_ga.loc[result_ga.error_RMSE_Acel < 2.5] # 1687
#    result_valide_calibrarion[['v0', 'tau', 's0', 'error_RMSE_Acel']].hist(bins = 50) # 1687
    
    
#    parameters = result_ga.loc[result_ga.Vehicle_ID_V2 == 1323]
#    data_vehicle = dataset.loc[dataset['Vehicle_ID_V2'] == 1323].sort_values('Frame_ID')
#    
#    X = np.array(data_vehicle[['v_Vel_V2', 'delta_V', 'delta_Y']])
#    Y = np.array(data_vehicle[['v_Acc_V2']]).flatten()
#    
#    a_simul = IDM_3_parameters(float(parameters.v0), float(parameters.tau), float(parameters.s0), X)
#    fig = plt.figure()
#    plt.scatter( np.arange(len(X)), a_simul,  s=5)
#    plt.scatter( np.arange(len(X)), Y,  s=5)
#    plt.plot( a_simul , label= 'calibrated a by simulation')
#    plt.plot( np.array(Y), label= 'real accel')
#    
#    plt.ylabel('acceleration')
#    plt.legend()
#
#    a_simul_cumsum = np.insert(np.cumsum(a_simul)/10, 0, 0.0)
#    a_simul_cumsum = np.insert(np.cumsum(a_simul)/10, 0, 0.0)
#    
#    fig = plt.figure()
#    plt.scatter( np.arange(len(X)), X[0][0] + a_simul_cumsum[:-1],  s=5)
#    plt.scatter( np.arange(len(X)), X[:, 0],  s=5)
#    plt.plot( X[0][0] + a_simul_cumsum[:-1] , label= 'calibrated a by simulation')
#    plt.plot( X[:, 0] , label= 'real speed')
#    plt.ylabel('speed')
#    plt.legend()
#    
    
    
#    g = dataset[[ "THW", 'Vehicle_ID_V2']].groupby('Vehicle_ID_V2').count() 
#    g.columns = ['nbr_frames']
#    veh2 = vehicles2.merge(g, on='Vehicle_ID_V2')
#    
#    g = dataset[[ "delta_Y", 'Vehicle_ID_V2']].groupby('Vehicle_ID_V2').min() 
#    g.columns = ['min_distance']
#    veh2 = veh2.merge(g, on='Vehicle_ID_V2')
#     
#    veh2 = veh2.merge(result_valide_calibrarion, on = 'Vehicle_ID_V2') # 673
#    veh2 = veh2.merge(g, on='Vehicle_ID_V2') # 1687
#    
#    veh2_selec = veh2.loc[veh2.nbr_frames > 400] # 1451
#    veh2_selec.THW_min.hist(bins=50)
#    veh2_selec.hist(bins = 50)
#    veh2_selec[['v0', 'tau', 's0', 'error_RMSE_Acel']].hist(bins = 50)
#    veh2_long = veh2_selec.loc[veh2.THW_min > 1.8] # 83
    
#    veh2_long = veh2_selec.loc[veh2.THW_min > 2] # 33
    

#    veh2_long[['v0', 'tau', 's0', 'error_RMSE_Acel']].hist(bins = 50)
#    veh2_long[['v_Vel_V2_max', 'THW_min']].hist(bins = 50)
#    
#    veh2_selec = veh2.loc[veh2.nbr_frames > 400]
#    
#    veh2_aggressif = veh2_selec.loc[veh2.THW_mean < 0.85] # 47
##    veh2_aggressif = veh2_selec.loc[veh2.THW_mean < 1] # 108
#    veh2_aggressif[['v0', 'tau', 's0', 'error_RMSE_Acel']].hist(bins = 50)
#    
#    path = r'D:\LocalData\a022927\OneDrive - Alliance\projet_Lu\1st_work\NGSIM_simulation\data\data_processing\result_ga_03_mse_10optimi_agres.csv'
#    veh2_aggressif[['Vehicle_ID_V2', 'v0', 'tau', 's0', 'THW_min', 'error_RMSE_Acel' ]].to_csv(path, index = False)
#    
#    path = r'D:\LocalData\a022927\OneDrive - Alliance\projet_Lu\1st_work\NGSIM_simulation\data\data_processing\result_ga_03_mse_10optimi_long.csv'
#    veh2_long[['Vehicle_ID_V2', 'v0', 'tau', 's0', 'error_RMSE_Acel']].to_csv(path, index = False)
#    
#    path = r'D:\LocalData\a022927\OneDrive - Alliance\projet_Lu\1st_work\NGSIM_simulation\data\data_processing\veh2_selec_nb_frames_400.csv'
#    veh2_selec[['Vehicle_ID_V2', 'v0', 'tau', 's0', 'error_RMSE_Acel']].to_csv(path, index = False)
#    
#    data_vehicle = dataset.loc[dataset['Vehicle_ID_V2'] == 1323].sort_values('Frame_ID')
    
    
#    h = 2
#    w = 2
#    fig=plt.figure(figsize=(8, 8))
#    ax = fig.add_subplot(1, 1, 1)
#    ax.set_title('Display of vehicle ' + str(1323) + ' time sequential data and the leading vehicule speed')
#    ax.axis('off')
#    fig.add_subplot(h, w, 1)
#    plt.scatter( data_vehicle['Frame_ID'], data_vehicle['v_Vel_V1']*3.6,  s=5, c ='#1f77b4', label= 'speed V1')
#    plt.scatter(data_vehicle['Frame_ID'], data_vehicle['v_Vel_V2']*3.6, s=5, color='#ff7f0e', label= 'speed V2')
#    plt.plot(data_vehicle['Frame_ID'], data_vehicle['v_Vel_V1']*3.6, c='#1f77b4', label= 'speed V1')
#    plt.plot(data_vehicle['Frame_ID'], data_vehicle['v_Vel_V2']*3.6,  color='#ff7f0e', label='speed V2')
#    plt.ylabel('speed (km/h)')
#    plt.legend()
#    
#    fig.add_subplot(h, w, 2)
#    plt.scatter(data_vehicle['Frame_ID'], data_vehicle['delta_V'],  s=5, label= 'relative_speed')
#    plt.plot(data_vehicle['Frame_ID'], data_vehicle['delta_V'], label= 'relative_speed')
#    plt.ylabel('speed (m/s)')
#    plt.legend()
    
##    fig.add_subplot(h, w, 3)
##    plt.scatter(data_vehicle['Frame_ID'], data_vehicle['Local_X_V2'],  s=5, label= 'position V2')
##    plt.scatter(data_vehicle['Frame_ID'], data_vehicle['Local_X_V1'],  s=5,  label= 'position V1')
#    
##    plt.plot(data_vehicle['Frame_ID'], data_vehicle['Local_X_V2'])
##    plt.plot(data_vehicle['Frame_ID'], data_vehicle['Local_X_V1'])
#    plt.ylabel('leader V1 and follower V2 position X (m)')

#    fig.add_subplot(h, w, 2)
#    plt.scatter(data_vehicle['Frame_ID'], data_vehicle['v_Acc_V2'],  s=5, label= 'Accel V2', color='#ff7f0e')
#
#    plt.plot(data_vehicle['Frame_ID'], data_vehicle['v_Acc_V2'], label= 'Accel V2',  color='#ff7f0e')
#    plt.ylabel('follower acceleration (m/s2)')
#    
#    fig.add_subplot(h, w, 3)
#    plt.scatter(data_vehicle['Frame_ID'], data_vehicle['delta_Y'],  s=5, label= 'distance', color='#ff7f0e')
#    plt.plot( data_vehicle['Frame_ID'], data_vehicle['delta_Y'], label= 'distance',  color='#ff7f0e')
#    plt.ylabel('distance (m)')
#    plt.legend()
#    
#    fig.add_subplot(h, w, 5)
#    plt.scatter(data_vehicle['Frame_ID'], data_vehicle['delta_X'],  s=5)
#    plt.plot( data_vehicle['Frame_ID'], data_vehicle['delta_X'], label= 'delta_X')
#    plt.ylabel('distance X (m)')
#    plt.legend()
#    
#    fig.add_subplot(h, w, 4)
#    plt.scatter(data_vehicle['Frame_ID'], data_vehicle['THW'],  s=5, label= 'THW',  color='#ff7f0e')
#    plt.plot( data_vehicle['Frame_ID'], data_vehicle['THW'], label= 'THW',  color='#ff7f0e')
#    plt.ylabel('THW (s)')
#    plt.legend()

 
#    fig.add_subplot(h, w, 8)
#    plt.scatter(data_vehicle['Frame_ID'], data_vehicle['TTC'],  s=5)
#    plt.plot( data_vehicle['Frame_ID'], data_vehicle['TTC'], label= 'TTC')
#    plt.ylabel('TTC (m)')
#    plt.legend()    
#    fig.add_subplot(h, w, 6)
##    plt.scatter(data_vehicle['Frame_ID'], data_vehicle['Lane_ID_V1'],  s=5)
#    plt.scatter(data_vehicle['Frame_ID'], data_vehicle['Lane_ID_V2'],  s=5)
#    plt.ylabel('lane for V2')
#    plt.legend()
    
#    
#    plt.show()
#    fig=plt.figure()
#    (data_vehicle[['v_Vel_V2']]*3.6).hist(bins=20)
#    plt.xlabel('speed (km/h)')
#    plt.legend()
#    
#    fig=plt.figure()
#    plt.scatter(vehicles2.THW_mean, vehicles2.TTC_mean)
#    plt.xlabel('THW_mean (s)')
#    plt.ylabel('TTC_mean (s)')
#    plt.legend()
#    
    
#    df_auto = df0.loc[df0['v_Class'] == 2].copy()
#    df_auto = df_auto.loc[df_auto['Lane_ID'] != 6]
#    df_auto = df_auto.loc[df_auto['Lane_ID'] != 7]
#    df_auto = df_auto.loc[df_auto['Lane_ID'] != 8]
#    
#    df = df_auto.copy()
#    df['Lane_ID'] = df['Lane_ID'].replace([5, 4, 3, 2, 1], [0, 1, 2, 3, 4])
#    g = df[[ "Time_Hdwy", 'Vehicle_ID']].groupby('Vehicle_ID').min()
#    g.columns = ['THW_min']
#    df_merged = df.merge(g, on='Vehicle_ID')
#    
#    g = df[["Time_Hdwy", 'Vehicle_ID']].groupby('Vehicle_ID').max()
#    g.columns = ['THW_max']
#    df_merged = df_merged.merge(g, on='Vehicle_ID')
#    
#    g = df[[ "Time_Hdwy", 'Vehicle_ID']].groupby('Vehicle_ID').mean()
#    g.columns = ['THW_mean']
#    df_merged = df_merged.merge(g, on='Vehicle_ID')
#    
#    df_merged = df_merged[[ "Vehicle_ID", 'THW_min', 'THW_max', 'THW_mean']].groupby('Vehicle_ID').mean()
#    
#    trajectories = df.sort_values('Frame_ID').groupby('Vehicle_ID')['Lane_ID'].unique()
#    trajectories.name = 'traj'
#    df_merged = df_merged.join(trajectories, on='Vehicle_ID')
#    
#    lane_change = trajectories.apply(lambda x: len(x)-1) # 486 lane changes in total 
#    lane_change.name = 'nb_lane_changes'
#    df_merged = df_merged.join(lane_change, on='Vehicle_ID')
#
#    
#    g = df0[[ "Time_Hdwy", 'Vehicle_ID']].groupby('Vehicle_ID').count() 
#    g.columns = ['nbr_frames']
#    df_merged = df_merged.merge(g, on='Vehicle_ID')
#    
    
#    h = 1
#    w = 3
#    fig=plt.figure(figsize=(8, 8))
#    ax = fig.add_subplot(1, 1, 1)
#    ax.axis('off')
#    fig.add_subplot(h, w, 1)
#    plt.scatter( data_vehicle['Frame_ID'], data_vehicle['v_Vel_V1']*3.6,  s=5, c ='#1f77b4', label= 'speed V1')
#    plt.scatter(data_vehicle['Frame_ID'], data_vehicle['v_Vel_V2']*3.6, s=5, color='#ff7f0e', label= 'speed V2')
#    plt.plot(data_vehicle['Frame_ID'], data_vehicle['v_Vel_V1']*3.6, c='#1f77b4', label= 'speed V1')
#    plt.plot(data_vehicle['Frame_ID'], data_vehicle['v_Vel_V2']*3.6,  color='#ff7f0e', label='speed V2')
#    plt.ylabel('speed (km/h)')
#    plt.legend()
    
#    fig.add_subplot(h, w, 2)
#    plt.scatter(data_vehicle['Frame_ID'], data_vehicle['delta_V'],  s=5, label= 'relative_speed')
#    plt.plot(data_vehicle['Frame_ID'], data_vehicle['delta_V'], label= 'relative_speed')
#    plt.ylabel('speed (m/s)')
#    plt.legend()
    
##    fig.add_subplot(h, w, 3)
##    plt.scatter(data_vehicle['Frame_ID'], data_vehicle['Local_X_V2'],  s=5, label= 'position V2')
##    plt.scatter(data_vehicle['Frame_ID'], data_vehicle['Local_X_V1'],  s=5,  label= 'position V1')
#    
##    plt.plot(data_vehicle['Frame_ID'], data_vehicle['Local_X_V2'])
##    plt.plot(data_vehicle['Frame_ID'], data_vehicle['Local_X_V1'])
#    plt.ylabel('leader V1 and follower V2 position X (m)')
    

#    
#    
#    veh_agressifs_dataset.to_csv(r'D:\LocalData\a022927\OneDrive - Alliance\projet_Lu\1st_work\src_matlab\veh_agressifs_dataset.csv', index=False)
#    veh_longs_dataset.to_csv(r'D:\LocalData\a022927\OneDrive - Alliance\projet_Lu\1st_work\src_matlab\veh_longs_dataset.csv', index=False)
#    
#    
#    cal_speed = pd.read_csv(r'D:\LocalData\a022927\OneDrive - Alliance\projet_Lu\1st_work\src_matlab\data\cal_speed.csv', header=None)
#    cal_speed.columns = ['Vehicle_ID_V2', 'v0', 'tau', 's0', 'error_RMSE_speed', 'nb_iterations']
#    veh_agressif_merged = veh_agressifs[['Vehicle_ID_V2']].merge(cal_speed, on = 'Vehicle_ID_V2' )
#    veh_longs_merged = veh_longs[['Vehicle_ID_V2']].merge(cal_speed, on = 'Vehicle_ID_V2' )
#    
#
#    cal_speed = pd.read_csv(r'D:\LocalData\a022927\OneDrive - Alliance\projet_Lu\1st_work\src_matlab\data\cal_speed_actionTime.csv', header=None)
#    cal_speed.columns = ['Vehicle_ID_V2', 'v0', 'tau', 's0', 'action_time', 'error_RMSE_speed', 'nb_iterations']
#    veh_agressif_merged = veh_agressifs[['Vehicle_ID_V2']].merge(cal_speed, on = 'Vehicle_ID_V2' )
#    veh_longs_merged = veh_longs[['Vehicle_ID_V2']].merge(cal_speed, on = 'Vehicle_ID_V2' )
#    
#    path = r'D:\LocalData\a022927\OneDrive - Alliance\projet_Lu\1st_work\NGSIM_simulation\data\data_processing\result_ga_reaction_time_agressifs.csv'
#    veh_agressif_merged[['Vehicle_ID_V2', 'v0', 'tau', 's0', 'action_time', 'error_RMSE_speed' ]].to_csv(path, index = False)
#    
#    path = r'D:\LocalData\a022927\OneDrive - Alliance\projet_Lu\1st_work\NGSIM_simulation\data\data_processing\result_ga_reaction_time_longs.csv'
#    veh_longs_merged[['Vehicle_ID_V2', 'v0', 'tau', 's0', 'action_time', 'error_RMSE_speed']].to_csv(path, index = False)
#    
#    
#    cal_position = pd.read_csv(r'D:\LocalData\a022927\OneDrive - Alliance\projet_Lu\1st_work\src_matlab\data\cal_position_action_time.csv', header=None)
#    cal_position.columns = ['Vehicle_ID_V2', 'v0', 'tau', 's0', 'action_time', 'error_RMSE_pos', 'nb_iterations']
#    veh_agressif_merged_pos = veh_agressifs[['Vehicle_ID_V2']].merge(cal_position, on = 'Vehicle_ID_V2' )
#    veh_longs_merged_pos = veh_longs[['Vehicle_ID_V2']].merge(cal_position, on = 'Vehicle_ID_V2' )
#    
#    
#    cal_position = pd.read_csv(r'D:\LocalData\a022927\OneDrive - Alliance\projet_Lu\1st_work\src_matlab\data\cal_position.csv', header=None)
#    cal_position.columns = ['Vehicle_ID_V2', 'v0', 'tau', 's0', 'error_RMSE_pos', 'nb_iterations']
#    veh_agressif_merged_pos = veh_agressifs[['Vehicle_ID_V2']].merge(cal_position, on = 'Vehicle_ID_V2' )
#    veh_longs_merged_pos = veh_longs[['Vehicle_ID_V2']].merge(cal_position, on = 'Vehicle_ID_V2' )
    
#    
#    cal_speed = pd.read_csv(r'D:\LocalData\a022927\OneDrive - Alliance\projet_Lu\1st_work\src_matlab\data\cal_speed_actionTime_ab2.csv', header=None)
#    cal_speed.columns = ['Vehicle_ID_V2', 'v0', 'tau', 's0', 'action_time','accel','decel', 'error_RMSE_speed', 'nb_iterations', 'error_RMSE_pos']
#    veh_agressif_merged = veh_agressifs[['Vehicle_ID_V2']].merge(cal_speed, on = 'Vehicle_ID_V2' )
#    veh_longs_merged = veh_longs[['Vehicle_ID_V2']].merge(cal_speed, on = 'Vehicle_ID_V2' )
#    path = r'D:\LocalData\a022927\OneDrive - Alliance\projet_Lu\1st_work\NGSIM_simulation\data\data_processing\result_ga_reaction_time6_agressifs.csv'
#    veh_agressif_merged[['Vehicle_ID_V2', 'v0', 'tau', 's0', 'action_time','accel','decel', 'error_RMSE_speed' ]].to_csv(path, index = False)
#    
#    path = r'D:\LocalData\a022927\OneDrive - Alliance\projet_Lu\1st_work\NGSIM_simulation\data\data_processing\result_ga_reaction_time6_longs.csv'
#    veh_longs_merged[['Vehicle_ID_V2', 'v0', 'tau', 's0', 'action_time','accel','decel', 'error_RMSE_speed',]].to_csv(path, index = False)
#    
#    
#    
#    
    
#    '[list_vehs result nb_iter speed_error position_rmpse speed_rmpse]'
#    cal_position = pd.read_csv(r'E:\1st_work_data_matlab\cal_position_all_15min.csv', header=None)
#    cal_position.columns = ['Vehicle_ID_V2', 'v0', 'tau', 's0', 'action_time','accel','decel', 
#                            'error_RMSE_pos', 'nb_iterations', 'error_RMSE_speed', 'position_rmpse', 'speed_rmpse']
#    
#    
#    veh2_selec_cal = veh2_selec.merge(cal_position, on = 'Vehicle_ID_V2' )
#    veh2_selec_cal = veh2_selec_cal.loc[veh2_selec_cal.position_rmpse<0.3]
#    veh2_selec_cal = veh2_selec_cal.loc[veh2_selec_cal.error_RMSE_pos<20]
#    
#    veh_agressifs = veh2_selec.sort_values('THW_mean').iloc[0:100]  # <0.875s
##    
#    veh_longs = veh2_selec.sort_values('THW_min').iloc[-100:] # > 1.84s 
#    
#    path = r'D:\LocalData\a022927\OneDrive - Alliance\projet_Lu\1st_work\NGSIM_simulation\data\data_processing\result_ga_reaction_time6_agressifs.csv'
#    veh_agressif_merged_pos[['Vehicle_ID_V2', 'v0', 'tau', 's0', 'action_time','accel','decel', 'error_RMSE_pos' ]].to_csv(path, index = False)
#    
#    path = r'D:\LocalData\a022927\OneDrive - Alliance\projet_Lu\1st_work\NGSIM_simulation\data\data_processing\result_ga_reaction_time6_longs.csv'
#    veh_longs_merged_pos[['Vehicle_ID_V2', 'v0', 'tau', 's0', 'action_time','accel','decel', 'error_RMSE_pos',]].to_csv(path, index = False)
    
#    path = r'C:\Users\A022927\OneDrive - Alliance\projet_Lu\1st_work\NGSIM_simulation\data\data_processing\result_ga_reaction_time6_agressifs.csv'
#    
#    veh_ag = pd.read_csv(path)
#    
#    veh_ag.merge(veh_agressifs, on = 'Vehicle_ID_V2')
#    
#    
#    path = r'C:\Users\A022927\OneDrive - Alliance\projet_Lu\1st_work\NGSIM_simulation\data\data_processing\result_ga_reaction_time6_longs.csv'
#    
#    veh_long = pd.read_csv(path)
#    
#    veh_long.merge(veh_longs, on = 'Vehicle_ID_V2')
#    
#    
#    veh_agressif_merged_pos = veh_agressifs[['Vehicle_ID_V2']].merge(cal_position, on = 'Vehicle_ID_V2' )
#    veh_longs_merged_pos = veh_longs[['Vehicle_ID_V2']].merge(cal_position, on = 'Vehicle_ID_V2' )
#    
#    path = r'C:\Users\A022927\OneDrive - Alliance\projet_Lu\1st_work\NGSIM_simulation\simulation_initial_100\data\result_cal_pos_100_agressifs.csv'
#    veh_agressif_merged_pos[['Vehicle_ID_V2', 'v0', 'tau', 's0', 'action_time','accel','decel', 'error_RMSE_pos' ]].to_csv(path, index = False)
#    
#    path = r'C:\Users\A022927\OneDrive - Alliance\projet_Lu\1st_work\NGSIM_simulation\simulation_initial_100\data\result_cal_pos_100_longs.csv'
#    veh_longs_merged_pos[['Vehicle_ID_V2', 'v0', 'tau', 's0', 'action_time','accel','decel', 'error_RMSE_pos',]].to_csv(path, index = False)
#    
#
#    

    
    

    
    
    