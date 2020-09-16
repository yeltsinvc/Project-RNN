# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:59:45 2020

@author: GRETTIA
"""


import numpy as np

#objet that represent the lane
class Geometry:
    def __init__(self,name):
        self.name=name
        self.isLane=[]


#funtion to create csv with the geometry(border of lanes) -----> yvc
def getGeometry(indDirectory,img_world):
    from geopandas import GeoSeries
    from shapely.geometry import Polygon
    import pandas as pd
    
    dataframe=pd.DataFrame()
    
    def on_mouse(event, x, y, buttons,param):
        FINAL_LINE_COLOR = (255, 255, 255)
        
        global user_param
        
           
        if event == cv2.EVENT_LBUTTONDOWN:
            # Left click means adding a point at current position to the list of points
            #print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
            if user_param is None:
                
                user_param=[]
            else:
                 user_param.append((x, y))
            
            print(user_param)
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Left click means adding a point at current position to the list of points
            #print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
            user_param.append((x, y))
            cv2.polylines(img, np.array([user_param]), True, FINAL_LINE_COLOR, 1)
            n_lane=input("Enter ID of lane: ")
            
            dataframe['Lane '+n_lane]=[Polygon(np.array(user_param)*0.0126999352667008*12)]#####modifi
            dataframe.to_csv(indDirectory + "\geometry.csv")
            user_param=[]
            print(user_param)
    
    
    
    import cv2
    global user_param
    
    #img = cv2.imread("world.jpg")
    img = cv2.imread(indDirectory+"/"+img_world)
    cv2.namedWindow("figure")
    param=[]
    cv2.setMouseCallback("figure", on_mouse)
    user_param=[] 
    while True:
        # both windows are displaying the same img
        
        cv2.imshow("figure", img)
        param=[]
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


### copy some funtions of run_track_visualization
    
def read_all_recordings_from_csv(base_path="../data/"):
    """
    This methods reads the tracks and meta information for all recordings given the path of the inD dataset.
    :param base_path: Directory containing all csv files of the inD dataset
    :return: a tuple of tracks, static track info and recording meta info
    """
    tracks_files = base_path + "*_tracks.csv")
    static_tracks_files = base_path + "*_tracksMeta.csv")
    recording_meta_files = base_path + "*_recordingMeta.csv")

    all_tracks = []
    all_static_info = []
    all_meta_info = []
    for track_file, static_tracks_file, recording_meta_file in zip(tracks_files,
                                                                   static_tracks_files,
                                                                   recording_meta_files):
        logger.info("Loading csv files {}, {} and {}", track_file, static_tracks_file, recording_meta_file)
        tracks, static_info, meta_info = read_from_csv(track_file, static_tracks_file, recording_meta_file)
        all_tracks.extend(tracks)
        all_static_info.extend(static_info)
        all_meta_info.extend(meta_info)

    return all_tracks, all_static_info, all_meta_info