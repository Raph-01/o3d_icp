# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:49:36 2024

@author: Raphael
https://www.open3d.org/docs/latest/tutorial/pipelines/icp_registration.html
"""

# Import Statements
import pyrealsense2 as rs
import numpy as np
import time
import math
import threading
import sys
import open3d as o3d # 0.16.0
import cv2
import pickle
import os
import socket
import struct
import copy

######################################### 1: User-defined Variables #######################################################
Select_Mode = "Read" # "Read" "Live" "Record Live"
platform = "Orin" # "Orin" "Raph" "Xavier"
send_package_udp = False # [bool]
visualizer_enabled = True # [bool]

######################################### 3: Functions #######################################################
def draw_result(source_pcd, target_pcd, transformation):
    source_pcd_temp = copy.deepcopy(source_pcd)
    target_pcd_temp = copy.deepcopy(target_pcd)
    inverse_transformation = np.linalg.inv(transformation)
    transformed_target_pcd = target_pcd_temp.transform(inverse_transformation)
    source_pcd_temp.paint_uniform_color([0, 0, 1]) # Blue
    transformed_target_pcd.paint_uniform_color([1, 0, 0]) # Red
    o3d.visualization.draw_geometries([source_pcd_temp, transformed_target_pcd])

def read_recorded_data():
    global savedrecording_frameNUM
    global savedrecording_segNUM
    global savedrecording
    end_recording = False

    if len(savedrecording) == 0 or savedrecording_frameNUM > len(savedrecording)-1:
        try:
            savedrecording = np.load(read_video_path+str(savedrecording_segNUM)+".npy", allow_pickle=True)
            savedrecording_frameNUM = 1
            my_frame_data = savedrecording[savedrecording_frameNUM] # ["Depth Verts", "Time Stamp"]
            print(f"Video segment {savedrecording_segNUM}, frame {savedrecording_frameNUM} successfully loaded")
            savedrecording_segNUM += 1
            savedrecording_frameNUM += 1
        except:
            end_recording = True
            print(f"End of recording reached at video segment {savedrecording_segNUM-1}, frame {savedrecording_frameNUM-1}")
            my_frame_data = []

    else:
        my_frame_data = savedrecording[savedrecording_frameNUM] # ["Depth Verts", "Time Stamp"]
        print(f"Video segment {savedrecording_segNUM-1}, frame {savedrecording_frameNUM} successfully loaded")
        savedrecording_frameNUM += 1

    return my_frame_data, not(end_recording)

def read_live_data():
    # Wait for a coherent pair of frames: depth and color       
    frames = pipeline.wait_for_frames()
    
    frame_timestamp = frames.get_timestamp()
    depth_frame = frames.get_depth_frame()
    depth_frame = decimate.process(depth_frame) # The Decimate reduces resolution by 2 folds

    points = pc.calculate(depth_frame)

    # Pointcloud data to array
    v = points.get_vertices()
    verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
    return [verts, frame_timestamp]

def record_data(data):
    recording.append(data)
    
    if len(recording) >=10:
        recording = np.asarray(recording, dtype=object)
        np.save(record_video_path+str(savedrecording_segNUM)+".npy", recording, allow_pickle=True)
        savedrecording_segNUM += 1
        recording = [recording_header]

def registration_ICP_point2point(model_pcd, target_pcd, distance_threshold_ICP, transform):
    transform_ICP = o3d.pipelines.registration.registration_icp(
            model_pcd, target_pcd, distance_threshold_ICP, transform.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=5000,
                                                              relative_fitness=0.0000001))
    return transform_ICP

def registration_RANSAC(model_down, target_down, model_fpfh, target_fpfh):
    transform_RANSAC = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        model_down, target_down, model_fpfh, target_fpfh, True,
        distance_threshold_RANSAC,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold_RANSAC)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return transform_RANSAC

def estimate_pose(transformation):
    # Rz(gamma) Ry(beta) Rx(alpha) [degrees]
    gamma = np.arctan2(transformation[1,0], transformation[0,0])
    beta = np.arctan2(-transformation[2,0], np.sqrt(transformation[2,1]**2+transformation[2,2]**2))
    alpha = np.arctan2(transformation[2,1], transformation[2,2])
    beta = round(np.rad2deg(beta),3) # Ry
    alpha = round(np.rad2deg(alpha),3) # Rx
    gamma = round(np.rad2deg(gamma),3) # Rz
    
    # Translations [mm]
    x = round(transformation[0, 3]*1000,2)
    y = round(transformation[1, 3]*1000,2)
    z = round(transformation[2, 3]*1000,2)
    return [x, y, z, alpha, beta, gamma]

def data_reduction_verts(verts, **kwargs):
    centre = kwargs.get('target_camera_X_Z', None)
        
    if centre:
        delta_centreZ = 0.5
        delta_centreX = 0.5
        # Remove points that are too far (camera Z-axis)
        indicesZfar = np.where(verts[:,2] > centre[1]+delta_centreZ)[0] # find where centre + /2 target thickness
        verts = np.delete(verts, indicesZfar, axis=0)

        # Remove points that are too close (camera Z-axis)
        indicesZclose = np.where(verts[:,2] < centre[1]-delta_centreZ)[0] # find where centre - /2 target thickness
        verts = np.delete(verts, indicesZclose, axis=0)
    
        # Remove points that are too far (camera X-axis)
        indicesXright = np.where(verts[:,0] > centre[0]+delta_centreX)[0] # find where centre + /2 target thickness
        verts = np.delete(verts, indicesXright, axis=0)

        # Remove points that are too close (camera X-axis)
        indicesXleft = np.where(verts[:,0] < centre[0]-delta_centreX)[0] # find where centre - /2 target thickness
        verts = np.delete(verts, indicesXleft, axis=0)
    
    else:
        # Remove points that are too far (camera Z-axis)
        indicesZfar = np.where(verts[:,2] > 4)[0] # find where z > 4 m
        verts = np.delete(verts, indicesZfar, axis=0)

        # Remove points that are too close (camera Z-axis)
        indicesZclose = np.where(verts[:,2] <= 0)[0] # find where z <= 0 m
        verts = np.delete(verts, indicesZclose, axis=0)

    # *** For Y-axis, signs are resversed for some reason. Added top crop (will need to be updated for MECH mechanism. Too remove myself, is not needed if no manipulator)
    # Remove points that are too low (camera Y-axis). Is the target reflection
    indicesYlow = np.where(verts[:,1] > 0.15)[0] # find where y < 0.15 m (15cm)
    verts = np.delete(verts, indicesYlow, axis=0)

    # *** TO REMOVE
    # Remove points that are too high (camera Y-axis).
    indicesYhigh = np.where(verts[:,1] < -0.25)[0] # find where y > -0.15 m (15cm)
    verts = np.delete(verts, indicesYhigh, axis=0)  
    
    return verts

def noise_reduction_pc(pc):
    reduced_pc = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)[0]
    return reduced_pc

######################################### 1: Initialization ######################################################
# RANSAC and ICP parameters
voxel_size = 0.03  # (m) Change depending on point cloud size
radius_normal = voxel_size * 2
radius_feature = voxel_size * 5
distance_threshold_RANSAC = voxel_size * 2 # 1.5
distance_threshold_ICP = voxel_size * 0.4 # 0.4

# Set script states
read_recording, record_enabled = (True, False) if Select_Mode == "Read" else (False, True) if Select_Mode == "Record Live" else (False, False)

# UDP Setup
if send_package_udp:
    sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    server_address = ('192.168.1.110',30172)

# Paths
if platform == "Orin":
    ## Video: "/home/spot-vision/Documents/o3d_icp/recording/myrecording"
    ## Video Intrinsics: "/home/spot-vision/Documents/o3d_icp/recording/myrecordingintrinsics.npy"
    read_video_path = "/home/spot-vision/Documents/o3d_icp/recording/myrecording" # Path String with file name, without index nor extension
    record_video_path = "/home/spot-vision/Documents/o3d_icp/recording/myrecording" # Path String with file name, without index nor extension
    pose_estimation_path = "/home/spot-vision/Documents/o3d_icp/pose_estimation/pose_estimation.csv" # Path String with file name with extension
    model_path = "/home/spot-vision/Documents/o3d_icp/Target_v39.stl"
    model_pc_path = "/home/spot-vision/Documents/o3d_icp/Target_v39_pc.npy"
elif platform == "Raph":
    read_video_path = "E:/Ecole/Carleton/Fall2024/MAAE4907Q/Scripts/o3d_icp/recording/myrecording" # Path String with file name, without index nor extension
    record_video_path = "E:/Ecole/Carleton/Fall2024/MAAE4907Q/Scripts/o3d_icp/recording/myrecording" # Path String with file name, without index nor extension
    pose_estimation_path = "E:/Ecole/Carleton/Fall2024/MAAE4907Q/Scripts/o3d_icp/pose_estimation/pose_estimation.csv" # Path String with file name with extension
    model_path = "E:/Ecole/Carleton/Fall2024/MAAE4907Q/Scripts/o3d_icp/Target_v10.stl"
    model_pc_path = "E:/Ecole/Carleton/Fall2024/MAAE4907Q/Scripts/o3d_icp/Target_v10_pc.npy"
elif platform == "Xavier":
    print("Not set for Xavier")
    exit(0)

# Model Initialization
try:
    print("Model Initialization: Start")
    origin = np.array([0, 0, 0])
    model = o3d.io.read_triangle_mesh(model_path)
    model_pcd = model.sample_points_poisson_disk(number_of_points=10000)
    model_pcd.scale(0.001, origin)
    model_down = model_pcd.voxel_down_sample(voxel_size) # Downsample and estimate normals for point cloud
    model_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    # Compute FPFH features
    model_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        model_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    print("Model Initialization: Done")
except:
    print("Could not find Target Model.")
    exit(0)

# Camera frame to SPOT frame transform
RotZ90 = np.array([[0,-1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]])
RotX90 = np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]])
transform_camera = np.dot(RotZ90,RotX90)

# Configure depth stream
if not read_recording:    
    pipeline = rs.pipeline()
    config = rs.config()
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    try:    
        pipeline_profile = config.resolve(pipeline_wrapper)
    except:
        print("Connect L515 Camera")
        exit(0)
    
    device = pipeline_profile.get_device()

    # Realsense Objects Initialization
    pc = rs.pointcloud()
    decimate = rs.decimation_filter()
    decimate.set_option(rs.option.filter_magnitude, 2 ** 1)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)

# Script Variables - General
target_pcd = o3d.geometry.PointCloud()
recording_header = ["Depth Verts", "Time Stamp"]
recording = [recording_header]
transform = None
transform_array_header = ["frame_timestamp",
                          "ICP_timestamp", "ICP_x", "ICP_y", "ICP_z", "ICP_Rx", "ICP_Ry", "ICP_Rz",
                          "RANSAC_timestamp", "RANSAC_x", "RANSAC_y", "RANSAC_z", "RANSAC_Rx", "RANSAC_Ry", "RANSAC_Rz",
                          "ICP_fitness", "ICP_rmse", "RANSAC_fitness", "RANSAC_rmse"]
transform_array = [transform_array_header]

# Script Variables - Reading recording
savedrecording = []
savedrecording_frameNUM = 1 # If recording frame iterator to replace camera stream. Skip 0 (column title)
savedrecording_segNUM = 1 # Video segment number, used for both read and write recording

######################################### 3: Main Script #######################################################
print(f"Selected Mode: {Select_Mode}, Platform: {platform}, UDP Enabled: {send_package_udp}")

try:
    run_loop = True
    while run_loop:       
        ICP_pose = [0, 0, 0, 0, 0, 0]
        ICP_transform_fitness = 0
        ICP_transform_rmse = 0       
        RANSAC_timestamp = 0
        RANSAC_pose = [0, 0, 0, 0, 0, 0]
        RANSAC_transform_fitness = 0
        RANSAC_transform_rmse = 0

        # Live Camera
        if not read_recording:
            frame_timestamp_index = 1 # To handle alternative Legacy Recording
            frame_data = read_live_data() # [verts, frame_timestamp]
            if record_enabled:
                record_data(frame_data)
        
        # Reading a Recorded Video
        elif read_recording:
            frame_timestamp_index = 3 # Legacy Recording
            frame_data, run_loop = read_recorded_data() # read_recorded_data sets run_loop used to flag the end of the recording
            if not(run_loop):
                break

        verts = frame_data[0]
        frame_timestamp = frame_data[frame_timestamp_index] # If recorded from old script index = 3, else index = 1

        # Data and noise reduction.
        if transform is None: # Coarse Reduction
            verts_reduced = data_reduction_verts(verts)        
        else: # Refined Reduction
            distance_Z = transform.transformation[2, 3]
            distance_X = transform.transformation[0, 3]
            verts_reduced = data_reduction_verts(verts, target_camera_X_Z=[distance_X, distance_Z])
        target_pcd.points = o3d.utility.Vector3dVector(verts_reduced)
        # target_pcd = noise_reduction_pc(target_pcd)
        
        # RANSAC
        if transform is None:
            target_down = target_pcd.voxel_down_sample(voxel_size) # Downsample and estimate normals for point cloud
            try: 
                target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))  
                target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                    target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
                transform = registration_RANSAC(model_down, target_down, model_fpfh, target_fpfh)
                RANSAC_transform_fitness = transform.fitness
                RANSAC_transform_rmse = transform.inlier_rmse
                print(f"{time.strftime('%H:%M:%S')} RANSAC Success")
            
                # Logic to validate ICP output to bee added in the ICP criteria. Check what is the error output of registration_icp
                transform_relative = np.dot(transform_camera,transform.transformation) # Transform from camera frame to inertial standard
                RANSAC_pose = estimate_pose(transform_relative) # X, Y, Z, Rx, Ry, Rz (millimeters and degrees, relative Chaser-Target)
                RANSAC_timestamp = time.strftime('%H:%M:%S')
                print(RANSAC_pose)
                if visualizer_enabled:
                    draw_result(model_pcd, target_pcd, transform.transformation)
                """"""
                """"""
                distance_Z = transform.transformation[2, 3]
                distance_X = transform.transformation[0, 3]
                verts_reduced = data_reduction_verts(verts, target_camera_X_Z=[distance_X, distance_Z])
                target_pcd.points = o3d.utility.Vector3dVector(verts_reduced)
                """"""
                """"""               
                target_down = target_pcd.voxel_down_sample(voxel_size/3) # Downsample and estimate normals for point cloud
                target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2/3, max_nn=30))  
                target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                    target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5/3, max_nn=100))
                transform = registration_RANSAC(model_down, target_down, model_fpfh, target_fpfh)
                RANSAC_transform_fitness = transform.fitness
                RANSAC_transform_rmse = transform.inlier_rmse
                print(f"{time.strftime('%H:%M:%S')} RANSAC Success")
            
                # Logic to validate ICP output to bee added in the ICP criteria. Check what is the error output of registration_icp
                transform_relative = np.dot(transform_camera,transform.transformation) # Transform from camera frame to inertial standard
                RANSAC_pose = estimate_pose(transform_relative) # X, Y, Z, Rx, Ry, Rz (millimeters and degrees, relative Chaser-Target)
                RANSAC_timestamp = time.strftime('%H:%M:%S')
                print(RANSAC_pose)
                if visualizer_enabled:
                    draw_result(model_pcd, target_pcd, transform.transformation)
            except KeyboardInterrupt:
                print("\nCtrl+C pressed, exiting loop...")
                break
            
            except:
                transform = None
                print(f"{time.strftime('%H:%M:%S')} RANSAC Fail")
                continue

        # ICP REFINEMENT
        try:
            transform = registration_ICP_point2point(model_pcd, target_pcd, distance_threshold_ICP, transform)
            ICP_transform_fitness = transform.fitness
            ICP_transform_rmse = transform.inlier_rmse
            print(f"{time.strftime('%H:%M:%S')} ICP Success")
            
            # Logic to validate ICP output to bee added in the ICP criteria. Check what is the error output of registration_icp
            transform_relative = np.dot(transform_camera,transform.transformation) # Transform from camera frame to inertial standard
            ICP_pose = estimate_pose(transform_relative) # X, Y, Z, Rx, Ry, Rz (meters and degrees, relative Chaser-Target)
            ICP_timestamp = time.strftime('%H:%M:%S')
            print(ICP_pose)
        except:
            transform = None
            print(f"{time.strftime('%H:%M:%S')} ICP Fail")
            continue
        
        if send_package_udp:
            x, y, theta =  ICP_pose[0], ICP_pose[1], ICP_pose[5]
            udp_data = bytearray(struct.pack("fff",x,y,theta))
            try:
                sock.sendto(udp_data,server_address)
                print(f"{time.strftime('%H:%M:%S')} UPD package sent: x={x}, y={y}, theta={theta}.")
            except:
                print(f"{time.strftime('%H:%M:%S')} Failed to send UDP package")

        transform_array.append([frame_timestamp,
                                ICP_timestamp, ICP_pose[0], ICP_pose[1], ICP_pose[2], ICP_pose[3], ICP_pose[4], ICP_pose[5],
                                RANSAC_timestamp, RANSAC_pose[0], RANSAC_pose[1], RANSAC_pose[2], RANSAC_pose[3], RANSAC_pose[4], RANSAC_pose[5],
                                ICP_transform_fitness, ICP_transform_rmse, RANSAC_transform_fitness, RANSAC_transform_rmse])

        if visualizer_enabled:
            draw_result(model_pcd, target_pcd, transform.transformation)

finally:
    np.savetxt(pose_estimation_path, transform_array, fmt="%s", delimiter=',')
    if not read_recording:
        # profile.stop() # AttributeError: pipeline_profile object has no attribute "stop"
        pipeline.stop()
        profile = pipeline = None

    if record_enabled:
        recording = np.asarray(recording, dtype=object)
        np.save(record_video_path+str(savedrecording_segNUM)+".npy", recording, allow_pickle=True)

    if send_package_udp:
        sock.close()