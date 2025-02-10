
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
platform = "Raph" # "Orin" "Raph" "Xavier"
send_package_udp = False # [bool]
visualizer_enabled = False # [bool]

######################################### 3: Functions #######################################################
def model_initialization(point_num, transform_model):
    print("Initializing Model: Start")
    origin = np.array([0, 0, 0])
    model = o3d.io.read_triangle_mesh(model_path)
    model_pcd = model.sample_points_poisson_disk(number_of_points=point_num)
    # model_pcd = model_pcd.transform(transform_model)
    model_pcd = model_pcd.transform(transform_model)
    model_pcd.scale(0.001, origin)

    points = np.asarray(model_pcd.points)

    half_width = 0.15
    xmin = -half_width +0.005
    xmax = half_width -0.005
    ymin = -half_width +0.005
    ymax = half_width -0.005
    zmin = -1
    zmax = 2*half_width +0.02

    mask = np.logical_or(
    np.logical_or(
        (points[:, 0] < xmin) | (points[:, 0] > xmax),  # Check x-range
        (points[:, 1] < ymin) | (points[:, 1] > ymax)   # Check y-range
        ),
        (points[:, 2] < zmin) | (points[:, 2] > zmax)  # Check z-range
        )

    filtered_points = points[mask]
    filtered_model_pcd = o3d.geometry.PointCloud()
    filtered_model_pcd.points = o3d.utility.Vector3dVector(filtered_points)

    return filtered_model_pcd

def data_reduction_pcd(pcd, **kwargs):
    pcd_reduced = o3d.geometry.PointCloud()
    centre = kwargs.get('target_transform', None)
    verts = np.array(pcd.points)

    if centre is not None:
        distance_X = centre[0, 3]
        distance_Y = centre[1, 3]
        offset_x = 0.5
        offset_y = 1
        # Remove points that are too far (X-axis)
        indicesXfar = np.where(verts[:,0] > distance_X+offset_x)[0] # find where centre + /2 target thickness
        verts = np.delete(verts, indicesXfar, axis=0)

        # Remove points that are too close (X-axis)
        indicesXclose = np.where(verts[:,0] < distance_X-offset_x)[0] # find where centre - /2 target thickness
        verts = np.delete(verts, indicesXclose, axis=0)
    
        # Remove points that are too far (Y-axis)
        indicesYright = np.where(verts[:,1] < distance_Y+offset_y)[0] # find where centre + /2 target thickness
        verts = np.delete(verts, indicesYright, axis=0)

        # Remove points that are too close (Y-axis)
        indicesYleft = np.where(verts[:,1] > distance_Y-offset_y)[0] # find where centre - /2 target thickness
        verts = np.delete(verts, indicesYleft, axis=0)
    
    else:
        # Remove points that are too far (X-axis)
        indicesXfar = np.where(verts[:,0] > 4)[0] # find where X > 4 m
        verts = np.delete(verts, indicesXfar, axis=0)

        # Remove points that are too close (X-axis)
        indicesXclose = np.where(verts[:,0] <= 0.15)[0] # find where X <= 0.15 m (lower would mean target is inside the chaser)
        verts = np.delete(verts, indicesXclose, axis=0)

    # *** For Y-axis, signs are resversed for some reason. Added top crop (will need to be updated for MECH mechanism. Too remove myself, is not needed if no manipulator)
    # Remove points that are too low (camera Y-axis). Is the target reflection
    indicesZhigh = np.where(verts[:,2] > 0.15)[0] # find where Z > 0.15 m (15cm)
    verts = np.delete(verts, indicesZhigh, axis=0)

    # *** TO REMOVE
    # Remove points that are too high (camera Y-axis).
    indicesZlow = np.where(verts[:,2] < -0.115)[0] # find where Z < -0.15 m (15cm)
    verts = np.delete(verts, indicesZlow, axis=0)  
    pcd_reduced.points = o3d.utility.Vector3dVector(verts)
    return pcd_reduced

def pcd_downsample_fpfh(pcd, voxel_size, radius_normal, radius_feature):
    model_down = pcd.voxel_down_sample(voxel_size) # Downsample and estimate normals for point cloud
    model_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    # Compute FPFH features
    model_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        model_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return model_down, model_fpfh

def draw_result(source_pcd, target_pcd, transformation):
    source_pcd_temp = copy.deepcopy(source_pcd)
    target_pcd_temp = copy.deepcopy(target_pcd)
    source_pcd_temp.transform(transformation)
    source_pcd_temp.paint_uniform_color([0, 0, 1]) # Blue
    target_pcd_temp.paint_uniform_color([1, 0, 0]) # Red
    o3d.visualization.draw_geometries([source_pcd_temp,target_pcd_temp])

def read_specific_frame(seg_num, frame_num):
    try:
        savedrecording = np.load(read_video_path+str(seg_num)+".npy", allow_pickle=True)
        my_frame_data = savedrecording[frame_num] # ["Depth Verts", "Time Stamp"]
        print(f"Video segment {seg_num}, frame {frame_num} successfully loaded")
    except Exception as e:
        print(e)
    return my_frame_data

def registration_RANSAC(model_down, target_down, model_fpfh, target_fpfh, distance_threshold, confidence):
    ransac_start = time.time()
    transform_RANSAC = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        model_down, target_down, model_fpfh, target_fpfh, True,
        distance_threshold, 
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, confidence))
    ransac_end = time.time()
    time_ransac = ransac_end-ransac_start
    if visualizer_enabled:
        draw_result(model_down, target_down, transform_RANSAC.transformation)
    return transform_RANSAC, time_ransac

def estimate_pose(transformation):
    # Rz(gamma) Ry(beta) Rx(alpha) [degrees]
    gamma = np.arctan2(transformation[1,0], transformation[0,0])
    beta = np.arctan2(-transformation[2,0], np.sqrt(transformation[2,1]**2+transformation[2,2]**2))
    alpha = np.arctan2(transformation[2,1], transformation[2,2])
    # In Deg
    beta = round(np.rad2deg(beta),3) # Ry
    alpha = round(np.rad2deg(alpha),3) # Rx
    gamma = round(np.rad2deg(gamma),3) # Rz
    
    # In Rad
    # beta = round(beta,3) # Ry
    # alpha = round(alpha,3) # Rx
    # gamma = round(gamma,3) # Rz

    # Translations [mm]
    x = round(transformation[0, 3],4)
    y = round(transformation[1, 3],4)
    z = round(transformation[2, 3],4)
    # x = round(transformation[0, 3]*1000,2)
    # y = round(transformation[1, 3]*1000,2)
    # z = round(transformation[2, 3]*1000,2)
    return [x, y, z, alpha, beta, gamma]

######################################### 1: Initialization ######################################################
# Camera frame to SPOT frame transform
""" ############################################### """
a_model_num_point = [10000, 100000] # 10000 [1000, 10000, 100000]
seg_num = 1
frame_num = 1
a_voxel_size = [0.1, 0.05, 0.03, 0.01, 0.005]  # 0.03
a_distance_threshold_mult = [1, 2, 4] # 2
a_radius_normal_mult = [1, 2, 4] # 2
a_radius_feature_mult = [3, 5, 7] # 5
a_confidence = [0.99, 0.999, 0.9999] # 0.99
""" ############################################### """

RotXNeg90 = np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]])
RotY90 = np.array([[0,0,1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]])
RotZ90 = np.array([[0,-1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]])
RotZNeg90 = np.array([[0,1,0,0],[-1,0,0,0],[0,0,1,0],[0,0,0,1]])
RotX90 = np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]])

transform_camera = np.dot(RotY90, RotZNeg90)
transform_camera[0,3] = 0.15
transform_camera[1,3] = 0.087
transform_model = np.dot(RotZ90,RotX90)

# Set script states
read_recording, record_enabled = (True, False) if Select_Mode == "Read" else (False, True) if Select_Mode == "Record Live" else (False, False)

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
    model_path = "E:/Ecole/Carleton/Fall2024/MAAE4907Q/Scripts/o3d_icp/Target_v39.stl"
    model_pc_path = "E:/Ecole/Carleton/Fall2024/MAAE4907Q/Scripts/o3d_icp/Target_v39_pc.npy"
    result_path = "E:/Ecole/Carleton/Fall2024/MAAE4907Q/Scripts/o3d_icp/TestFolder/ransac_test_results.csv"
    result_pc_path = "E:/Ecole/Carleton/Fall2024/MAAE4907Q/Scripts/o3d_icp/TestFolder/ransac_test_results_pc_"
elif platform == "Xavier":
    print("Not set for Xavier")
    exit(0)

# Script Variables - General
target_pcd = o3d.geometry.PointCloud()
transform = None

result_array = []
result_array_header = ["test_id", "ransac_iter", "seg_num", "frame_num", "model_num_point", "voxel_size", "radius_normal_mult", "radius_feature_mult",
                       "distance_threshold_mult", "confidence",
                       "time_ransac", "RANSAC_x", "RANSAC_y", "RANSAC_z", "RANSAC_Rx", "RANSAC_Ry", "RANSAC_Rz", "RANSAC_fitness", "RANSAC_rmse"]
result_array = [result_array_header]

######################################### 3: Main Script #######################################################
print(f"Selected Mode: {Select_Mode}, Platform: {platform}, UDP Enabled: {send_package_udp}")

frame_data = read_specific_frame(seg_num, frame_num)
verts = frame_data[0]
target_pcd.points = o3d.utility.Vector3dVector(verts)
target_pcd.transform(transform_camera)
target_pcd = data_reduction_pcd(target_pcd)

test_id = 405

try:
    for model_num_point in a_model_num_point:
        model_pcd = model_initialization(model_num_point, RotZ90)

        for voxel_size in a_voxel_size:
            print(voxel_size)
            for distance_threshold_mult in a_distance_threshold_mult:
                distance_threshold = voxel_size * distance_threshold_mult
                
                for radius_normal_mult in a_radius_normal_mult:
                    radius_normal = voxel_size * radius_normal_mult
                    
                    for radius_feature_mult in a_radius_feature_mult:                   
                        radius_feature = voxel_size * radius_feature_mult
                        model_down, model_fpfh = pcd_downsample_fpfh(model_pcd, voxel_size, radius_normal, radius_feature)
                        target_down, target_fpfh = pcd_downsample_fpfh(target_pcd, voxel_size, radius_normal, radius_feature)

                        for confidence in a_confidence:
                            result_array_pc = []
                            result_array_pc_header = ["model_pc", "target_pc"]
                            result_array_pc = [result_array_pc_header]
                            for ransac_iter in range(25):
                                transform, time_ransac = registration_RANSAC(model_down, target_down, model_fpfh, target_fpfh, distance_threshold, confidence)
                                RANSAC_fitness = transform.fitness
                                RANSAC_rmse = transform.inlier_rmse
                                RANSAC_pose = estimate_pose(transform.transformation) # X, Y, Z, Rx, Ry, Rz (meters and degrees, relative Chaser-Target) 
                                result_array.append([test_id, ransac_iter, seg_num, frame_num, model_num_point, voxel_size, 
                                                    radius_normal_mult, radius_feature_mult, distance_threshold_mult, confidence,
                                                    time_ransac, RANSAC_pose[0], RANSAC_pose[1], RANSAC_pose[2], RANSAC_pose[3],
                                                    RANSAC_pose[4], RANSAC_pose[5], RANSAC_fitness, RANSAC_rmse])
                                print(RANSAC_fitness)  
                                model_pcd_temp = copy.deepcopy(model_down)
                                model_pcd_temp.transform(transform.transformation)
                                vert_target_pcd = np.asarray(target_down.points)
                                vert_model_pcd_temp = np.asarray(model_pcd_temp.points)
                                result_array_pc.append([vert_model_pcd_temp,vert_target_pcd])
                            result_array_pc = np.asarray(result_array_pc, dtype=object)
                            np.save(f"{result_pc_path}{test_id}.npy", result_array_pc, allow_pickle=True)
                            test_id += 1

finally:
    np.savetxt(result_path, result_array, fmt="%s", delimiter=',')