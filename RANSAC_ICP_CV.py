# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:49:36 2024

@author: Raphael
"""

# Import Statements
import pyrealsense2 as rs
import numpy as np
import time
import math
import threading
import sys
import open3d as o3d
import cv2
import pickle
import os

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 3 # 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ], criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),)
    return result

def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 2 # 0.4
#    result = o3d.pipelines.registration.registration_icp(
#       source, target, distance_threshold, transform.transformation,
#        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    result = o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd, distance_threshold, transform.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
    return result

def read_recording_data():
    compoundframe = savedrecording[readrecording_frameNUM] # ["Depth Frame", "Color Frame", "Texture Coordinate", "Time Stamp"]
    frame_timestamp = compoundframe[3]
    delta_T = compoundframe[3] - last_recordedtimestamp # (ms)
    
    last_recordedtimestamp = compoundframe[3] # Update last timestamp for next iteration
    
    verts = compoundframe[0]
    texcoords = compoundframe[1]
    color_source = compoundframe[2]
    readrecording_frameNUM += 1
    
    if readrecording_frameNUM > len(savedrecording)-1:
        try:
            recording_segNUM += 1
            savedrecording = np.load(read_video_path+str(recording_segNUM)+".npy", allow_pickle=True)
            print(recording_segNUM)
            readrecording_frameNUM = 1
        except Exception: # next segment doesnt exist -> end of recording reached
            run_loop = False

def read_live_data():
    # Wait for a coherent pair of frames: depth and color       
    frames = pipeline.wait_for_frames()

    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    frame_timestamp = frames.get_timestamp()
    depth_frame = decimate.process(depth_frame) # The Decimate reduces resolution by 2 folds

    # Grab new intrinsics (may be changed by decimation)
    depth_intrinsics = rs.video_stream_profile(
        depth_frame.profile).get_intrinsics()
    w, h = depth_intrinsics.width, depth_intrinsics.height # The Decimate reduces resolution by 2 folds

    depth_image = np.asanyarray(depth_frame.get_data())
    ### Maybe check if depth_image is not empty, if empty next loop.

    color_image = np.asanyarray(color_frame.get_data())

    depth_colormap = np.asanyarray(
        colorizer.colorize(depth_frame).get_data())

    points = pc.calculate(depth_frame)

    # Pointcloud data to arrays
    v, t = points.get_vertices(), points.get_texture_coordinates()
    verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
    texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv

def record_data():
    recording.append([verts, texcoords, color_source, frame_timestamp])
    
    if len(recording) >=10:
        recording = np.asarray(recording, dtype=object)
        np.save(record_video_path+str(recording_segNUM)+".npy", recording, allow_pickle=True)
        recording_segNUM += 1
        recording = [["Depth Frame", "Color Frame", "Texture Coordinate", "Time Stamp"]]

"""
USER INPUT START
"""
Select_Mode = "Live" # "Read" "Live" "Record Live"
control_fps = False  # Relevant if "Read" 

# Orin Paths
## Video: "/home/spot-vision/Documents/o3d_icp/recording/myrecording"
## Video Intrinsics: "/home/spot-vision/Documents/o3d_icp/recording/myrecordingintrinsics.npy"

read_video_path = "/home/spot-vision/Documents/o3d_icp/recording/myrecording" # Path String with file name, without index nor extension
read_video_intrinsics_path = "/home/spot-vision/Documents/o3d_icp/recording/myrecordingintrinsics.npy" # Path String with file name

record_video_path = "/home/spot-vision/Documents/o3d_icp/recording/myrecording" # Path String with file name, without index nor extension
record_video_intrinsics_path = "/home/spot-vision/Documents/o3d_icp/recording/myrecordingintrinsics.npy" # Path String with file name

pose_estimation_path = "/home/spot-vision/Documents/o3d_icp/pose_estimation/pose_estimation.csv" # Path String with file name with extension

source = o3d.io.read_triangle_mesh("/home/spot-vision/Documents/o3d_icp/Target_v10.stl")
"""
USER INPUT END
"""

# RANSAC and ICP parameters
voxel_size = 0.05  # (m) Change depending on point cloud size
radius_normal = voxel_size * 2
radius_feature = voxel_size * 10
rmse_threshold = 0.05  # User-defined RMSE threshold

# Model Initialization
## Convert Source Mesh to point cloud
origin = np.array([0, 0, 0])
source_pcd = source.sample_points_poisson_disk(number_of_points=5000)
source_pcd.scale(0.001, origin)
source_down = source_pcd.voxel_down_sample(voxel_size) # Downsample and estimate normals for point cloud
source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
# Compute FPFH features
source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

# Set script states
read_recording, record_enabled = (True, False) if Select_Mode == "Read" else (False, True) if Select_Mode == "Record Live" else (False, False)

# Camera frame to SPOT frame transform
RotZ90 = np.array([[0,-1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]])
RotX90 = np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]])
transform_camera = np.dot(RotZ90,RotX90)

# Configure depth and color streams
if not read_recording:    
    pipeline = rs.pipeline()
    config = rs.config()
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break

    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Get stream profile and camera intrinsics
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    depth_intrinsics = depth_profile.get_intrinsics()
    w, h = depth_intrinsics.width, depth_intrinsics.height
    if record_enabled:
        recording_intrinsics = [w, h]
        np.save(record_video_intrinsics_path, recording_intrinsics, allow_pickle=True)
else:
    savedrecording = np.load(read_video_path+str(1)+".npy", allow_pickle=True)
    last_recordedtimestamp = savedrecording[1][3]
    savedrecordingintrinsics = np.load(read_video_intrinsics_path, allow_pickle=True)
    w, h = savedrecordingintrinsics[0], savedrecordingintrinsics[1]

# Processing blocks
pc = rs.pointcloud()
decimate = rs.decimation_filter()
decimate.set_option(rs.option.filter_magnitude, 2 ** 1)
colorizer = rs.colorizer()

# Script
target_pcd = o3d.geometry.PointCloud()
out = np.empty((h, w, 3), dtype=np.uint8)

run_loop = True
i=0
recording = [["Depth Frame", "Color Frame", "Texture Coordinate", "Time Stamp"]]
readrecording_frameNUM = 2 # If read recording frame iterator to replace camera stream. Skip 0 and 1 (column title + 1st frame)
recording_segNUM = 1 # Video segment number, used for both read and write recording
transform = None
transform_array = []
try:
    while run_loop:       
        # Live Camera
        if not read_recording:
            print("Loop")
            read_live_data()
            if record_enabled:
                record_data()

        # Reading a Recorded Video
        elif read_recording:
            read_recording_data()

        # Pointcloud Vertices (verts) processing 
        ## remove background (index z not within  0-2m and delete that point)
        #indicesZ = np.where(not(0 < verts[:,2] > 2))[0] # find where 0 m > z > 2 m
        indicesZ = np.where(verts[:,2] > 2)[0] # find where z > 1.7 m
        verts = np.delete(verts, indicesZ, axis=0)
        
        target_pcd.points = o3d.utility.Vector3dVector(verts)
        target_pcd = target_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)[0]
        target_down = target_pcd.voxel_down_sample(voxel_size) # Downsample and estimate normals for point cloud
        
        try:
            target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))  
            target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        except:
            next
            
        # RANSAC
        if transform is None:
            transform = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
            print("RANSAC")

        # ICP REFINEMENT
        transform = refine_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
        transform_fitness = transform.fitness
        transform_rmse = transform.inlier_rmse

        best_transformation = None
        best_fitness = -1
        best_rmse = float('inf')

        # Update best transformation based on fitness and RMSE threshold
        if transform_rmse < rmse_threshold and transform_fitness > best_fitness:
            # best_fitness = transform_fitness
            best_rmse = transform_rmse
            best_transformation = transform.transformation


        # Check if a valid transformation was found
        if best_transformation is not None:
            #print(transform.transformation)
            transform_relative = np.dot(transform_camera,transform.transformation) # Transform from camera frame to inertial standard
            #print(transform_relative)
            transform_package = [transform_relative[0,3],transform_relative[1,3], transform_relative[1,1]] # X, Y, rotation of X-axis about Z (rad) [relative chaser-target]
            print(transform_package)
            transform_array.append([frame_timestamp, transform_package[0], transform_package[1],transform_package[2], transform_fitness, transform_rmse]) # SEND TRANSFORM ARRAY TO GNC
            print("ICP SUCCESS")
        else:
            transform = None # TO REVIEW LOGIC, DO WE WANT TO JUMP STRAIGHT BACK TO RANSAC OR TRY ICP AGAIN. Do we send data to GNC anyway? With fitness score?
            print("ICP FAIL")

        if read_recording and control_fps:
            time.sleep(delta_T/1000)

# Stop streaming
finally:
    np.savetxt(pose_estimation_path, transform_array, delimiter=',')
    if not read_recording:
        pipeline.stop()
        profile = pipeline = None

    if record_enabled:
        recording = np.asarray(recording, dtype=object)
        np.save(record_video_path+str(recording_segNUM)+".npy", recording, allow_pickle=True)