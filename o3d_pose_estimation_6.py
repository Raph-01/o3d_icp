
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
def draw_result(source_pcd, target_pcd, transformation):
    source_pcd_temp = copy.deepcopy(source_pcd)
    target_pcd_temp = copy.deepcopy(target_pcd)
    source_pcd_temp.transform(transformation)
    source_pcd_temp.paint_uniform_color([0, 0, 1]) # Blue
    target_pcd_temp.paint_uniform_color([1, 0, 0]) # Red
    o3d.visualization.draw_geometries([source_pcd_temp,target_pcd_temp, axis])

def receive_udp():
    IP_ADDRESS = '' # Set the IP to allow connections from any address
    PORT = 12345
    NUM_DOUBLES = 9

    # Create a UDP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Bind the socket to the port
    server_address = (IP_ADDRESS, PORT)
    print('Starting up on {} port {}'.format(server_address[0], server_address[1]))
    server_socket.bind(server_address)

    # Set the socket to non-blocking
    server_socket.setblocking(0)
    data = b''
    try:
        # Receive the data
        data, client_address = server_socket.recvfrom(NUM_DOUBLES * 8 - len(data))

        # Unpack and display the data if enough bytes have been received
        if len(data) == NUM_DOUBLES * 8:
            doubles = struct.unpack('d' * NUM_DOUBLES, data)
            print('Received doubles: {}'.format(doubles))
            data = b''  # Clear the data buffer for the next set of doubles

    finally:
        # Close the socket
        server_socket.close()
        return data

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
        except Exception as e:
            print(e)
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
    global savedrecording
    global savedrecording_segNUM

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

def registration_ICP_point2plane(model_pcd, target_pcd, distance_threshold_ICP, transform):
    transform_ICP = o3d.pipelines.registration.registration_icp(
            model_pcd, target_pcd, distance_threshold_ICP, transform.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return transform_ICP

def registration_RANSAC(model_down, target_down, model_fpfh, target_fpfh, distance_threshold, confidence):
    best_ransac = None
    best_fitness = 0
    global visualizer_enabled
    i = 0
    loop_flag = True
    while loop_flag:    
        time_iter_start = time.time()
        transform_RANSAC = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            model_down, target_down, model_fpfh, target_fpfh, True,
            distance_threshold, 
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, confidence))
        print(f"{i} Ransac {transform_RANSAC.fitness}")
        time_iter_end = time.time()
        print(f"Ransac duration = {time_iter_end-time_iter_start}s")
        if transform_RANSAC.fitness > best_fitness:
            best_fitness = transform_RANSAC.fitness
            best_ransac = transform_RANSAC
            draw_result(model_down, target_down, best_ransac.transformation)
        if best_fitness > 0.75 or i > 4:
            loop_flag = False
        i+=1
    if visualizer_enabled:
        draw_result(model_down, target_down, best_ransac.transformation)
    exit(0)
    return best_ransac

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
        indicesYright = np.where(verts[:,1] > distance_Y+offset_y)[0] # find where centre + /2 target thickness
        verts = np.delete(verts, indicesYright, axis=0)

        # Remove points that are too close (Y-axis)
        indicesYleft = np.where(verts[:,1] < distance_Y-offset_y)[0] # find where centre - /2 target thickness
        verts = np.delete(verts, indicesYleft, axis=0)
    
    else:
        # Remove points that are too far (X-axis)
        indicesXfar = np.where(verts[:,0] > 4)[0] # find where X > 4 m
        verts = np.delete(verts, indicesXfar, axis=0)

        # Remove points that are too close (X-axis)
        indicesXclose = np.where(verts[:,0] <= 0)[0] # find where X <= 0 m
        verts = np.delete(verts, indicesXclose, axis=0)

    # *** For Y-axis, signs are resversed for some reason. Added top crop (will need to be updated for MECH mechanism. Too remove myself, is not needed if no manipulator)
    # Remove points that are too low (camera Y-axis). Is the target reflection
    indicesZlow = np.where(verts[:,2] > 0.15)[0] # find where Z < 0.15 m (15cm)
    verts = np.delete(verts, indicesZlow, axis=0)

    # *** TO REMOVE
    # Remove points that are too high (camera Y-axis).
    indicesZhigh = np.where(verts[:,2] < -0.5)[0] # find where Z > -0.15 m (15cm)
    verts = np.delete(verts, indicesZhigh, axis=0)  
    pcd_reduced.points = o3d.utility.Vector3dVector(verts)
    return pcd_reduced

def noise_reduction_pc(pc):
    reduced_pc = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)[0]
    return reduced_pc

######################################### 1: Initialization ######################################################
# Camera frame to SPOT frame transform
RotXNeg90 = np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]])
RotY90 = np.array([[0,0,1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]])
RotZ90 = np.array([[0,-1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]])
RotZNeg90 = np.array([[0,1,0,0],[-1,0,0,0],[0,0,1,0],[0,0,0,1]])
RotX90 = np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]])
transform_camera = np.dot(RotY90, RotZNeg90)
# transform_camera[0,3] = 0.15
# transform_camera[1,3] = 0.087
transform_model = np.dot(RotZ90,RotX90)

# RANSAC and ICP parameters
# (same for icp except ICP threshold 
voxel_size_RANSAC_coarse = 0.05  # (m) Change depending on point cloud size (0.05)
radius_normal_RANSAC_coarse = voxel_size_RANSAC_coarse * 4
radius_feature_RANSAC_coarse = voxel_size_RANSAC_coarse * 7 # (10)
distance_threshold_RANSAC_coarse = voxel_size_RANSAC_coarse * 1 # 1.5 (3)

voxel_size_RANSAC_fine = 0.01
radius_normal_RANSAC_fine = voxel_size_RANSAC_fine * 2
radius_feature_RANSAC_fine = voxel_size_RANSAC_fine * 5
distance_threshold_RANSAC_fine = voxel_size_RANSAC_fine * 2 # 1.5

voxel_size_ICP = 0.01 # (0.05)
distance_threshold_ICP = voxel_size_ICP * 0.4 # 0.4 (10)

ransac_confidence = 0.9999

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
    model_path = "E:/Ecole/Carleton/Fall2024/MAAE4907Q/Scripts/o3d_icp/Target_v39.stl"
    model_pc_path = "E:/Ecole/Carleton/Fall2024/MAAE4907Q/Scripts/o3d_icp/Target_v39_pc.npy"
elif platform == "Xavier":
    print("Not set for Xavier")
    exit(0)

# Model Initialization
try:
    print("Initializing Model: Start")
    origin = np.array([0, 0, 0])
    model = o3d.io.read_triangle_mesh(model_path)
    model_pcd = model.sample_points_poisson_disk(number_of_points=1000)
    # model_pcd = model_pcd.transform(transform_model)
    model_pcd = model_pcd.transform(RotZ90)
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
    
    # Apply the mask to filter out points inside the box
    filtered_points = points[mask]

    # If you want to create a new PointCloud object with the filtered points
    filtered_model_pcd = o3d.geometry.PointCloud()
    filtered_model_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    # verts = np.delete(model_pcd, mask, axis=0)
    
    model_down = filtered_model_pcd.voxel_down_sample(voxel_size_ICP) # Downsample and estimate normals for point cloud
    model_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal_RANSAC_fine, max_nn=30))
    # Compute FPFH features
    model_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        model_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature_RANSAC_fine, max_nn=100))
    
    model_down.paint_uniform_color([1, 0, 0]) # Red
    
    # Create the coordinate axis (standard Open3D axes with 1m length by default)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=origin)  # Size=0.3m (30cm)
    # X-axis -> Red (points along the X direction)
    # Y-axis -> Green (points along the Y direction)
    # Z-axis -> Blue (points along the Z direction)

    # Visualize the filtered point cloud along with the coordinate axes
    # o3d.visualization.draw_geometries([filtered_model_pcd, axis])
    print("Initializing Model: End")

except KeyboardInterrupt:
    print("\nCtrl+C pressed, exiting loop...")
    exit(0)
except Exception as e:
    print(e)
    print("Could not find Target Model.")
    exit(0)

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
                          "ICP_fitness", "ICP_rmse", "RANSAC_fitness", "RANSAC_rmse",
                          "ground_truth_x", "ground_truth_y", "ground_truth_theta"]
transform_array = [transform_array_header]

# Script Variables - Reading recording
savedrecording = []
savedrecording_frameNUM = 1 # If recording frame iterator to replace camera stream. Skip 0 (column title)
savedrecording_segNUM = 1 # Video segment number, used for both read and write recording

######################################### 3: Main Script #######################################################
print(f"Selected Mode: {Select_Mode}, Platform: {platform}, UDP Enabled: {send_package_udp}")

try:
    reset_counter = 1
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
            frame_data = read_live_data() # [verts, frame_timestamp]
            ground_truth_raw = receive_udp() # x(m), y(m), theta(rad), other
            ground_truth_x = ground_truth_raw[0]
            ground_truth_y = ground_truth_raw[1]
            ground_truth_theta = np.rad2deg(ground_truth_raw[2])
            # https://en.wikipedia.org/wiki/Rotation_matrix
            ground_truth_transform = np.array([[np.cos(ground_truth_theta),-np.sin(ground_truth_theta),0,ground_truth_x],[np.sin(ground_truth_theta),np.cos(ground_truth_theta),0,ground_truth_y],[0,0,1,0.15],[0,0,0,1]])
            if record_enabled:
                record_data(frame_data)
        
        # Reading a Recorded Video
        elif read_recording:
            frame_data, run_loop = read_recorded_data() # read_recorded_data sets run_loop used to flag the end of the recording
            if not(run_loop):
                break

        verts = frame_data[0]
        target_pcd.points = o3d.utility.Vector3dVector(verts)
        target_pcd.transform(transform_camera) 
        
        frame_timestamp = frame_data[1] # If recorded from old script index = 3, else index = 1

        # Data and noise reduction.
        if transform is None: # Coarse Reduction
            target_pcd = data_reduction_pcd(target_pcd)        
        else: # Refined Reduction
            target_pcd = data_reduction_pcd(target_pcd, target_transform=transform.transformation)
        # target_pcd = noise_reduction_pc(target_pcd)
        
        # RANSAC
        if transform is None:
            ransac_valid = False
            while not(ransac_valid):
                target_down = target_pcd.voxel_down_sample(voxel_size_RANSAC_coarse) # Downsample and estimate normals for point cloud
                try: 
                    target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal_RANSAC_coarse, max_nn=30))  
                    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                        target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature_RANSAC_coarse, max_nn=100))
                    transform = registration_RANSAC(model_down, target_down, model_fpfh, target_fpfh, distance_threshold_RANSAC_coarse, ransac_confidence)
                    RANSAC_transform_fitness = transform.fitness
                    RANSAC_transform_rmse = transform.inlier_rmse
                    print(f"{time.strftime('%H:%M:%S')} RANSAC Success")
                
                    # Logic to validate ICP output to bee added in the ICP criteria. Check what is the error output of registration_icp
                    # transform_relative = np.dot(transform_camera,transform.transformation) # Transform from camera frame to inertial standard
                    RANSAC_pose = estimate_pose(transform.transformation) # X, Y, Z, Rx, Ry, Rz (millimeters and degrees, relative Chaser-Target)
                    RANSAC_timestamp = time.strftime('%H:%M:%S')
                    print(RANSAC_pose)
                    if visualizer_enabled:
                        draw_result(model_down, target_down, transform.transformation)
                    target_pcd = data_reduction_pcd(target_pcd)
                    """UNCOMMENT BELOW TO TEST RANDSAC ONLY""" 
                    # transform = None
                    # continue
                
                    # Check if RANSAC output is valid
                    valid_Rx = RANSAC_pose[3]%180 < 5 or RANSAC_pose[3]%180 > 175
                    valid_Ry = RANSAC_pose[3]%180 < 5 or RANSAC_pose[3]%180 > 175
                    ransac_valid = valid_Rx and valid_Ry

                except KeyboardInterrupt:
                    print("\nCtrl+C pressed, exiting loop...")
                    break          
                except Exception as e:
                    print(e)
                    transform = None
                    print(f"{time.strftime('%H:%M:%S')} RANSAC Fail")
                    continue

        # ICP REFINEMENT
        try:
            target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            transform = registration_ICP_point2plane(model_down, target_pcd, distance_threshold_ICP, transform)
            ICP_transform_fitness = transform.fitness
            ICP_transform_rmse = transform.inlier_rmse
            print(f"{time.strftime('%H:%M:%S')} ICP Success")
            
            # Logic to validate ICP output to bee added in the ICP criteria. Check what is the error output of registration_icp
            # transform_relative = np.dot(transform_camera,transform.transformation) # Transform from camera frame to inertial standard
            ICP_pose = estimate_pose(transform.transformation) # X, Y, Z, Rx, Ry, Rz (meters and degrees, relative Chaser-Target)
            ICP_timestamp = time.strftime('%H:%M:%S')
            print(ICP_pose)
        except Exception as e:
            print(e)
            transform = None
            print(f"{time.strftime('%H:%M:%S')} ICP Fail")
            continue
        
        transform_result_x = ICP_pose[0]+0.15
        transform_result_y = ICP_pose[1]+0.087
        transform_result_theta = ICP_pose[5]

        try:
            perc_error_x = (transform_result_x-ground_truth_x)*100/ground_truth_x
            perc_error_y =(transform_result_y-ground_truth_y)*100/ground_truth_y
            perc_error_theta =(transform_result_theta-ground_truth_theta)*100/ground_truth_theta
        finally:
            print(f"% Error X = {perc_error_x}, Error Y = {perc_error_y}, Error Theta = {perc_error_theta}.")
        
        if send_package_udp:
            # + values below to account for camera offset
            x, y, theta, rmse_ICP=  ICP_pose[0]+0.15, ICP_pose[1]+0.087, ICP_pose[5], ICP_transform_rmse
            udp_data = bytearray(struct.pack("fffff",x,y,np.deg2rad(theta),rmse_ICP, reset_counter))
            try:
                sock.sendto(udp_data,server_address)
                print(f"{time.strftime('%H:%M:%S')} UPD package sent: x={x}, y={y}, theta={theta}.")
            except Exception as e:
                print(e)
                print(f"{time.strftime('%H:%M:%S')} Failed to send UDP package")

        transform_array.append([frame_timestamp,
                                ICP_timestamp, transform_result_x, transform_result_y, ICP_pose[2], ICP_pose[3], ICP_pose[4], ICP_pose[5],
                                RANSAC_timestamp, RANSAC_pose[0], RANSAC_pose[1], RANSAC_pose[2], RANSAC_pose[3], RANSAC_pose[4], RANSAC_pose[5],
                                ICP_transform_fitness, ICP_transform_rmse, RANSAC_transform_fitness, RANSAC_transform_rmse,
                                ground_truth_x, ground_truth_y, ground_truth_theta])

        if visualizer_enabled:
            draw_result(model_down, target_pcd, transform.transformation)
        
        # if not(reset_counter%20):
        #    print("reset transform")
        #    transform = None
        reset_counter +=1

finally:
    np.savetxt(pose_estimation_path, transform_array, fmt="%s", delimiter=',')
    if not read_recording:
        pipeline.stop()
        profile = pipeline = None

    if record_enabled:
        recording = np.asarray(recording, dtype=object)
        np.save(record_video_path+str(savedrecording_segNUM)+".npy", recording, allow_pickle=True)

    if send_package_udp:
        sock.close()
