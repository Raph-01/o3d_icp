
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:49:36 2024


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
remove_tank = True # from model
sequence = 20

######################################### 3: Functions #######################################################
def read_specific_frame(seg_num, frame_num):
    try:
        savedrecording = np.load(read_video_path+str(seg_num)+".npy", allow_pickle=True)
        my_frame_data = savedrecording[frame_num] # ["Depth Verts", "Time Stamp"]
        print(f"Video segment {seg_num}, frame {frame_num} successfully loaded")
    except Exception as e:
        print(e)
    return my_frame_data

def draw_result(source_pcd, target_pcd, transformation):
    source_pcd_temp = copy.deepcopy(source_pcd)
    target_pcd_temp = copy.deepcopy(target_pcd)
    source_pcd_temp.transform(transformation)
    source_pcd_temp.paint_uniform_color([0, 0, 1]) # Blue
    target_pcd_temp.paint_uniform_color([1, 0, 0]) # Red
    # X-axis: Red
    # Y-axis: Green
    # Z-axis: Blue
    o3d.visualization.draw_geometries([source_pcd_temp,target_pcd_temp, axis])

def model_initialization(point_num, transform_model):
    global remove_tank
    origin = np.array([0, 0, 0])
    model = o3d.io.read_triangle_mesh(model_path)
    model_pcd = model.sample_points_poisson_disk(number_of_points=point_num)
    model_pcd = model_pcd.transform(transform_model)
    model_pcd.scale(0.001, origin)

    points = np.asarray(model_pcd.points)

    # Remove inside
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
    
    # Remove Tank
    if remove_tank:
        half_width = 0.15
        xmin = -half_width
        xmax = half_width
        ymin = -half_width
        ymax = 0.02
        zmin = 2*half_width +0.02
        zmax = 3*half_width

        mask = np.logical_or(
        np.logical_or(
            (filtered_points[:, 0] < xmin) | (filtered_points[:, 0] > xmax),  # Check x-range
            (filtered_points[:, 1] < ymin) | (filtered_points[:, 1] > ymax)   # Check y-range
            ),
            (filtered_points[:, 2] < zmin) | (filtered_points[:, 2] > zmax)  # Check z-range
            )
        filtered_points = filtered_points[mask]

    filtered_model_pcd = o3d.geometry.PointCloud()
    filtered_model_pcd.points = o3d.utility.Vector3dVector(filtered_points)

    return filtered_model_pcd

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

def read_recorded_data_txt(sequence, frameNUM):
    end_recording = False
    point_cloud = o3d.geometry.PointCloud()
    my_frame_data = None
    try:
        point_cloud_txt = np.loadtxt(f"{read_data_path_txt}PC_{sequence}/point_cloud_{frameNUM:04d}.txt", delimiter=" ")
        point_cloud.points = o3d.utility.Vector3dVector(point_cloud_txt[:, :3])
        verts = np.array(point_cloud.points)
        my_frame_data = [verts, 0]
        print(f"Frame {frameNUM} successfully loaded")
    except:
        end_recording = True

    return my_frame_data, not(end_recording)

def read_recorded_ground_truth_txt(sequence, frameNUM):
    # x(m), y(m), theta(rad), other
    ground_truth_raw = np.loadtxt(f"{read_data_path_txt}Phase_{sequence}/phasespace_data_{frameNUM:04d}.txt", delimiter=" ")
    ground_truth_raw = ground_truth_raw[:3]
    return ground_truth_raw

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

def registration_ICP_point2plane(model_pcd, target_pcd, distance_threshold_ICP, transform):
    transform_ICP = o3d.pipelines.registration.registration_icp(
            model_pcd, target_pcd, distance_threshold_ICP, transform.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return transform_ICP

def registration_RANSAC(model_down, target_down, model_fpfh, target_fpfh, distance_threshold, confidence):
    global visualizer_enabled

    transform_RANSAC = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        model_down, target_down, model_fpfh, target_fpfh, True,
        distance_threshold, 
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, confidence))
 
    # if visualizer_enabled:
        # draw_result(model_down, target_down, transform_RANSAC.transformation)
    return transform_RANSAC

def estimate_pose(transformation):
    # Rz(gamma) Ry(beta) Rx(alpha) [degrees]
    gamma = np.arctan2(transformation[1,0], transformation[0,0])
    beta = np.arctan2(-transformation[2,0], np.sqrt(transformation[2,1]**2+transformation[2,2]**2))
    alpha = np.arctan2(transformation[2,1], transformation[2,2])
    # In Deg
    beta = round(np.rad2deg(beta),3) # Ry
    alpha = round(np.rad2deg(alpha),3) # Rx
    gamma = round(np.rad2deg(gamma),3) # Rz

    # Translations [m]
    x = round(transformation[0, 3],4)
    y = round(transformation[1, 3],4)
    z = round(transformation[2, 3],4)
    return [x, y, z, alpha, beta, gamma]

def pcd_downsample_fpfh(pcd, voxel_size, radius_normal, radius_feature):
    model_down = pcd.voxel_down_sample(voxel_size) # Downsample and estimate normals for point cloud
    model_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    # Compute FPFH features
    model_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        model_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return model_down, model_fpfh

def data_reduction_pcd(pcd):
    pcd_reduced = o3d.geometry.PointCloud()
    verts = np.array(pcd.points)

    # Remove points that are too far (X-axis)
    indicesXfar = np.where(verts[:,0] > 4)[0] # find where X > 4 (m)
    verts = np.delete(verts, indicesXfar, axis=0)

    # Remove points that are too close (X-axis)
    indicesXclose = np.where(verts[:,0] <= 0)[0] # find where X <= 0 (m)
    verts = np.delete(verts, indicesXclose, axis=0)

    # Remove points that are too high (Z-axis). Is the target reflection
    # indicesZhigh = np.where(verts[:,2] > 0.25)[0] # find where Z > threshold (m)
    # verts = np.delete(verts, indicesZhigh, axis=0)

    # Remove points that are too low (Z-axis).
    indicesZlow = np.where(verts[:,2] < -0.115)[0] # find where Z < threshold (m)
    verts = np.delete(verts, indicesZlow, axis=0)  
    
    pcd_reduced.points = o3d.utility.Vector3dVector(verts)
    return pcd_reduced

######################################### 1: Initialization ######################################################
# Camera frame to SPOT frame transform
Identity = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
RotXNeg90 = np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]])
RotY90 = np.array([[0,0,1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]])
RotZ90 = np.array([[0,-1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]])
RotZ180 = np.array([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]])
RotZNeg90 = np.array([[0,1,0,0],[-1,0,0,0],[0,0,1,0],[0,0,0,1]])
RotZNeg180 = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
RotX90 = np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]])
transform_camera = np.dot(RotY90, RotZNeg90)
# transform_camera[0,3] = 0.15
# transform_camera[1,3] = 0.087

origin = np.array([0, 0, 0])
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=origin)

# RANSAC and ICP parameters
# Promising test result: values from 712

model_sampling = 10000 # number of points
voxel_size_RANSAC = 0.01  # (m) Change depending on point cloud size (0.05)
radius_normal_RANSAC = voxel_size_RANSAC * 2
radius_feature_RANSAC = voxel_size_RANSAC * 3 # (10)
distance_threshold_RANSAC = voxel_size_RANSAC * 8 # 4
ransac_confidence = 0.9999 # (0.999)

# Duplicated from RANSAC
voxel_size_ICP = 0.01 # (0.05)
distance_threshold_ICP = voxel_size_ICP * 8 # 4

# Set script states
read_recording = True if Select_Mode == "Read" else (False)

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

    result_pc_path = "E:/Ecole/Carleton/Fall2024/MAAE4907Q/Scripts/o3d_icp/Data/test/"
    read_data_path_txt = "E:/Ecole/Carleton/Fall2024/MAAE4907Q/Scripts/o3d_icp/Data/"


elif platform == "Xavier":
    print("Not set for Xavier")
    exit(0)

# Model Initialization
try:
    print("Initializing Model: Start")
    model_pcd = model_initialization(model_sampling, RotZ180)
    model_down, model_fpfh = pcd_downsample_fpfh(model_pcd, voxel_size_RANSAC, radius_normal_RANSAC, radius_feature_RANSAC) # Downsample and estimate normals for point cloud
    model_down.paint_uniform_color([1, 0, 0]) # Red
    print("Initializing Model: End")

except KeyboardInterrupt:
    print("\nCtrl+C pressed, exiting loop...")
    exit(0)
except Exception as e:
    print(e)
    print("Could not find Target Model.")
    exit(0)

if visualizer_enabled:
    o3d.visualization.draw_geometries([model_down, axis])

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
ground_truth_x = 9999
ground_truth_y = 9999
ground_truth_theta = 9999

target_pcd = o3d.geometry.PointCloud()
recording_header = ["Depth Verts", "Time Stamp"]
recording = [recording_header]
transform = None
transform_array_header = ["sample", "frame",
                          "ICP_timestamp", "time_ICP", "ICP_x", "ICP_y", "ICP_z", "ICP_Rx", "ICP_Ry", "ICP_Rz",
                          "RANSAC_timestamp", "ransac_count", "time_RANSAC", "RANSAC_x", "RANSAC_y", "RANSAC_z", "RANSAC_Rx", "RANSAC_Ry", "RANSAC_Rz",
                          "ICP_fitness", "ICP_rmse", "RANSAC_fitness", "RANSAC_rmse",
                          "ground_truth_x", "ground_truth_y", "ground_truth_theta"]
transform_array = [transform_array_header]

# For .txt data read
frame = 371 # Starting frame

######################################### 3: Main Script #######################################################
print(f"Selected Mode: {Select_Mode}, Platform: {platform}, UDP Enabled: {send_package_udp}")

try:
    for sample in range(25):
        frame = 371
        transform = None
        
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
            
            # Reading a Recorded Video
            elif read_recording:
                frame_data, run_loop = read_recorded_data_txt(sequence, frame) # read_recorded_data sets run_loop used to flag the end of the recording
                if not(run_loop):
                    print(f"End sample. Starting Sample {sample}")
                    continue
                ground_truth_x, ground_truth_y, ground_truth_theta = read_recorded_ground_truth_txt(sequence, frame)
                ground_truth_theta = np.rad2deg(ground_truth_theta)
                
                """
                frame_data = read_specific_frame(1, 1)
                if reset_counter > 100:
                    break
                """
            verts = frame_data[0]
            target_pcd.points = o3d.utility.Vector3dVector(verts)
            target_pcd.transform(transform_camera)
            target_pcd = data_reduction_pcd(target_pcd)
            target_down, target_fpfh = pcd_downsample_fpfh(target_pcd, voxel_size_RANSAC, radius_normal_RANSAC, radius_feature_RANSAC)
            # if visualizer_enabled:
            #    o3d.visualization.draw_geometries([target_down, axis])

            frame_timestamp = frame_data[1]
            
            # RANSAC
            # if True:
            ransac_count = 0
            if transform is None:
                try: 
                    valid_ransac = False
                    while not(valid_ransac):    
                        ransac_count += 1
                        ransac_start = time.time()
                        transform = registration_RANSAC(model_down, target_down, model_fpfh, target_fpfh, distance_threshold_RANSAC, ransac_confidence)
                        RANSAC_transform_fitness = transform.fitness
                        RANSAC_transform_rmse = transform.inlier_rmse           
                        RANSAC_pose = estimate_pose(transform.transformation) # X, Y, Z, Rx, Ry, Rz (millimeters and degrees, relative Chaser-Target)
                        if -10 < RANSAC_pose[3] < 10 and -10 < RANSAC_pose[4] < 10 and 0 < RANSAC_pose[5] < 180:
                            valid_ransac = True
                            print(f"{time.strftime('%H:%M:%S')} RANSAC Success.")
                        else:
                            print(f"RANSAC #{ransac_count} invalid, retrying.")
                            print(f"Rx = {RANSAC_pose[3]}, Ry = {RANSAC_pose[4]}, Rz = {RANSAC_pose[5]}")
                            continue
                        RANSAC_timestamp = time.strftime('%H:%M:%S')
                        time_RANSAC = time.time() - ransac_start
                        print(RANSAC_pose)

                except KeyboardInterrupt:
                    print("\nCtrl+C pressed, exiting loop...")
                    break          
                except Exception as e:
                    print(e)
                    transform = None
                    print(f"{time.strftime('%H:%M:%S')} RANSAC Fail")
                    continue
            
            if visualizer_enabled:
                draw_result(model_down, target_down, transform.transformation)

            # ICP REFINEMENT
            try:
                icp_start = time.time()
                target_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
                transform = registration_ICP_point2plane(model_down, target_down, distance_threshold_ICP, transform)
                ICP_transform_fitness = transform.fitness
                ICP_transform_rmse = transform.inlier_rmse
                print(f"{time.strftime('%H:%M:%S')} ICP Success")
                
                # Logic to validate ICP output to bee added in the ICP criteria. Check what is the error output of registration_icp
                # transform_relative = np.dot(transform_camera,transform.transformation) # Transform from camera frame to inertial standard
                ICP_pose = estimate_pose(transform.transformation) # X, Y, Z, Rx, Ry, Rz (meters and degrees, relative Chaser-Target)
                ICP_timestamp = time.strftime('%H:%M:%S')
                time_ICP = time.time() - icp_start
                print(ICP_pose)
            except Exception as e:
                print(e)
                transform = None
                print(f"{time.strftime('%H:%M:%S')} ICP Fail")
                continue
            
            # RANSAC
            RANSAC_transform_result_x = RANSAC_pose[0]+0.15
            RANSAC_transform_result_y = RANSAC_pose[1]+0.087
            RANSAC_transform_result_theta = RANSAC_pose[5]
            # ICP
            transform_result_x = ICP_pose[0]+0.15
            transform_result_y = ICP_pose[1]+0.087
            transform_result_theta = ICP_pose[5]
            
            try:
                error_x = (transform_result_x-ground_truth_x)*1000 # mm
                error_y =(transform_result_y-ground_truth_y)*1000 # mm
                error_theta =(transform_result_theta-ground_truth_theta)

                RANSAC_error_x = (RANSAC_transform_result_x-ground_truth_x)*1000 # mm
                RANSAC_error_y =(RANSAC_transform_result_y-ground_truth_y)*1000 # mm
                RANSAC_error_theta =(RANSAC_transform_result_theta-ground_truth_theta)

                diff_error_x = abs(RANSAC_error_x) - abs(error_x)
                diff_error_y = abs(RANSAC_error_y) - abs(error_y)
                diff_error_theta = abs(RANSAC_error_theta) - abs(error_theta)

                print(f"ICP: X = {transform_result_x}, Y = {transform_result_y}, Theta = {transform_result_theta}")
                print(f"Phase: X = {ground_truth_x}, Y = {ground_truth_y}, Theta = {ground_truth_theta}")
                print(f"Error ICP: X = {error_x}mm, Y = {error_y} mm, Theta = {error_theta}deg.")
                # print(f"Diff Error: X = {diff_error_x}mm, Y = {diff_error_y} mm, Theta = {diff_error_theta}deg.")

            except Exception as e:
                print(e)
            
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

            transform_array.append([sample, frame,
                                    ICP_timestamp, time_ICP, transform_result_x, transform_result_y, ICP_pose[2], ICP_pose[3], ICP_pose[4], ICP_pose[5],
                                    RANSAC_timestamp, ransac_count, time_RANSAC,  RANSAC_transform_result_x, RANSAC_transform_result_y, RANSAC_pose[2], RANSAC_pose[3], RANSAC_pose[4], RANSAC_pose[5],
                                    ICP_transform_fitness, ICP_transform_rmse, RANSAC_transform_fitness, RANSAC_transform_rmse,
                                    ground_truth_x, ground_truth_y, ground_truth_theta])

            if visualizer_enabled:
                draw_result(model_down, target_down, transform.transformation)
            
            # if not(reset_counter%20):
            #    print("reset transform")
            #    transform = None
            
            model_pcd_temp = copy.deepcopy(model_down)
            model_pcd_temp.transform(transform.transformation)
            vert_target_pcd = np.asarray(target_down.points)
            vert_model_pcd_temp = np.asarray(model_pcd_temp.points)
            result_array_pc = [vert_model_pcd_temp,vert_target_pcd]
            result_array_pc = np.asarray(result_array_pc, dtype=object)
            np.save(f"{result_pc_path}transform_result_{sample}_{frame}.npy", result_array_pc, allow_pickle=True)

            reset_counter +=1
            frame += 1

except Exception as e:
    print(e)

finally:
    np.savetxt(pose_estimation_path, transform_array, fmt="%s", delimiter=',')
    if not read_recording:
        pipeline.stop()
        profile = pipeline = None

    if send_package_udp:
        sock.close()