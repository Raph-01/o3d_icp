
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:49:36 2024

@author: Vision's Awesome Core Team, with the help of Alex
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

# Class Creation
## Custom Classes
class StoppableCameraThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

    def __init__(self, pipeline, *args, **kwargs):
        super(StoppableCameraThread, self).__init__(*args, **kwargs)
        self._stop_event = threading.Event()
        self.pointcloud_frame = None
        self.mypipeline = pipeline

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def pull_pointcloud(self):
        # print("trying Pull"); sys.stdout.flush()
        return self.pointcloud_frame

    def run(self):
        # print("thread started"); sys.stdout.flush()
        while not self._stop_event.is_set():
            # print("thread working"); sys.stdout.flush()
            time.sleep(1)
            frames = self.mypipeline.wait_for_frames()
            depth = frames.get_depth_frame()
            if not depth:
                continue
            pc = rs.pointcloud()
            self.pointcloud_frame = pc.calculate(depth)

## PyRealSense2 Class
### https://github.com/IntelRealSense/librealsense/blob/development/wrappers/python/examples/opencv_pointcloud_viewer.py
class AppState:

    def __init__(self, *args, **kwargs):
        self.WIN_NAME = 'RealSense'
        self.pitch, self.yaw = math.radians(-10), math.radians(-15)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 1
        self.scale = True
        self.color = False

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)

# PyRealSense2 Function

def mouse_cb(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:
        state.mouse_btns[0] = True

    if event == cv2.EVENT_LBUTTONUP:
        state.mouse_btns[0] = False

    if event == cv2.EVENT_RBUTTONDOWN:
        state.mouse_btns[1] = True

    if event == cv2.EVENT_RBUTTONUP:
        state.mouse_btns[1] = False

    if event == cv2.EVENT_MBUTTONDOWN:
        state.mouse_btns[2] = True

    if event == cv2.EVENT_MBUTTONUP:
        state.mouse_btns[2] = False

    if event == cv2.EVENT_MOUSEMOVE:

        h, w = out.shape[:2]
        dx, dy = x - state.prev_mouse[0], y - state.prev_mouse[1]

        if state.mouse_btns[0]:
            state.yaw += float(dx) / w * 2
            state.pitch -= float(dy) / h * 2

        elif state.mouse_btns[1]:
            dp = np.array((dx / w, dy / h, 0), dtype=np.float32)
            state.translation -= np.dot(state.rotation, dp)

        elif state.mouse_btns[2]:
            dz = math.sqrt(dx**2 + dy**2) * math.copysign(0.01, -dy)
            state.translation[2] += dz
            state.distance -= dz

    if event == cv2.EVENT_MOUSEWHEEL:
        dz = math.copysign(0.1, flags)
        state.translation[2] += dz
        state.distance -= dz

    state.prev_mouse = (x, y)

def project(v):
    """project 3d vector array to 2d"""
    h, w = out.shape[:2]
    view_aspect = float(h)/w

    # ignore divide by zero for invalid depth
    with np.errstate(divide='ignore', invalid='ignore'):
        proj = v[:, :-1] / v[:, -1, np.newaxis] * \
            (w*view_aspect, h) + (w/2.0, h/2.0)

    # near clipping
    znear = 0.03
    proj[v[:, 2] < znear] = np.nan
    return proj


def view(v):
    """apply view transformation on vector array"""
    return np.dot(v - state.pivot, state.rotation) + state.pivot - state.translation


def line3d(out, pt1, pt2, color=(0x80, 0x80, 0x80), thickness=1):
    """draw a 3d line from pt1 to pt2"""
    p0 = project(pt1.reshape(-1, 3))[0]
    p1 = project(pt2.reshape(-1, 3))[0]
    if np.isnan(p0).any() or np.isnan(p1).any():
        return
    p0 = tuple(p0.astype(int))
    p1 = tuple(p1.astype(int))
    rect = (0, 0, out.shape[1], out.shape[0])
    inside, p0, p1 = cv2.clipLine(rect, p0, p1)
    if inside:
        cv2.line(out, p0, p1, color, thickness, cv2.LINE_AA)


def grid(out, pos, rotation=np.eye(3), size=1, n=10, color=(0x80, 0x80, 0x80)):
    """draw a grid on xz plane"""
    pos = np.array(pos)
    s = size / float(n)
    s2 = 0.5 * size
    for i in range(0, n+1):
        x = -s2 + i*s
        line3d(out, view(pos + np.dot((x, 0, -s2), rotation)),
               view(pos + np.dot((x, 0, s2), rotation)), color)
    for i in range(0, n+1):
        z = -s2 + i*s
        line3d(out, view(pos + np.dot((-s2, 0, z), rotation)),
               view(pos + np.dot((s2, 0, z), rotation)), color)


def axes(out, pos, rotation=np.eye(3), size=0.075, thickness=2):
    """draw 3d axes"""
    line3d(out, pos, pos +
           np.dot((0, 0, size), rotation), (0xff, 0, 0), thickness)
    line3d(out, pos, pos +
           np.dot((0, size, 0), rotation), (0, 0xff, 0), thickness)
    line3d(out, pos, pos +
           np.dot((size, 0, 0), rotation), (0, 0, 0xff), thickness)


def frustum(out, intrinsics, color=(0x40, 0x40, 0x40)):
    """draw camera's frustum"""
    orig = view([0, 0, 0])
    w, h = intrinsics.width, intrinsics.height

    for d in range(1, 6, 2):
        def get_point(x, y):
            p = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)
            line3d(out, orig, view(p), color)
            return p

        top_left = get_point(0, 0)
        top_right = get_point(w, 0)
        bottom_right = get_point(w, h)
        bottom_left = get_point(0, h)

        line3d(out, view(top_left), view(top_right), color)
        line3d(out, view(top_right), view(bottom_right), color)
        line3d(out, view(bottom_right), view(bottom_left), color)
        line3d(out, view(bottom_left), view(top_left), color)


def pointcloud(out, verts, texcoords, color, painter=True):
    """draw point cloud with optional painter's algorithm"""
    if painter:
        # Painter's algo, sort points from back to front

        # get reverse sorted indices by z (in view-space)
        # https://gist.github.com/stevenvo/e3dad127598842459b68
        v = view(verts)
        s = v[:, 2].argsort()[::-1]
        proj = project(v[s])
    else:
        proj = project(view(verts))

    if state.scale:
        proj *= 0.5**state.decimate

    h, w = out.shape[:2]

    # proj now contains 2d image coordinates
    j, i = proj.astype(np.uint32).T

    # create a mask to ignore out-of-bound indices
    im = (i >= 0) & (i < h)
    jm = (j >= 0) & (j < w)
    m = im & jm

    cw, ch = color.shape[:2][::-1]
    if painter:
        # sort texcoord with same indices as above
        # texcoords are [0..1] and relative to top-left pixel corner,
        # multiply by size and add 0.5 to center
        v, u = (texcoords[s] * (cw, ch) + 0.5).astype(np.uint32).T
    else:
        v, u = (texcoords * (cw, ch) + 0.5).astype(np.uint32).T
    # clip texcoords to image
    np.clip(u, 0, ch-1, out=u)
    np.clip(v, 0, cw-1, out=v)

    # perform uv-mapping
    out[i[m], j[m]] = color[u[m], v[m]]

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

"""
USER INPUT START
"""
### Choose read from file or live camera capture, or record live capture ###
Enable_Open3D_Visualiser = False
Enable_OpenCV_Visualiser = False

Select_Mode = "Read" # "Read" "Live" "Record Live"
control_fps = False
max_recorded_frame = 50 # Will record until max frame reached, then exit script. Condition checked only if Select_Mode == "Record Live"

# Orin Paths
## Video: "/home/spot-vision/Documents/o3d_icp/Recording/myrecording"
## Video Intrinsics: "/home/spot-vision/Documents/o3d_icp/Recording/myrecordingintrinsics.npy"

read_video_path = "E:/Ecole/Carleton/Fall 2024/MAAE 4907 Q (Autonomous Spacecraft Vehicule)/Scripts/o3d_icp/recording/myrecording" # Path String with file name, without index nor extension
read_video_intrinsics_path = "E:/Ecole/Carleton/Fall 2024/MAAE 4907 Q (Autonomous Spacecraft Vehicule)/Scripts/o3d_icp/recording/myrecordingintrinsics.npy" # Path String with file name

record_video_path = "E:/Ecole/Carleton/Fall 2024/MAAE 4907 Q (Autonomous Spacecraft Vehicule)/Scripts/o3d_icp/recording/myrecording" # Path String with file name, without index nor extension
record_video_intrinsics_path = "E:/Ecole/Carleton/Fall 2024/MAAE 4907 Q (Autonomous Spacecraft Vehicule)/Scripts/o3d_icp/recording/myrecordingintrinsics.npy" # Path String with file name

pose_estimation_path = "E:/Ecole/Carleton/Fall 2024/MAAE 4907 Q (Autonomous Spacecraft Vehicule)/Scripts/o3d_icp/pose_estimation/pose_estimation.csv" # Path String with file name with extension

source = o3d.io.read_triangle_mesh("E:/Ecole/Carleton/Fall 2024/MAAE 4907 Q (Autonomous Spacecraft Vehicule)/Scripts/o3d_icp/target_model/Target_v10.stl")
"""
USER INPUT END
"""

# RANSAC and ICP parameters
voxel_size = 0.05  # (m) Change depending on point cloud size
radius_normal = voxel_size * 2
radius_feature = voxel_size * 10
rmse_threshold = 0.05  # User-defined RMSE threshold
# num_ransac_iterations = 5

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
state = AppState() # Is used somewhere else than ocv visualizer
pc = rs.pointcloud()
decimate = rs.decimation_filter()
decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)
colorizer = rs.colorizer()

# Initialize OpenCV Visualiser
if Enable_OpenCV_Visualiser:
    cv2.namedWindow(state.WIN_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow(state.WIN_NAME, w, h)
    cv2.setMouseCallback(state.WIN_NAME, mouse_cb)

# Initialize Open3D Visualiser
if Enable_Open3D_Visualiser:
    vis = o3d.visualization.Visualizer()
    vis.create_window("Tests")
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=origin)

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
        if not state.paused and not read_recording:
            # Wait for a coherent pair of frames: depth and color       
            frames = pipeline.wait_for_frames()

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            frame_timestamp = frames.get_timestamp()
            # depth_frame = decimate.process(depth_frame)

            # Grab new intrinsics (may be changed by decimation)
            depth_intrinsics = rs.video_stream_profile(
                depth_frame.profile).get_intrinsics()
            w, h = depth_intrinsics.width, depth_intrinsics.height # The Decimate reduces resolution by 2 folds

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            depth_colormap = np.asanyarray(
                colorizer.colorize(depth_frame).get_data())

            if state.color:
                mapped_frame, color_source = color_frame, color_image
            else:
                mapped_frame, color_source = depth_frame, depth_colormap

            points = pc.calculate(depth_frame)
            pc.map_to(mapped_frame)

            # Pointcloud data to arrays
            v, t = points.get_vertices(), points.get_texture_coordinates()
            verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
            texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv
        
        # Reading a Recorded Video
        elif not state.paused and read_recording:
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

        if record_enabled:
            recording.append([verts, texcoords, color_source, frame_timestamp])
            
            if len(recording) >=10:
                recording = np.asarray(recording, dtype=object)
                np.save(record_video_path+str(recording_segNUM)+".npy", recording, allow_pickle=True)
                recording_segNUM += 1
                recording = [["Depth Frame", "Color Frame", "Texture Coordinate", "Time Stamp"]]

            if i > max_recorded_frame:
                run_loop = False   
            i+=1
            print(i)

        # Pointcloud Vertices (verts) processing 
        ## remove background (index z not within  0-2m and delete that point)
        #indicesZ = np.where(not(0 < verts[:,2] > 2))[0] # find where 0 m > z > 2 m
        indicesZ = np.where(verts[:,2] > 2)[0] # find where z > 1.7 m
        verts = np.delete(verts, indicesZ, axis=0)
        
        target_pcd.points = o3d.utility.Vector3dVector(verts)
        target_pcd = target_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)[0]
        target_down = target_pcd.voxel_down_sample(voxel_size) # Downsample and estimate normals for point cloud
        
        target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))  
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        
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
            best_fitness = transform_fitness
            best_rmse = transform_rmse
            best_transformation = transform.transformation


        # Check if a valid transformation was found
        if best_transformation is not None:
            print(transform.transformation)
            transform_relative = np.dot(transform_camera,transform.transformation) # Transform from camera frame to inertial standard
            print(transform_relative)
            transform_package = [transform_relative[0,3],transform_relative[1,3], transform_relative[1,1]] # X, Y, rotation of X-axis about Z (rad) [relative chaser-target]
            print(transform_package)
            transform_array.append([frame_timestamp, transform_package[0], transform_package[1],transform_package[2], transform_fitness, transform_rmse]) # SEND TRANSFORM ARRAY TO GNC
            print("ICP SUCCESS")
        else:
            transform = None # TO REVIEW LOGIC, DO WE WANT TO JUMP STRAIGHT BACK TO RANSAC OR TRY ICP AGAIN. Do we send data to GNC anyway? With fitness score?
            print("ICP FAIL")

        # Vizualizers
        ## o3d
        if Enable_Open3D_Visualiser and best_transformation is not None:
            inverse_best_transformation = np.linalg.inv(best_transformation)
            # Apply the best transformation to the target point cloud
            transformed_target_pcd = target_pcd.transform(inverse_best_transformation)
            vis.add_geometry(target_pcd)
            vis.add_geometry(transformed_target_pcd)
            target_pcd.clear()
            vis.update_geometry(target_pcd)
            vis.update_geometry(transformed_target_pcd)
            vis.poll_events()
            vis.update_renderer()

        ## OpenCV
        if Enable_OpenCV_Visualiser:
            now = time.time()
            out.fill(0)
            grid(out, (0, 0.5, 1), size=1, n=10)
            # frustum(out, depth_intrinsics)
            axes(out, view([0, 0, 0]), state.rotation, size=0.1, thickness=1)

            if not state.scale or out.shape[:2] == (h, w):
                pointcloud(out, verts, texcoords, color_source)
            else:
                tmp = np.zeros((h, w, 3), dtype=np.uint8)
                pointcloud(tmp, verts, texcoords, color_source)
                tmp = cv2.resize(
                    tmp, out.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
                np.putmask(out, tmp > 0, tmp)   

            if any(state.mouse_btns):
                axes(out, view(state.pivot), state.rotation, thickness=4)

            dt = time.time() - now

            cv2.setWindowTitle(
                state.WIN_NAME, "RealSense (%dx%d) %dFPS (%.2fms) %s" %
                (w, h, 1.0/dt, dt*1000, "PAUSED" if state.paused else ""))

            cv2.imshow(state.WIN_NAME, out)
            key = cv2.waitKey(1)

            if key == ord("p"):
                state.paused ^= True

            if key == ord("d"):
                state.decimate = (state.decimate + 1) % 3
                decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)

            if key == ord("z"):
                state.scale ^= True

            if key == ord("c"):
                state.color ^= True

            if key == ord("s"):
                cv2.imwrite('./out.png', out)

            if key == ord("e"):
                points.export_to_ply('./out.ply', mapped_frame)

            if key in (27, ord("q")) or cv2.getWindowProperty(state.WIN_NAME, cv2.WND_PROP_AUTOSIZE) < 0:
                break

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
