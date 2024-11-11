# -*- coding: utf-8 -*-

# Import Statements
import pyrealsense2 as rs
import numpy as np
import time
import threading
import cv2
import sys

# Class Creation
class StoppableCameraThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

    def __init__(self, pipeline, *args, **kwargs):
        super(StoppableCameraThread, self).__init__(*args, **kwargs)
        self._stop_event = threading.Event()
        self.depth_frame = None
        self.mypipeline = pipeline

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def pull_frame(self):
        print("trying Pull"); sys.stdout.flush()
        return self.depth_frame

    def run(self):
        print("thread started"); sys.stdout.flush()
        while not self._stop_event.is_set():
            print("thread working"); sys.stdout.flush()
            time.sleep(1)
            frames = self.mypipeline.wait_for_frames()
            self.depth_frame = frames.get_depth_frame()
 
    def get_intrinsics(self):
            # Get intrinsics for point cloud calculations. Requires stream to be started
            frames = self.mypipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
            fx, fy = depth_intrinsics.fx, depth_intrinsics.fy
            cx, cy = depth_intrinsics.ppx, depth_intrinsics.ppy
            return(fx, fy, cx, cy)


# Script
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

depthstream = StoppableCameraThread(pipeline)
depthstream.start()
time.sleep(5)
fx, fy, cx, cy = depthstream.get_intrinsics()
print("waiting pull"); sys.stdout.flush()
depth_frame = depthstream.pull_frame()
print("done pull"); sys.stdout.flush()

point_cloud_data = []
print(type(depth_frame)); sys.stdout.flush()
if depth_frame is not None:
    print("inside"); sys.stdout.flush()
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.05), cv2.COLORMAP_JET)
    
    # Point Cloud Calculation
    height, width = depth_image.shape
    z = depth_image * 0.001  # Convert to meters
    valid_depth = z > 0
    
    x = ((np.arange(width) - cx) * z) / fx
    y = ((np.arange(height)[:, np.newaxis] - cy) * z) / fy
    
    point_cloud_array = np.dstack((x, y, z))
    point_cloud_array = point_cloud_array[valid_depth]
    
    # Capture timestamp for each frame and store points in the list
    current_time = time.time()
    frame_points = [{'Timestamp': round(current_time, 2), 'X': p[0], 'Y': p[1], 'Z': p[2]} for p in point_cloud_array]
    point_cloud_data.extend(frame_points)  # Use extend to add all points from the current frame

    cv2.imshow('Depth Stream', depth_colormap)

print("sleeping 5sec"); sys.stdout.flush()
time.sleep(5)
depthstream.stop()
depthstream.join()
cv2.destroyAllWindows()

pipeline.stop()

