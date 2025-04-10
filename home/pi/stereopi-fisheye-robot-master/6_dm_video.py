# Copyright (C) 2019 Eugene a.k.a. Realizator, stereopi.com, virt2real team
#
# This file is part of StereoPi tutorial scripts.
#
# StereoPi tutorial is free software: you can redistribute it 
# and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, either version 3 of the 
# License, or (at your option) any later version.
#
# StereoPi tutorial is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with StereoPi tutorial.  
# If not, see <http://www.gnu.org/licenses/>.
#
#          <><><> SPECIAL THANKS: <><><>
#
# Thanks to Adrian and http://pyimagesearch.com, as a lot of
# code in this tutorial was taken from his lessons.
#  
# Thanks to RPi-tankbot project: https://github.com/Kheiden/RPi-tankbot
#
# Thanks to rakali project: https://github.com/sthysel/rakali


from picamera import PiCamera
import time
import cv2
import numpy as np
import json
from datetime import datetime

print("You can press Q to quit this script!")
time.sleep(2)

# Depth map default preset
SWS = 5
PFS = 5
PFC = 29
MDS = -30
NOD = 160
TTH = 100
UR = 10
SR = 14
SPWS = 100

useStripe = False
dm_colors_autotune = True
disp_max = -100000
disp_min = 10000

# Camera settings
cam_width = 1280
cam_height = 480
scale_ratio = 0.5

cam_width = int((cam_width+31)/32)*32
cam_height = int((cam_height+15)/16)*16
print("Used camera resolution: "+str(cam_width)+" x "+str(cam_height))

img_width = int(cam_width * scale_ratio)
img_height = int(cam_height * scale_ratio)
capture = np.zeros((img_height, img_width, 4), dtype=np.uint8)
print("Scaled image resolution: "+str(img_width)+" x "+str(img_height))

camera = PiCamera(stereo_mode='side-by-side', stereo_decimate=False)
camera.resolution = (cam_width, cam_height)
camera.framerate = 20
camera.vflip = True
camera.hflip = True

cv2.namedWindow("Disparity")
cv2.moveWindow("Disparity", 50,100)
cv2.namedWindow("Left")
cv2.moveWindow("Left", 450,100)
cv2.namedWindow("Right")
cv2.moveWindow("Right", 850,100)

# StereoBM setup
sbm = cv2.StereoBM_create(numDisparities=160, blockSize=21)

def calculate_depth(disparity):
    focal_length = 11.9055  # adjust as needed
    baseline = 0.06         # adjust as needed (meters)

    with np.errstate(divide='ignore'):
        depth = (focal_length * baseline) / (disparity.astype(np.float32) + 1e-5)

    depth_cm = depth * 100
    depth_cm = np.clip(depth_cm, 0, 500)
    return depth_cm

def stereo_depth_map(rectified_pair):
    global disp_max, disp_min
    dmLeft, dmRight = rectified_pair

    disparity_raw = sbm.compute(dmLeft, dmRight).astype(np.float32)
    disparity_raw /= 16.0  # normalize disparity

    local_max, local_min = disparity_raw.max(), disparity_raw.min()

    if dm_colors_autotune:
        disp_max = max(local_max, disp_max)
        disp_min = min(local_min, disp_min)
        local_max, local_min = disp_max, disp_min

    # Normalize for visualization
    disparity_vis = (disparity_raw - local_min) * (255.0 / (local_max - local_min))
    disparity_vis = np.uint8(np.clip(disparity_vis, 0, 255))
    disparity_color = cv2.applyColorMap(disparity_vis, cv2.COLORMAP_JET)

    # Calculate depth
    depth_map = calculate_depth(disparity_raw)

    center_y, center_x = depth_map.shape[0] // 2, depth_map.shape[1] // 2
    center_depth_cm = depth_map[center_y, center_x]

    # Visual marker
    cv2.circle(disparity_color, (center_x, center_y), 5, (0, 255, 0), -1)
    cv2.imshow("Disparity", disparity_color)

    # Debug values
    print(f"Disparity at center: {disparity_raw[center_y, center_x]:.2f}")
    print(f"Estimated depth at center: {center_depth_cm:.2f} cm")

    # Display left/right images
    dmLeft_bgr = cv2.cvtColor(dmLeft, cv2.COLOR_GRAY2BGR)
    dmRight_bgr = cv2.cvtColor(dmRight, cv2.COLOR_GRAY2BGR)

    cv2.circle(dmLeft_bgr, (center_x, center_y), 5, (0, 255, 0), -1)
    cv2.circle(dmRight_bgr, (center_x, center_y), 5, (0, 255, 0), -1)

    cv2.imshow("Left", dmLeft_bgr)
    cv2.imshow("Right", dmRight_bgr)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        quit()

    return disparity_raw

def load_map_settings(fName):
    global SWS, PFS, PFC, MDS, NOD, TTH, UR, SR, SPWS
    print('Loading parameters from file...')
    with open(fName, 'r') as f:
        data = json.load(f)
        SWS = data['SADWindowSize']
        PFS = data['preFilterSize']
        PFC = data['preFilterCap']
        MDS = data['minDisparity']
        NOD = data['numberOfDisparities']
        TTH = data['textureThreshold']
        UR = data['uniquenessRatio']
        SR = data['speckleRange']
        SPWS = data['speckleWindowSize']

    sbm.setPreFilterType(1)
    sbm.setPreFilterSize(PFS)
    sbm.setPreFilterCap(PFC)
    sbm.setMinDisparity(MDS)
    sbm.setNumDisparities(NOD)
    sbm.setTextureThreshold(TTH)
    sbm.setUniquenessRatio(UR)
    sbm.setSpeckleRange(SR)
    sbm.setSpeckleWindowSize(SPWS)
    print('Parameters loaded.')

# Load settings and calibration
load_map_settings("3dmap_set.txt")

try:
    npzfile = np.load(f'./calibration_data/{img_height}p/stereo_camera_calibration.npz')
except:
    print(f"Camera calibration data not found: ./calibration_data/{img_height}p/stereo_camera_calibration.npz")
    exit(0)

leftMapX = npzfile['leftMapX']
leftMapY = npzfile['leftMapY']
rightMapX = npzfile['rightMapX']
rightMapY = npzfile['rightMapY']

# Main loop
for frame in camera.capture_continuous(capture, format="bgra", use_video_port=True, resize=(img_width,img_height)):
    pair_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    imgLeft = pair_img[:, :img_width//2]
    imgRight = pair_img[:, img_width//2:]

    imgL = cv2.remap(imgLeft, leftMapX, leftMapY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    imgR = cv2.remap(imgRight, rightMapX, rightMapY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    if useStripe:
        imgL = imgL[80:160, :]
        imgR = imgR[80:160, :]

    stereo_depth_map((imgL, imgR))





