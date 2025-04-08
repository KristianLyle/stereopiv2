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

print ("You can press Q to quit this script!")
time.sleep (5)

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

# Use the whole image or a stripe for depth map?
useStripe = False
dm_colors_autotune = True
disp_max = -100000
disp_min = 10000

# Camera settimgs
cam_width = 1280
cam_height = 480

# Final image capture settings
scale_ratio = 0.5

# Camera resolution height must be dividable by 16, and width by 32
cam_width = int((cam_width+31)/32)*32
cam_height = int((cam_height+15)/16)*16
print ("Used camera resolution: "+str(cam_width)+" x "+str(cam_height))

# Buffer for captured image settings
img_width = int (cam_width * scale_ratio)
img_height = int (cam_height * scale_ratio)
capture = np.zeros((img_height, img_width, 4), dtype=np.uint8)
print ("Scaled image resolution: "+str(img_width)+" x "+str(img_height))

# Initialize the camera
camera = PiCamera(stereo_mode='side-by-side',stereo_decimate=False)
camera.resolution=(cam_width, cam_height)
camera.framerate = 20
camera.vflip = True
camera.hflip = True

# Initialize interface windows
cv2.namedWindow("Image")
cv2.moveWindow("Image", 50,100)
cv2.namedWindow("left")
cv2.moveWindow("left", 450,100)
cv2.namedWindow("right")
cv2.moveWindow("right", 850,100)


disparity = np.zeros((img_width, img_height), np.uint8)
sbm = cv2.StereoBM_create(numDisparities=0, blockSize=21)


def stereo_depth_map(rectified_pair):
    global disp_max, disp_min
    dmRight, dmLeft = rectified_pair

    disparity = sbm.compute(dmLeft, dmRight)
    local_max, local_min = disparity.max(), disparity.min()

    if dm_colors_autotune:
        disp_max, disp_min = max(local_max, disp_max), min(local_min, disp_min)
        local_max, local_min = disp_max, disp_min

    disparity_grayscale = (disparity - local_min) * (65535.0 / (local_max - local_min))
    disparity_fixtype = cv2.convertScaleAbs(disparity_grayscale, alpha=(255.0 / 65535.0))
    disparity_color = cv2.applyColorMap(disparity_fixtype, cv2.COLORMAP_JET)

    depth_map = calculate_depth(disparity)

    center_y, center_x = depth_map.shape[0] // 2, depth_map.shape[1] // 2
    center_depth_cm = depth_map[center_y, center_x]

    # Add visual indicator on center pixel
    cv2.circle(disparity_color, (center_x, center_y), radius=5, color=(0, 255, 0), thickness=-1)

    print(f"Estimated depth at center: {center_depth_cm:.2f} cm")

    cv2.imshow("Image", disparity_color)

    # Add indicators to left and right images
    dmLeft_indicator = cv2.cvtColor(dmLeft, cv2.COLOR_GRAY2BGR)
    dmRight_indicator = cv2.cvtColor(dmRight, cv2.COLOR_GRAY2BGR)

    cv2.circle(dmLeft_indicator, (center_x, center_y), radius=5, color=(0, 255, 0), thickness=-1)
    cv2.circle(dmRight_indicator, (center_x, center_y), radius=5, color=(0, 255, 0), thickness=-1)

    cv2.imshow("left", dmLeft_indicator)
    cv2.imshow("right", dmRight_indicator)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        quit()

    return disparity_color

def calculate_depth(disparity):
    # Assuming you have already loaded these values from the calibration data
    focal_length = 11.9055  # Example focal length in pixels (adjust this value)
    baseline = 0.06  # Example baseline in meters (adjust this value)
    
    # Convert disparity to depth using the formula
    with np.errstate(divide='ignore', invalid='ignore'):
        depth = (focal_length * baseline) / (disparity + 1e-5)  # Avoid division by zero

    # Convert depth to centimeters
    depth_cm = depth * 100  # Convert from meters to centimeters
    depth_cm = np.clip(depth_cm, 0, 500)  # Optional: clip values to reasonable range
    
    return depth_cm

def load_map_settings( fName ):
    global SWS, PFS, PFC, MDS, NOD, TTH, UR, SR, SPWS, loading_settings
    print('Loading parameters from file...')
    f=open(fName, 'r')
    data = json.load(f)
    SWS=data['SADWindowSize']
    PFS=data['preFilterSize']
    PFC=data['preFilterCap']
    MDS=data['minDisparity']
    NOD=data['numberOfDisparities']
    TTH=data['textureThreshold']
    UR=data['uniquenessRatio']
    SR=data['speckleRange']
    SPWS=data['speckleWindowSize']    
    #sbm.setSADWindowSize(SWS)
    sbm.setPreFilterType(1)
    sbm.setPreFilterSize(PFS)
    sbm.setPreFilterCap(PFC)
    sbm.setMinDisparity(MDS)
    sbm.setNumDisparities(NOD)
    sbm.setTextureThreshold(TTH)
    sbm.setUniquenessRatio(UR)
    sbm.setSpeckleRange(SR)
    sbm.setSpeckleWindowSize(SPWS)
    f.close()
    print ('Parameters loaded from file '+fName)


load_map_settings ("3dmap_set.txt")
try:
    npzfile = np.load('./calibration_data/{}p/stereo_camera_calibration.npz'.format(img_height))
except:
    print("Camera calibration data not found in cache, file ", './calibration_data/{}p/stereo_camera_calibration.npz'.format(img_height))
    exit(0)
    
imageSize = tuple(npzfile['imageSize'])
leftMapX = npzfile['leftMapX']
leftMapY = npzfile['leftMapY']
rightMapX = npzfile['rightMapX']
rightMapY = npzfile['rightMapY']
disparityToDepthMap = npzfile['dispartityToDepthMap']


# capture frames from the camera
for frame in camera.capture_continuous(capture, format="bgra", use_video_port=True, resize=(img_width,img_height)):
    t1 = datetime.now()
    pair_img = cv2.cvtColor (frame, cv2.COLOR_BGR2GRAY)
    imgLeft = pair_img [0:img_height,0:int(img_width/2)] #Y+H and X+W
    imgRight = pair_img [0:img_height,int(img_width/2):img_width] #Y+H and X+W
    imgL = cv2.remap(imgLeft, leftMapX, leftMapY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    imgR = cv2.remap(imgRight, rightMapX, rightMapY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    if (useStripe):
        imgRcut = imgR [80:160,0:int(img_width/2)]
        imgLcut = imgL [80:160,0:int(img_width/2)]
    else:
        imgRcut = imgR
        imgLcut = imgL
        
    rectified_pair = (imgLcut, imgRcut)
    disparity = stereo_depth_map(rectified_pair)
    # show the frame
    cv2.imshow("left", imgLcut)
    cv2.imshow("right", imgRcut)    

    t2 = datetime.now()
    #print ("DM build time: " + str(t2-t1))
    print("\n\n\nLINTEEEEEEEEEEEEEE")
    print(calculate_depth(disparity))
    print("CM\n\n\n\n")




