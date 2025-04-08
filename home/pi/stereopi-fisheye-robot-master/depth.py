from picamera import PiCamera
import time
import cv2
import numpy as np
import json
from datetime import datetime

print("You can press Q to quit this script!")
time.sleep(3)

# Stereo camera settings
cam_width = 1280
cam_height = 480
scale_ratio = 0.5

# Adjust camera resolution
cam_width = int((cam_width + 31) / 32) * 32
cam_height = int((cam_height + 15) / 16) * 16
print(f"Used camera resolution: {cam_width} x {cam_height}")

# Image buffer settings
img_width = int(cam_width * scale_ratio)
img_height = int(cam_height * scale_ratio)
capture = np.zeros((img_height, img_width, 4), dtype=np.uint8)
print(f"Scaled image resolution: {img_width} x {img_height}")

# Initialize the stereo camera
camera = PiCamera(stereo_mode='side-by-side', stereo_decimate=False)
camera.resolution = (cam_width, cam_height)
camera.framerate = 20

# OpenCV windows
cv2.namedWindow("Disparity Map")
cv2.namedWindow("Left Image")
cv2.namedWindow("Right Image")

# Stereo block matching algorithm
sbm = cv2.StereoBM_create(numDisparities=64, blockSize=21)

# Load stereo camera calibration data
try:
    npzfile = np.load('./calibration_data/{}p/stereo_camera_calibration.npz'.format(img_height))
    imageSize = tuple(npzfile['imageSize'])
    leftMapX = npzfile['leftMapX']
    leftMapY = npzfile['leftMapY']
    rightMapX = npzfile['rightMapX']
    rightMapY = npzfile['rightMapY']
    focal_length = 2714
    baseline = 6.0  # Distance between left & right cameras in meters
except:
    print(f"Camera calibration data not found in cache for resolution {img_height}p.")
    exit(0)
2
def stereo_depth_map(left_img, right_img):
    """ Computes the disparity map and depth estimation """
    disparity = sbm.compute(left_img, right_img).astype(np.float32) / 16.0  # Convert to float
    disparity[disparity <= 0] = 0.1  # Prevent division by zero

    # Compute depth (Z = (f * B) / disparity)
    depth_map = (focal_length * baseline) / disparity

    # Normalize for visualization
    disparity_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    disparity_color = cv2.applyColorMap(disparity_vis, cv2.COLORMAP_JET)

    return disparity_color, depth_map

# Start capturing frames
for frame in camera.capture_continuous(capture, format="bgra", use_video_port=True, resize=(img_width, img_height)):
    t1 = datetime.now()

    # Convert to grayscale
    pair_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    imgLeft = pair_img[:, :img_width // 2]  # Left half
    imgRight = pair_img[:, img_width // 2:]  # Right half

    # Rectify images
    imgL = cv2.remap(imgLeft, leftMapX, leftMapY, cv2.INTER_LINEAR)
    imgR = cv2.remap(imgRight, rightMapX, rightMapY, cv2.INTER_LINEAR)

    # Compute disparity & depth maps
    disparity_color, depth_map = stereo_depth_map(imgL, imgR)

    # Get depth of the center pixel
    center_x = imgL.shape[1] // 2
    center_y = imgL.shape[0] // 2
    depth_cm = depth_map[center_y, center_x] * 100  # Convert meters to cm

    print(f"Depth at center ({center_x},{center_y}): {depth_cm:.2f} cm")

    # Display images
    cv2.imshow("Left Image", imgL)
    cv2.imshow("Right Image", imgR)
    cv2.imshow("Disparity Map", disparity_color)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    t2 = datetime.now()
    print(f"Frame processing time: {t2 - t1}")

cv2.destroyAllWindows()
