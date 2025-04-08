
import numpy as np

data = np.load('stereo_camera_calibration.npz')

print("keys: ", data.files)

imageSize = data['imageSize']
disparity = data['dispartityToDepthMap']


for item in data:
	print(data[item])
#print(imageSize)

depth_map = data.disparityToDepthMap[disparity_pixel_value]
print(disparity)
