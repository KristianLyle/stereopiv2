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


import cv2
import os

# Global variables preset
total_photos = 20

# Photos to be cutted resolution
photo_width = 1280
photo_height = 480

# Left and right images resolution
img_width = 640
img_height = 480

# Visualization options
ShowImages = False

# Counter setup
photo_counter = 0

# Main pair cut cycle
if (os.path.isdir("./pairs")==False):
    os.makedirs("./pairs")
while photo_counter != total_photos:
    photo_counter +=1
    filename = './scenes/scene_'+str(photo_width)+'x'+str(photo_height)+\
               '_'+str(photo_counter) + '.png'
    if os.path.isfile(filename) == False:
        print ("No file named "+filename)
        continue
    pair_img = cv2.imread(filename,-1)
    
    if (ShowImages):
        cv2.imshow("ImagePair", pair_img)
        cv2.waitKey(0)
    imgLeft = pair_img [0:img_height,0:img_width] #Y+H and X+W
    imgRight = pair_img [0:img_height,img_width:photo_width]
    leftName = './pairs/left_'+str(photo_counter).zfill(2)+'.png'
    rightName = './pairs/right_'+str(photo_counter).zfill(2)+'.png'
    cv2.imwrite(leftName, imgLeft)
    cv2.imwrite(rightName, imgRight)
    print ('Pair No '+str(photo_counter)+' saved.')
    
print ('End cycle')

