import os
import time
import cv2
import numpy as np
import RPi.GPIO as GPIO
from datetime import datetime
from picamera import PiCamera
from picamera.array import PiRGBArray

# GPIO setup with pull-up resistor
BUTTON_PIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Enable internal pull-up resistor

# Create directory for captured images
capture_dir = "./captures"
os.makedirs(capture_dir, exist_ok=True)

# Camera settings
cam_width = 1280
cam_height = 480
frame_rate = 20

# Adjust resolution to match PiCamera constraints
cam_width = int((cam_width + 31) / 32)
cam_height = int((cam_height + 15) / 16)

print(f"Camera resolution: {cam_width} x {cam_height}")

# Initialize stereo camera
camera = PiCamera(stereo_mode='side-by-side', stereo_decimate=False)
camera.resolution = (cam_width, cam_height)
camera.framerate = frame_rate
camera.vflip = True  # Swap left and right cameras
camera.hflip = True

# Create a buffer for OpenCV display
raw_capture = PiRGBArray(camera, size=(cam_width, cam_height))
time.sleep(2)  # Allow camera to warm up

print("Press the button to capture an image. Hold for 3 seconds to exit.")

try:
    while True:
        # Capture frame into buffer
        camera.capture(raw_capture, format="bgr", use_video_port=True)
        frame = raw_capture.array

        # Display the frame
        cv2.imshow("Stereo Preview", frame)

        # Check if button is pressed
        if GPIO.input(BUTTON_PIN) == GPIO.LOW:  # Button pressed
            press_start_time = time.time()
            print("Button Pressed!")

            while GPIO.input(BUTTON_PIN) == GPIO.LOW:  # Wait while still held
                time.sleep(0.1)  # Prevent excessive CPU usage
                if time.time() - press_start_time >= 3:  # If held for 3 seconds
                    print("\nButton held for 3 seconds. Exiting program...")
                    raise KeyboardInterrupt  # Exit the loop

            # Capture image on short press
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"{capture_dir}/stereo_{cam_width}x{cam_height}_{timestamp}.png"

            # Save the captured frame
            cv2.imwrite(filename, frame)
            print(f"Captured: {filename}")

            time.sleep(0.5)  # Simple debounce

        # Clear buffer for next frame
        raw_capture.truncate(0)

        # Exit if 'Q' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

except KeyboardInterrupt:
    print("\nExiting program.")

# Cleanup
cv2.destroyAllWindows()
camera.close()
GPIO.cleanup()
print("Program terminated.")

