import time
import RPi.GPIO as GPIO

# GPIO setup
BUTTON_PIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Enable internal pull-up resistor

print("Press and release the button to see status. Hold to measure duration. Press Ctrl+C to exit.")

try:
    while True:
        if GPIO.input(BUTTON_PIN) == GPIO.LOW:  # Button pressed
            press_start = time.time()
            print("Button Pressed!")

            while GPIO.input(BUTTON_PIN) == GPIO.LOW:  # Wait while button is held
                time.sleep(0.1)  # Prevent CPU overload

            press_duration = time.time() - press_start
            print(f"Button Released! Held for {press_duration:.2f} seconds.")

        time.sleep(0.1)  # Polling delay

except KeyboardInterrupt:
    print("\nExiting program.")

# Cleanup
GPIO.cleanup()
print("GPIO cleanup complete. Program terminated.")

