import cv2
from picamera import PiCamera
from picamera.array import PiRGBArray
import time

# Initialize the camera
camera = PiCamera()
camera.resolution = (640, 480)
raw_capture = PiRGBArray(camera)

# Allow the camera to warm up
time.sleep(0.1)

# Continuously capture frames from the camera
for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
    # Access the frame data
    image = frame.array

    # Display the frame
    cv2.imshow("Frame", image)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Clear the stream in preparation for the next frame
    raw_capture.truncate(0)

# Release resources
cv2.destroyAllWindows()
