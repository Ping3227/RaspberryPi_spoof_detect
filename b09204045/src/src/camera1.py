from picamera2 import Picamera2, Preview
import time
picam2 = Picamera2()
camera_config = picam2.create_still_configuration(main={"size": (3280,2464)}, lores={"size": (640, 480)}, display="lores")
picam2.configure(camera_config)
picam2.start_preview(Preview.QT)
picam2.start()
time.sleep(1)
picam2.capture_file("test1.jpg")