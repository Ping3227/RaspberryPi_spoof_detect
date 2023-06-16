import RPi.GPIO as GPIO
import time as time_fun
final_result=0
GPIO.setmode(GPIO.BCM)
if(final_result == 1): ##false  red light 
    # 操作 GPIO 4（Pin 7）
    pin = 27
else : ## true green light 
    pin = 17
# 設定為 GPIO 為輸入模式
GPIO.setup(pin, GPIO.OUT)
# 設定 GPIO 輸出值為高電位
GPIO.output(pin, GPIO.HIGH)
print("light ")    
# 等待一秒鐘 lighting for 1 second 
time_fun.sleep(2)
print("light ")
# 設定 GPIO 輸出值為低電位
GPIO.output(pin, GPIO.LOW)
GPIO.cleanup()

