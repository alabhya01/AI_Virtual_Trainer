import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pandas as pd
import RPi.GPIO as GPIO
import time
from gpiozero import LED
from mediapipe.tasks import python
from mediapipe.tasks.python import vision



cap = cv2.VideoCapture(0)

red_led = LED(17)
green_led = LED(27)
on_led = LED(18)
GPIO.setmode(GPIO.BCM)
GPIO.setup(22, GPIO.IN, pull_up_down=GPIO.PUD_UP)#bicep
GPIO.setup(14, GPIO.IN, pull_up_down=GPIO.PUD_UP)#shoulder 
GPIO.setup(15, GPIO.IN, pull_up_down=GPIO.PUD_UP)# on / off pin


green_led.off()
red_led.off()
on_led.off()

# Helper function to calculate angle
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle


#mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
# STEP 2: Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)


def bicep_detect():
    print("Detecting Bicep Curl")
    model = tf.keras.models.load_model('bicep_lstm.h5')
    frame_num = 0
    data = []
    while cap.isOpened():
        print("Data Len = ", len(data))
        frame_num+=1
        print("Capturing Frame", frame_num)
        ret, frame = cap.read()
        if not ret:
            break

        # Convert color
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        results = detector.detect(mp_image)

        try:
            landmarks = results.pose_landmarks[0]
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)

            # Add to data
            data.append([
                shoulder[0], shoulder[1],
                elbow[0], elbow[1],
                wrist[0], wrist[1],
                angle
            ])
            if(len(data) == 90):
                np_data = np.array(data)
                frame_num = 0
                print(np_data.shape)
                input_data = np.reshape(np_data, (1, 90, 7))
                pred = model.predict(input_data)[0][0]
                print(pred)
                if(pred>0.75):
                    print("Damn baby good going")
                    red_led.off()
                    green_led.on()
                else:
                    print("Mofo Correct the pose")
                    red_led.on()
                    green_led.off()
                data = []

        except:
            pass

        if GPIO.input(22) == GPIO.HIGH or GPIO.input(15) == GPIO.HIGH:# Replace this with GPIO setting
            print("Exiting Bicep curl")
            return



def shoulder_detect():
    data = []
    frame_num = 0
    print("Detecting Shoulder Press")
    model = tf.keras.models.load_model('shoulder_lstm.h5')
    while cap.isOpened():
        print("Data Len = ", len(data))
        frame_num+=1
        print("Frame Num = ", frame_num)
        ret, frame = cap.read()
        if not ret:
            break

        # Convert color
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        results = detector.detect(mp_image)

        try:
            landmarks = results.pose_landmarks[0]
            shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow_l = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist_l = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist_r = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Calculate angle
            angle_l = calculate_angle(shoulder_l, elbow_l, wrist_l)
            angle_r = calculate_angle(shoulder_r, elbow_r, wrist_r)

            # Add to data
            data.append([
                shoulder_l[0], shoulder_l[1],
                shoulder_r[0], shoulder_r[1],
                elbow_l[0], elbow_l[1],
                elbow_r[0], elbow_r[1],
                wrist_l[0], wrist_l[1],
                wrist_r[0], wrist_r[1],
                angle_l,
                angle_r
            ])
            if(len(data) == 90):
                frame_num = 0
                np_data = np.array(data)
                print(np_data.shape)
                input_data = np.reshape(np_data, (1, 90, 14))
                pred = model.predict(input_data)[0][0]
                print(pred)
                if(pred>0.75):
                    print("Damn baby good going")
                    green_led.on()
                    red_led.off()
                else:
                    print("Mofo Correct the pose")
                    red_led.on()
                    green_led.off()
                data = []

        except:
            pass

        if GPIO.input(14) == GPIO.HIGH or GPIO.input(15) == GPIO.HIGH:# Replace this with GPIO settingi
            print("exiting shoulder press")
            return



while True:
    if GPIO.input(15) == GPIO.HIGH:
        on_led.off()
        print("Yet to start")
        continue
    else:
        on_led.on()
        if GPIO.input(14) == GPIO.LOW:
            shoulder_detect()
        elif GPIO.input(22) == GPIO.LOW:
            bicep_detect()
        else:
            time_count = 0
            print("code started, no exercise selected")
            red_led.off()
            green_led.off()
            continue




# Release
cap.release()
