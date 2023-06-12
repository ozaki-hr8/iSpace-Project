import cv2
import mediapipe as mp
import numpy as np
import csv

mp_drawing = mp.solutions.drawing_utils
mp_holistic =  mp.solutions.holistic

cap = cv2.VideoCapture(4)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        image =cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = holistic.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cv2.imshow('Mediapipe' ,image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            num_coords = len(results.pose_landmarks.landmark)

            landmarks = ['class']
            for val in range(1, num_coords+1):
                landmarks += ['skeleton_x{}'.format(val), 'skeleton_y{}'.format(val), 'skeleton_z{}'.format(val), 'skeleton_v{}'.format(val)]

            landmarks+=['human_x','human_y','human_z']

            for val in range(1, num_coords+1):
                landmarks += ['distance_s_x{}'.format(val), 'distance_s_y{}'.format(val), 'distance_s_z{}'.format(val), 'distance_s_v{}'.format(val)]

            with open('training_csv/action_3d.csv',mode='w' ,newline='') as f:
                csv_writer =csv.writer(f, delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(landmarks)
            break

cap.release()
cv2.destroyAllWindows()
