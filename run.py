
import pickle
import numpy as np
import pandas as pd

import mediapipe as mp # Import mediapipe
import cv2 # Import opencv


mp_drawing = mp.solutions.drawing_utils  # Drawing utilities
mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing_styles = mp.solutions.drawing_styles  # Drawing styles



with open('body_language.pkl', 'rb') as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)  # 0 for webcam

# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False # To improve performance by making image writeable to false as we are not going to edit the image 

        # Make detections
        results = holistic.process(image)

        # Recolor image back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        

        # Drawing face landmarks on the screen
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                image, results.face_landmarks, 
                mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None, 
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

        # Left Hand
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, 
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
            
#         used this line of code to change the colour of connections in the right hand
#         mp_drawing.DrawingSpec(colour=(0,0,255),thickness=2,circle_radius=2)

        # Right Hand
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, 
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))

        # Pose Detection
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, 
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            
        # Export coordinates
        try:

            #extracting pose landmarks
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

            #extracting face landmarks
            face = results.face_landmarks.landmark
            face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

            #concatenating both face and pose landmarks
            row = pose_row+face_row

            # #adding class name to the row
            # row.insert(0, class_name)

            # with open('coords.csv', mode='a', newline='') as f:
            #     csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            #     csv_writer.writerow(row)

            #make predictions
            X = pd.DataFrame([row])
            body_language_class = model.predict(X)[0]
            body_language_prob = model.predict_proba(X)[0]
            # print(body_language_class, body_language_prob)

            #grab ear coords
            coords = tuple(np.multiply(
                            np.array(
                                (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                        , [1920,1080]).astype(int))
            
            cv2.rectangle(image, (coords[0], coords[1]+10), (coords[0]+len(body_language_class)*30, coords[1]-30), (245, 117, 16), -1)

            cv2.putText(image, body_language_class, coords, 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            

            #get status box
            cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1) #top corner, bottom corner, colour, thickness

            #display class
            cv2.putText(image, 'CLASS'
                        , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, body_language_class.split(' ')[0]
                        , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            #display probability
            cv2.putText(image, 'PROB'
                        , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                        , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            

        except:
            pass

        cv2.imshow('Raw webcam feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
