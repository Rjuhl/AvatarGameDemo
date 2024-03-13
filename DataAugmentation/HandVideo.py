import cv2
import mediapipe as mp
import torch
import numpy as np
from GameDependencies.HandClassificationModel import HandMLP
from GameDependencies.Utils import Hand

MODEL_FILE = "/Model/ClassificationModel1"

model = HandMLP()
checkpoint = torch.load(MODEL_FILE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

hand = Hand()

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:

            with torch.no_grad():
                model_input = torch.tensor(hand.get_hand_vector(results)).to(torch.float32)
                output = model(model_input)
                output = np.argmin(output.numpy() * -1)
                if output == 0: print("Fire")
                if output == 1: print("Earth")

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        # print(type(image), image.shape)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
