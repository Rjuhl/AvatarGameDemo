import cv2
import mediapipe as mp
import torch
import numpy as np
from GameDependencies.HandClassificationModel import HandMLP
from Utils import Hand, PalmStandIn


def live_stream_hand(model_file,
                     image_lock,
                     image_size,
                     process_dict,
                     livestream_image,
                     hand_pose,
                     palm_coord):

    model = HandMLP()
    checkpoint = torch.load(model_file)
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
        process_dict["video_capture_loaded"] = True
        while cap.isOpened() and process_dict["live_stream_hand"]:
            success, image = cap.read()
            if not success: continue

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            output = 2
            palm = PalmStandIn()
            if results.multi_hand_landmarks:

                with torch.no_grad():
                    model_input = torch.tensor(hand.get_hand_vector(results)).to(torch.float32)
                    output = model(model_input)
                    output = np.argmin(output.numpy() * -1)

                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                palm = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.WRIST]

            image = cv2.resize(cv2.flip(image, 1), (image_size[1], image_size[0]), interpolation=cv2.INTER_AREA)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_lock.acquire()
            livestream_image *= 0
            livestream_image += image
            hand_pose *= 0
            hand_pose += output
            palm_coord *= 0
            palm_coord += np.array([palm.x, palm.y])
            image_lock.release()

    cap.release()
    print("thread closed")
