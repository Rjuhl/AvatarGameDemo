import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm
from GameDependencies.Constants import Constants

DATA_DIR = "/ASL_Alphabet_Dataset/asl_alphabet_train"
SAVE_DIR = "/ArticulationData"

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

constants = Constants()


# For static images:
def articulate_hands(files, cdir):
    hand_matrix = []
    with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5) as hands:

        for idx, f in tqdm(enumerate(files), total=len(files), desc=f'Processing {cdir}', leave=False):
            # Read an image, flip it around y-axis for correct handedness output (see
            # above).
            file = os.path.join(DATA_DIR, cdir, f)
            if not os.path.isfile(file): continue
            image = cv2.flip(cv2.imread(file), 1)
            # Convert the BGR image to RGB before processing.
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not results.multi_hand_landmarks: continue
            hand = results.multi_hand_landmarks[0]
            hand_vector = [hand.landmark[digit].x for digit in constants.PARTS_OF_HAND] + \
                          [hand.landmark[digit].y for digit in constants.PARTS_OF_HAND] + \
                          [hand.landmark[digit].z for digit in constants.PARTS_OF_HAND]

            hand_matrix.append(hand_vector)

    return np.array(hand_matrix)


def create_training_data():
    other_matrix, e_matrix, x_matrix = np.zeros((1, 21 * 3)), np.zeros((1, 21 * 3)), np.zeros((1, 21 * 3))
    for cdir in tqdm(os.listdir(DATA_DIR), desc='Directories'):
        if cdir[0] == ".": continue
        if cdir[0].lower() == "x":
            x_matrix = np.concatenate((x_matrix, articulate_hands(os.listdir(os.path.join(DATA_DIR, cdir)), cdir)),
                                      axis=0)
        elif cdir[0].lower() == "e":
            e_matrix = np.concatenate((e_matrix, articulate_hands(os.listdir(os.path.join(DATA_DIR, cdir)), cdir)),
                                      axis=0)
        else:
            other_matrix = np.concatenate(
                (other_matrix, articulate_hands(os.listdir(os.path.join(DATA_DIR, cdir)), cdir)), axis=0)

    print("OTHER MATRIX SHAPE: ", other_matrix.shape)
    print("E MATRIX SHAPE: ", e_matrix.shape)
    print("X MATRIX SHAPE: ", x_matrix.shape)

    np.save(os.path.join(SAVE_DIR, "other_tensor.npy"), other_matrix[1:, :])
    np.save(os.path.join(SAVE_DIR, "e_tensor.npy"), e_matrix[1:, :])
    np.save(os.path.join(SAVE_DIR, "x_tensor.npy"), x_matrix[1:, :])

    print(f"Data Saved to {SAVE_DIR}")

