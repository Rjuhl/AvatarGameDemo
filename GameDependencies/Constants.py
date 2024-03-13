import os
import mediapipe as mp
from collections import defaultdict

class ElementMove:
    def __init__(self, cost, speed, lifetime, axis, position, file, d="/Users/rainjuhl/PycharmProjects/CS131AvatarProject/GameAssets"):
        self.cost = cost
        self.speed = speed
        self.lifetime = lifetime
        self.axis = axis
        self.file = file
        self.position = position
        self.d = d

    def get_cost(self): return self.cost
    def get_speed(self): return self.speed
    def get_lifetime(self): return self.lifetime
    def get_axis(self): return self.axis
    def get_position(self): return self.position
    def get_file(self): return os.path.join(self.d, self.file)

class Constants:
    def __init__(self):
        self.mp_hands = mp.solutions.hands

        self.PARTS_OF_HAND = [
            self.mp_hands.HandLandmark.WRIST,
            self.mp_hands.HandLandmark.THUMB_CMC,
            self.mp_hands.HandLandmark.THUMB_MCP,
            self.mp_hands.HandLandmark.THUMB_IP,
            self.mp_hands.HandLandmark.THUMB_TIP,
            self.mp_hands.HandLandmark.INDEX_FINGER_MCP,
            self.mp_hands.HandLandmark.INDEX_FINGER_PIP,
            self.mp_hands.HandLandmark.INDEX_FINGER_DIP,
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_DIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_MCP,
            self.mp_hands.HandLandmark.RING_FINGER_PIP,
            self.mp_hands.HandLandmark.RING_FINGER_DIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_MCP,
            self.mp_hands.HandLandmark.PINKY_PIP,
            self.mp_hands.HandLandmark.PINKY_DIP,
            self.mp_hands.HandLandmark.PINKY_TIP,
        ]

        # General Args
        self.fps = 60
        self.model_file = "/Users/rainjuhl/PycharmProjects/CS131AvatarProject/Model/ClassificationModel2"

        # Screens
        self.hand_screen_size = (180, 320, 3)
        self.screen_size = (1280, 720)

        # Parameters for ElementData
        self.element_data_fps = 60
        self.null_id = 2
        self.min_positions = 20
        self.pose_limit = 1000
        self.store_limit = 10000


        # Parameters for EnergyBar
        self.energy_font_size = 48
        self.energy_fps = 5
        self.energy_total = 100
        self.MOVE_DICT = {
            0: {
                "block": ElementMove(30, 0, 10, (0, 0), (550, 100), "fire_block.png"),
                "attack": ElementMove(50, 300, 10, (1, 0), (300, 200), "fire_attack.png"),
                "ult": ElementMove(100, 1000, 10, (1, 0), (300, 200), "fire_ult.png")
            },
            1: {
                "block": ElementMove(10, 0, 10, (0, 0), (550, 200), "earth_block.png"),
                "attack": ElementMove(70, 250, 10, (1, 0), (350, 200), "earth_attack.png"),
                "ult": ElementMove(90, 600, 10, (1, 0), (300, 200), "earth_ult.png")
            },
            2: defaultdict(int)
        }

        # Parameters for ScoreMap
        self.tolerance = 700
        self.score_map_size = (180, 320)
        self.sub_sample_size = 600

        # Parameters for loading score_maps
        self.score_files = [
            "/Users/rainjuhl/PycharmProjects/CS131AvatarProject/ScoreMapFiles/Circle.jpg",
            "/Users/rainjuhl/PycharmProjects/CS131AvatarProject/ScoreMapFiles/Triangle.jpg",
            "/Users/rainjuhl/PycharmProjects/CS131AvatarProject/ScoreMapFiles/HardMove3.jpg"
        ]
        self.score_file_ids = ["block", "attack", "ult"]
        self.build_score_map_from_scratch = True

        # Smoother Args
        self.smooth_factor = 1
        self.smoothing_id = 2

        # Graphing args
        self.graph_score_maps = False

        # Game Object Args
        self.fade_step = 300
        self.border_thickness = 0.01
        self.player_file = "/Users/rainjuhl/PycharmProjects/CS131AvatarProject/GameAssets/player.png"


    def validate(self):
        assert self.hand_screen_size[0] <= self.screen_size[0] and self.hand_screen_size[1] <= self.screen_size[1]
