import numpy as np
from Utils import ElementData, EnergyBar, Smoother, ElementGameObject, BorderFlash
from GameDependencies.Constants import Constants
from Scorer import ScoreMap
import pygame
import time
import threading
from threading import Thread
from collections import deque
from GameDependencies.HandLiveStream import live_stream_hand

class Engine:
    def __init__(self, args=Constants()):

        args.validate()

        self.fps = args.fps
        self.screen_size = args.screen_size
        self.pose_limit = args.pose_limit
        self.model_file = args.model_file
        self.hand_screen_size = args.hand_screen_size
        self.hand_image = np.zeros(self.hand_screen_size)
        self.hand_image_lock = threading.Lock()
        self.last_hand_pose = np.array(2)
        self.current_hand_pose = np.array(2)
        self.palm_coordinates = np.zeros(2)
        self.process_dict = {"live_stream_hand": True, "video_capture_loaded": False}

        self.move_dict = args.MOVE_DICT

        self.position_data = ElementData(
            args.element_data_fps,
            null_id=args.null_id,
            min_positions=args.min_positions,
            return_limit=args.pose_limit,
            store_limit=args.store_limit
        )

        self.energy_bar = EnergyBar(
            args.energy_fps,
            energy_total=args.energy_total,
        )
        self.score_map = ScoreMap(
            sub_sample_size=args.sub_sample_size,
            tolerance=args.tolerance,
            score_map_size=args.score_map_size,
            graph_mode=args.graph_score_maps
        )

        self.smoother = Smoother(
            smooth_factor=args.smooth_factor,
            smoothing_id=args.smoothing_id
        )

        self.score_map.load(args.score_files, args.score_file_ids, generate=args.build_score_map_from_scratch)

        # Place optimally later
        self.energy_total = args.energy_total
        self.energy_font_size = args.energy_font_size
        self.fade_step = args.fade_step
        self.border_thickness = args.border_thickness
        self.player_file = args.player_file


    def start(self):

        # Init Pygame
        pygame.init()
        screen = pygame.display.set_mode(self.screen_size)
        screen.fill((255, 255, 255))
        pygame.display.set_caption('Avatar Game Demo')
        player = pygame.image.load(self.player_file)

        # Hand Surface
        hand_surface = pygame.Surface((self.hand_screen_size[1], self.hand_screen_size[0]))
        pygame.surfarray.blit_array(hand_surface, np.zeros(self.hand_screen_size).transpose((1, 0, 2)))

        # Energy Surfaces
        energy_font = pygame.font.Font(None, self.energy_font_size)
        energy_text = energy_font.render(f'{self.energy_total}', True, (0, 0, 0), (255, 255, 255))
        energy_text_rect = energy_text.get_rect()
        energy_text_rect.left = int(round(self.screen_size[0] * 0.05))  # Align to the left of the screen
        energy_text_rect.bottom = self.screen_size[1] - int(round(self.screen_size[1] * 0.05))

        dt = 0
        game_objects = deque([])
        clock = pygame.time.Clock()
        running = True

        # Init Video Capture Thread
        video_capture_thread = Thread(target=live_stream_hand, args=(self.model_file,
                                                                     self.hand_image_lock,
                                                                     self.hand_screen_size,
                                                                     self.process_dict,
                                                                     self.hand_image,
                                                                     self.current_hand_pose,
                                                                     self.palm_coordinates))

        video_capture_thread.start()
        while not self.process_dict["video_capture_loaded"]:
            time.sleep(1 / 100)

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.process_dict["live_stream_hand"] = False
                    video_capture_thread.join()
                    running = False

            # Update Energy Bar
            current_energy = self.energy_bar.update()
            energy_text = energy_font.render(f'{current_energy}', True, (0, 0, 0), (255, 255, 255))

            # Get New Pos, Gesture, and Hand Screen
            new_pos, gesture = None, None
            self.hand_image_lock.acquire()
            pygame.surfarray.blit_array(hand_surface, self.hand_image.transpose((1, 0, 2)))
            new_pos, gesture = self.palm_coordinates.copy(), self.current_hand_pose.copy()
            self.hand_image_lock.release()

            # Perform Move'
            gesture = self.smoother.get_gesture(gesture, dt)
            if gesture != self.last_hand_pose:
                list_of_positions = self.position_data.get_positions()
                if list_of_positions is not None:
                    move = self.score_map.score(list_of_positions)
                    if move is not False and self.energy_bar.perform_move(self.move_dict[int(self.last_hand_pose)][move].get_cost()):
                        # Add GameObject
                        object_info = self.move_dict[int(self.last_hand_pose)][move]
                        game_objects.append(
                            ElementGameObject(
                                screen,
                                object_info.get_file(),
                                object_info.get_position(),
                                object_info.get_speed(),
                                object_info.get_lifetime(),
                                object_info.get_axis()
                            )
                        )
                    else:
                        game_objects.append(BorderFlash(screen, self.fade_step, thickness=self.border_thickness))

                self.last_hand_pose = gesture

            # Update Element Data
            self.position_data.add_item(new_pos, gesture)

            # Update Screen
            screen.fill((255, 255, 255))

            for i in range(len(game_objects)):
                cur_obj = game_objects.popleft()
                cur_obj.update(dt)
                if cur_obj.alive(): game_objects.append(cur_obj)

            screen.blit(player, (self.screen_size[0] // 4, self.screen_size[1] // 4))
            screen.blit(hand_surface, (0, 0))
            screen.blit(energy_text, energy_text_rect)
            pygame.display.flip()

            dt = clock.tick(60) / 1000

        pygame.display.quit()
        pygame.quit()
        print("Session ended")


e = Engine()
e.start()


