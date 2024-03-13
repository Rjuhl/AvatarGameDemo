import time
import pygame
import numpy as np
from collections import deque
from GameDependencies.Constants import Constants

constants = Constants()


class Smoother:
    def __init__(self, smoothing_id=2, smooth_factor=5):
        self.smoothing_id = smoothing_id
        self.smooth_factor = smooth_factor

        self.last_input = smoothing_id
        self.cur_id_count = 0

    def get_gesture(self, gesture, dt):
        if gesture == self.smoothing_id and self.cur_id_count < self.smooth_factor:
            self.cur_id_count += dt
            return self.last_input

        self.cur_id_count = 0
        self.last_input = gesture
        return gesture


class PalmStandIn:
    def __init__(self):
        self.x = 0
        self.y = 0


class Hand:
    def __init__(self):
        self.parts = constants.PARTS_OF_HAND

    def __len__(self):
        return len(self.parts)

    def __getitem__(self, item):
        return self.parts[item]

    def get_hand_vector(self, results):
        landmark = results.multi_hand_landmarks[0].landmark
        return [landmark[digit].x for digit in self.parts] + \
               [landmark[digit].y for digit in self.parts] + \
               [landmark[digit].z for digit in self.parts]


class ElementData:
    def __init__(self, fps, null_id=2, min_positions=20, return_limit=30, store_limit=1000):
        self.fps = fps
        self.null_id = null_id
        self.min_positions = min_positions
        self.return_limit = return_limit
        self.store_limit = store_limit

        self.element_id = None
        self.positions = deque([])

        self.clock_tick = time.time()

    def add_item(self, position, element):
        tick = time.time()
        if tick - self.clock_tick < 1 / self.fps: return

        self.clock_tick = tick
        if element != self.element_id:
            self.element_id = element
            self.positions = deque([position])
        else:
            self.positions.append(position)
            while len(self.positions) > self.store_limit:
                self.positions.popleft()

    def get_positions(self):
        return_positions = np.array(list(self.positions))
        if len(return_positions) < self.min_positions or self.element_id == self.null_id: return None
        indices = np.round(np.linspace(0, return_positions.shape[0] - 1, num=self.return_limit)).astype(int)
        return return_positions[indices]

    def reset(self):
        self.element_id = None
        self.positions = deque([])
        self.clock_tick = time.time()


class EnergyBar:
    def __init__(self,
                 fps,
                 energy_total=100,
                 ):
        self.energy = energy_total
        self.energy_total = energy_total
        self.fps = fps
        self.clock_tick = time.time()

    def update(self):
        tick = time.time()
        if tick - self.clock_tick >= 1 / self.fps and self.energy < self.energy_total:
            self.energy += 1
            self.clock_tick = tick
        return self.energy

    def perform_move(self, cost):
        if cost <= self.energy:
            self.energy -= cost
            return True
        return False


class ElementGameObject:
    def __init__(self, canvas, image_dir, pos, speed, lifetime, axis):
        self.canvas = canvas
        self.speed = speed
        self.image = pygame.image.load(image_dir)
        self.pos = self.image.get_rect().move(pos[0], pos[1])
        self.age = 0
        self.lifetime = lifetime
        self.axis = axis

    def update(self, dt):
        if self.alive():
            self.pos = self.pos.move(self.speed * dt * self.axis[0], self.speed * dt * self.axis[1])
            self.canvas.blit(self.image, self.pos)
            self.age += dt

    def alive(self):
        return self.age < self.lifetime


class BorderFlash:
    def __init__(self, canvas, fade_step, thickness=0.01):
        self.color = [255, 0, 0]
        self.fade_step = fade_step
        self.canvas = canvas
        self.border = round(max(self.canvas.get_size()) * thickness)

    def update(self, dt):
        step = max(round(dt * self.fade_step), 1)
        if self.color[1] < 255:
            self.color[1] = min(self.color[1] + step, 255)
        if self.color[2] < 255:
            self.color[2] = min(self.color[2] + step, 255)

        width, height = self.canvas.get_size()
        pygame.draw.rect(self.canvas,
                         self.color,
                         (0, 0, width, height),
                         self.border)

    def alive(self):
        return not all([color >= 255 for color in self.color])
