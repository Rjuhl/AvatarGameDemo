import cv2
import math
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


class ScoreMap:
    def __init__(self, sub_sample_size=300, tolerance=400, score_map_size=(320, 180), graph_mode=False):
        self.sub_sample_size = sub_sample_size
        self.tolerance = tolerance
        self.size = score_map_size
        self.graph_mode = graph_mode
        self.score_maps_meta_info = []
        self.score_maps = np.zeros((1, score_map_size[0], score_map_size[1]))

    def load(self, files, ids, generate=True):
        if generate:
            self.generate_score_maps(files, ids)
        else:
            for i, file in enumerate(files):
                content = np.load(file)
                self.score_maps = np.concatenate((self.score_maps, content[0]))
                self.score_maps_meta_info.append((ids[i], content[1]))
            self.score_maps = self.score_maps[1:, :, :]

    def transform_image_wrapper(self, old_image, i, com):
        image = old_image.copy()
        return self.random_image_sample(
           image, # self.translate_image_to_score_map(image, self.score_maps_meta_info[i][1], com),
           self.sub_sample_size
        )

    def score(self, positions):
        image = np.zeros(self.size)
        positions = self.convert_positions_to_indices(positions, self.size)
        image[positions[:, 1], positions[:, 0]] = 255
        image = cv2.flip(image, 1)
        image = self.crop_image(image.T)
        image = self.resize_with_aspect_ratio(image, self.size)
        image[image != 0] = 1
        com = self.findCom(image)
        transform_image = np.array([self.transform_image_wrapper(image, i, com) for i in range(self.score_maps.shape[0])])
        scores = np.sum(self.score_maps * transform_image, axis=(1, 2))

        if self.graph_mode:
            for map_idx in range(self.score_maps.shape[0]):
                plt.subplot(1, self.score_maps.shape[0], map_idx + 1)
                plt.imshow(self.score_maps[map_idx], cmap="gray")
            plt.show()

            for t_idx in range(len(transform_image)):
                plt.subplot(1, len(transform_image), t_idx + 1)
                plt.imshow(transform_image[t_idx], cmap="gray")
            plt.show()

        print(scores)
        if scores.max() >= self.tolerance: return self.score_maps_meta_info[np.argmax(scores)][0]
        return False

    def generate_score_maps(self, files, ids):
        for i, file in enumerate(files):
            desired_sum = self.size[0] * self.size[1]
            base_image = 255 - np.array(Image.open(file).convert('L'))
            base_image = self.resize_with_aspect_ratio(base_image, self.size)
            base_image = cv2.GaussianBlur(base_image, (95, 95), 0, borderType=cv2.BORDER_CONSTANT)

            normalized_image = base_image / 255.0
            current_sum = np.sum(normalized_image)
            scale_factor = desired_sum / current_sum
            scaled_image = normalized_image * scale_factor
            x, y = self.findCom(scaled_image)

            miss_penalty = -desired_sum / np.sum(scaled_image == 0)
            score_map = scaled_image
            score_map[score_map == 0] = miss_penalty

            self.score_maps = np.concatenate((self.score_maps, score_map[np.newaxis, :, :]))
            self.score_maps_meta_info.append((ids[i], (x, y)))

        self.score_maps = self.score_maps[1:, :, :]

    def translate_image_to_score_map(self, resized_image, score_map_com, image_com):
        shift_x, shift_y = score_map_com[0] - image_com[0], score_map_com[1] - image_com[1]
        translated_image = np.zeros(self.size)

        base_start_x = max(0, shift_x)
        base_end_x = min(resized_image.shape[1], shift_x + translated_image.shape[1])
        base_start_y = max(0, shift_y)
        base_end_y = min(resized_image.shape[0], shift_y + translated_image.shape[0])

        overlay_start_x = max(0, -shift_x)
        overlay_end_x = overlay_start_x + (base_end_x - base_start_x)
        overlay_start_y = max(0, -shift_y)
        overlay_end_y = overlay_start_y + (base_end_y - base_start_y)

        translated_image[base_start_y:base_end_y, base_start_x:base_end_x] = \
            resized_image[overlay_start_y:overlay_end_y, overlay_start_x:overlay_end_x]

        return translated_image

    @staticmethod
    def resize_with_aspect_ratio(image, size):
        new_height, new_width = size
        h, w = image.shape[:2]

        f1, f2 = new_height / h, new_width / w
        f = f1 if f1 < f2 else f2
        new_h, new_w = math.floor(h * f), math.floor(w * f)

        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        new_image = np.zeros((new_height, new_width), dtype=np.uint8)

        x_offset = (new_width - new_w) // 2
        y_offset = (new_height - new_h) // 2

        new_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image

        return new_image

    @staticmethod
    def findCom(image):
        y_indices, x_indices = np.indices(image.shape)
        total_intensity = image.sum()
        x_weighted_sum = (x_indices * image).sum()
        y_weighted_sum = (y_indices * image).sum()
        com_x = x_weighted_sum / total_intensity
        com_y = y_weighted_sum / total_intensity
        return int(round(com_x)), int(round(com_y))

    @staticmethod
    def convert_positions_to_indices(positions, image_size):
        image_height, image_width = image_size
        positions[:, 0] = np.floor(positions[:, 0] * image_width)
        positions[:, 1] = np.floor(positions[:, 1] * image_height)
        positions[positions[:, 0] > image_width - 1, 0] = image_width - 1
        positions[positions[:, 1] > image_height - 1, 1] = image_height - 1
        return positions.astype(int)

    @staticmethod
    def crop_image(image):
        true_indices = np.where(image)
        min_row_index = np.min(true_indices[0])
        max_row_index = np.max(true_indices[0])
        min_col_index = np.min(true_indices[1])
        max_col_index = np.max(true_indices[1])
        return image[min_row_index:max_row_index + 1, min_col_index:max_col_index + 1].T

    @staticmethod
    def random_image_sample(image, num_to_keep):
        boolean_image = image != 0
        non_zero_points = np.sum(boolean_image)
        if num_to_keep >= non_zero_points: return image

        true_indices = np.where(boolean_image)[0]
        true_indices = np.random.choice(true_indices, non_zero_points - num_to_keep, replace=False)
        image[true_indices] = 0
        return image
