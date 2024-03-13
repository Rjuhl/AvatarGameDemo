import os
import numpy as np


def random_rotation_matrix():
    # Random rotation angle
    theta = np.random.uniform(-np.pi / 4, np.pi / 4)

    # Random rotation axis
    u = np.random.randn(3)
    u /= np.linalg.norm(u)

    # Compute the rotation matrix using the Rodriguez formula
    I = np.eye(3)
    u_x = np.array([[0, -u[2], u[1]],
                    [u[2], 0, -u[0]],
                    [-u[1], u[0], 0]])
    R = I + np.sin(theta) * u_x + (1 - np.cos(theta)) * np.dot(u_x, u_x)
    return R


def generate_multiple_rotations(num_rotations):
    rotation_tensor = np.zeros((num_rotations, 3, 3))
    for i in range(num_rotations): rotation_tensor[i, :, :] = random_rotation_matrix()
    return rotation_tensor

def get_augment_tensor(tensor_file, num_of_aug_per_data_sample):
    data = np.load(tensor_file)
    num_samples = data.shape[0]
    point_per_sample = data.shape[1] // 3
    data = data.reshape(-1, 3, point_per_sample).transpose(0, 2, 1)[:, np.newaxis, :, :]
    rotation_matrix = generate_multiple_rotations(num_of_aug_per_data_sample)
    aug_data = data @ rotation_matrix
    aug_data = np.concatenate(aug_data, axis=0).transpose(0, 2, 1).reshape(-1, point_per_sample * 3)
    return aug_data

def save_tensor(tensor, save_dir, name):
    np.save(os.path.join(save_dir, name), tensor)
    print(f"Tensor saved at {os.path.join(save_dir, name)} \nTensor shape = {tensor.shape}")


E_FILE = "/Users/rainjuhl/PycharmProjects/CS131AvatarProject/ArticulationData/e_tensor.npy"
X_FILE = "/Users/rainjuhl/PycharmProjects/CS131AvatarProject/ArticulationData/x_tensor.npy"
OTHER_FILE = "/Users/rainjuhl/PycharmProjects/CS131AvatarProject/ArticulationData/other_tensor.npy"
SAVE_DIR = "/Users/rainjuhl/PycharmProjects/CS131AvatarProject/ArticulationData"

save_tensor(get_augment_tensor(E_FILE, 12), SAVE_DIR, "aug_e_tensor")
save_tensor(get_augment_tensor(X_FILE, 12), SAVE_DIR, "aug_x_tensor")
save_tensor(get_augment_tensor(OTHER_FILE, 1), SAVE_DIR, "aug_other_tensor")
