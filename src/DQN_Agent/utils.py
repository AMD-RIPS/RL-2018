import sys
sys.dont_write_bytecode = True

import numpy as np
from skimage.transform import resize
import time
from PIL import Image
import cv2 as cv
import matplotlib.pyplot as plt

def pause():
    programPause = raw_input("Press the <ENTER> key to continue...")

def flip_state(image):
    image[:82, :] = cv.flip(image[:82, :], 1)
    return image

def unit_image(image):
    image  = np.array(image)
    return np.true_divide(image,255)

def grayscale_img(image):
    return np.dot(image[...,:3], [0.299, 0.587, 0.114])

def process_image(rgb_image, flip, detect_edges=False):
    if flip:
        rgb_image = flip_state(rgb_image)
    if detect_edges:
        edgy = cv.Canny(rgb_image, 150, 250, apertureSize=3)
        result = edgy
    else:
        gray = grayscale_img(rgb_image)
        gray = unit_image(gray)
        result = gray
    show(result)
    return result

def in_grass(state):
    cropped = state[66:78, 43:53]
    green = np.sum(cropped[..., 1] >= 204)
    return green >= 45

def show(image):
    # Or with plt:
    plt.imshow(image, cmap='gray')
    plt.show()

def document_parameters(agent):
    # document parameters
    with open(agent.model_path + '/params.txt', 'w') as file:
        file.write('Environment: ' + str(agent.env) + '\n')
        file.write('Architecture: ' + str(agent.architecture) + '\n')
        file.write('Explore Rate: ' + str(agent.explore_rate) + '\n')
        file.write('Learning Rate: ' + str(agent.learning_rate) + '\n')
        file.write('Discount: ' + str(agent.discount) + '\n')
        file.write('Batch Size: ' + str(agent.replay_memory.batch_size) + '\n')
        file.write('Memory Capacity: ' + str(agent.replay_memory.memory_capacity) + '\n')
        file.write('Num Episodes: ' + str(agent.training_metadata.num_episodes) + '\n')
        file.write('Learning Rate Drop Frame Limit: ' + str(agent.training_metadata.frame_limit) + '\n')

class Training_Metadata:

    def __init__(self, frame=0, frame_limit=1000000, episode=0, num_episodes=10000):
        self.frame = frame
        self.frame_limit = frame_limit
        self.episode = episode
        self.num_episodes = num_episodes

    def increment_frame(self):
        self.frame += 1

    def increment_episode(self):
        self.episode += 1
