import sys
sys.dont_write_bytecode = True

import numpy as np
from skimage.transform import resize
import time
from PIL import Image


def pause():
    programPause = raw_input("Press the <ENTER> key to continue...")

def unit_image(image):
    image  = np.array(image)
    # max_along_dim = np.amax(image)
    # return np.true_divide(image,max_along_dim)
    return np.true_divide(image,255)

def grayscale_img(image):
    return np.dot(image[...,:3], [0.299, 0.587, 0.114])

def process_image(rgb_image, crop=(None, None, None, None), downscaling_dimension=(84, 84)):
    rgb_image = rgb_image[crop[0]:crop[1], crop[2]:crop[3], :]
    gray = grayscale_img(rgb_image)
    gray = resize(gray, downscaling_dimension)
    gray = unit_image(gray)
    return gray

def process_nature_atari(rgb_image, downscaling_dimension = (84, 84)):
    gray = grayscale_img(rgb_image)
    downscaled_img = resize(gray, downscaling_dimension)
    normalized_downscaled_img = unit_image(downscaled_img)
    return normalized_downscaled_img

def show(image):
    im = Image.fromarray(image)
    im.show()

class Timer:

    def __init__(self):
        self.clocks = {}

    def add_timer(self, name):
        self.clocks[name] = time.time()

    def get_timer(self, name):
        current_time = time.time()
        elapsed_time = current_time - self.clocks[name]
        self.clocks[name] = current_time
        return elapsed_time


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
