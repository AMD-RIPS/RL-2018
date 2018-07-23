import sys
sys.dont_write_bytecode = True

import numpy as np
from skimage.transform import downscale_local_mean
import time

def pause():
    programPause = raw_input("Press the <ENTER> key to continue...")
    
def normalise_image(image):
    image  = np.array(image)
    return (image - np.mean(image))/np.std(image)

def process_image(rgb_image, crop = (None, None, None, None), downscaling_factor = (1, 1)):
    rgb_image = rgb_image[crop[0]:crop[1], crop[2]:crop[3], :]
    r, g, b = rgb_image[:,:,0], rgb_image[:,:,1], rgb_image[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray = downscale_local_mean(gray, downscaling_factor)
    gray = normalise_image(gray)
    return gray

def get_image_shape(env, crop, downscaling_factor):
	screen = env.reset()
	image = process_image(screen, crop, downscaling_factor)
	return np.shape(image)

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