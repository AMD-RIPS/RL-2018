import gym
import matplotlib.pyplot as plt
import cv2 as cv


def flip_state(img):
	img[0:82, :] = cv.flip(img[0:82,:], 1)
	return img

def vis_state_mirrored(img):
	plt.imshow(flip_state(img))
	plt.show()

def vis_state(img):
	plt.imshow(img)
	plt.show()


env = gym.make('CarRacing-v0')
state = env.reset()
done = False
index = 0
env.render()
while not done:
	state, _, done, _ = env.step(env.action_space.sample())
	if index == 400:
		vis_state(state)
		vis_state_mirrored(state)
	index += 1