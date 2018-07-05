import numpy as np
import gym
import math
import random as rand

class Cartpole():

	def __init__(self, nBUCKETS, BOUNDS, MIN_LEARNING_RATE, MIN_EXPLORE_RATE, DISCOUNT):
		self.env = gym.make('CartPole-v0')
		self.nBUCKETS = nBUCKETS
		self.BOUNDS = BOUNDS
		self.MIN_LEARNING_RATE = MIN_LEARNING_RATE
		self.MIN_EXPLORE_RATE = MIN_EXPLORE_RATE
		self.DISCOUNT = DISCOUNT
		self.sSize = self.getStatespaceSize()
		self.aSize = 2
		randomVector = np.random.normal(size = self.aSize * self.sSize)
		self.Q = [[randomVector[self.aSize * j + i] for i in range(self.aSize)] for j in range(self.sSize)]

	def getStatespaceSize(self):
		m = max(self.nBUCKETS)
		return (m - 1) * (1 + m + m**2 + m**3) + 1

	def learn(self, nEPISODES = 1000, MAXTIME = 250):
		trainingScores = []
		for episode in range(nEPISODES):
			self.runEpisode(MAXTIME, episode)

			# Keeping track of training performance
			if episode%20 == 0:
				testScore = self.test(visualise = False)
				trainingScores.append(testScore)
				if testScore > 190: break
		return episode

	def runEpisode(self, MAXTIME, episode):
		observation = self.env.reset()
		state = self.preprocess(observation)
		for time in range(MAXTIME):
			action = eGreedy(self.Q[state], time, self.MIN_EXPLORE_RATE)
			newObservation, reward, done, info = self.env.step(action)
			newState = self.preprocess(newObservation)

			# Decrease learning rate over time
			learningRate = max(self.MIN_LEARNING_RATE, min(1, 2.544 - math.log10(episode*MAXTIME + time + 1)))

			# Performing Q-learning update
			self.Q[state][action] += learningRate*(reward + self.DISCOUNT*np.max(self.Q[newState]) - self.Q[state][action])
			state = newState
			if done: break

	def test(self, testEpisodes = 10, testMaxTime = 250, visualise = True):
		testScores = []
		for episode in range(testEpisodes):
			observation = self.env.reset()
			state = self.preprocess(observation)

			# Stepping through episode
			for time in range(testMaxTime):
				if visualise: self.env.render()
				action = greedy(self.Q[state])
				newObservation, reward, done, info = self.env.step(action)
				newState = self.preprocess(newObservation)
				state = newState
				if done: break
			testScores.append(time)
		return sum(testScores)/float(testEpisodes)


	def preprocess(self, observation):
		buckets = []
		n = len(observation)

		for index in range(n):
			buckets.append(getBucket(observation[index], self.BOUNDS[index][0], self.BOUNDS[index][1], self.nBUCKETS[index]))

		# map to a non-negative integer
		m = max(self.nBUCKETS)
		answer = int(buckets[0] + m*buckets[1] + m**2 *buckets[2] + m**3 *buckets[3])
		return answer

def eGreedy(q, time, minExploreRate):
	aSize = len(q)
	u = rand.random()
	greedy = np.argmax(q)

	# Decrease exploration over time
	epsilon = max(minExploreRate, min(1, 2.544 - math.log10(time + 1)))
	return greedy if u > epsilon else int(rand.random()*aSize)

def greedy(q):
	return np.argmax(q)

def getBucket(value, lowerB, upperB, nBuckets):
	# Check whether out of bounds
	if value < lowerB:
			bucket = 0
	elif value > upperB:
		bucket = nBuckets - 1

	# assign buckets
	else:
		ran = upperB - lowerB
		bucket = math.floor(nBuckets*(value - lowerB) / ran)
	return bucket

def pause(self):
	programPause = raw_input("Press the <ENTER> key to continue...")

c = Cartpole(BOUNDS = [[-2.4, 2.4], [-2, 2], [-0.21, 0.21], [-2, 2]], nBUCKETS = [1, 1, 20, 20], MIN_EXPLORE_RATE = 0.05, MIN_LEARNING_RATE = 0.05, DISCOUNT = 0.99)
print('converged in {} episodes'.format(c.learn(nEPISODES = 100000)))
c.test()