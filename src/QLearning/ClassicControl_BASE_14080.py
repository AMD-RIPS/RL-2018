import numpy as np
import gym
import math
import random as rand
import matplotlib.pyplot as plt
import copy

class ClassicControl():

    def __init__(self, game, nBuckets, bounds, aSize):
        self.game = game
        self.env = gym.make(game)
        self.nBuckets = nBuckets
        self.bounds = bounds
        self.sSize = self.getStatespaceSize()
        self.aSize = aSize
        randomVector = np.random.normal(size = self.aSize * self.sSize)
        self.Q = [[randomVector[self.aSize * j + i] for i in range(self.aSize)] for j in range(self.sSize)]
        self.setLearningParameters()
        # Keeping track of time spent training
        self.trainingTime = 0

    def setLearningParameters(self, nEpisodes = 1000, maxEpisodeLength = 500, discount = 0.99, minLearningRate = 0.05, minExploreRate = 0.1):
        self.nEpisodes = nEpisodes
        self.maxEpisodeLength = maxEpisodeLength
        self.discount = discount
        self.minLearningRate = minLearningRate
        self.minExploreRate = minExploreRate

    def getStatespaceSize(self):
        m = max(self.nBuckets)
        return (m - 1) * sum([m**i for i in range(len(self.nBuckets))]) + 1

    def learn(self, plot = False):
        avgQ = []
        normQ = []
        rewards = []
        episodes = [i for i in range(0, self.nEpisodes)]

        for episode in range(self.nEpisodes):
            oldQ = copy.deepcopy(self.Q)

            # run episode
            reward = self.runEpisode()
 
            # Checking whether the algorithm has converged
            if episode%20 == 0 and self.converged(): break

            # Record results for plotting
            if plot:
                avgQ.append(np.average(self.Q))
                normQ.append(np.linalg.norm(np.subtract(self.Q, oldQ), 2))
                rewards.append(reward)
                
        if plot:
            # Plot Average Q Values
            plt.plot(episodes, avgQ)
            plt.xlabel("Episode")
            plt.ylabel("Average Q Value")
            plt.title("Average Q Value for " + self.game)
            plt.show()
            # Plot Q Norm
            plt.plot(episodes, normQ)
            plt.xlabel("Episode")
            plt.ylabel("Q Norm")
            plt.title("Q Norm for " + self.game)
            plt.show()
            # Plot Average Rewards
            plt.plot(episodes, rewards)
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.title("Rewards by Episode for " + self.game)
            plt.show()
            
        return episode

    def converged(self): 
        if self.game == 'CartPole-v0' and self.test(visualise = False) > 190: return True
        return False

    def runEpisode(self):
        observation = self.env.reset()
        state = self.preprocess(observation)
        totalReward = 0
        for time in range(self.maxEpisodeLength):
            action = eGreedy(self.Q[state], self.getExploreRate())
            newObservation, reward, done, info = self.env.step(action)
            totalReward += reward    
            newState = self.preprocess(newObservation)
            alpha = self.getLearningRate()

            # Performing Q-learning update
            self.Q[state][action] += alpha * (reward + self.discount * np.max(self.Q[newState]) - self.Q[state][action])
            state = newState
            if done: break
            self.trainingTime += 1
        return totalReward

    def getExploreRate(self): return max(self.minExploreRate, min(1, 5 - math.log10(self.trainingTime + 1)))

    def getLearningRate(self): return max(self.minLearningRate, min(1, 5 - math.log10(self.trainingTime + 1)))


    def test(self, testEpisodes = 10, testMaxTime = 250, visualise = True):
        testScores = []
        for episode in range(testEpisodes):
            observation = self.env.reset()
            state = self.preprocess(observation)
            totalReward = 0
            # Stepping through episode
            for time in range(testMaxTime):
                if visualise: self.env.render()
                action = greedy(self.Q[state])
                newObservation, reward, done, info = self.env.step(action)
                totalReward += reward
                newState = self.preprocess(newObservation)
                state = newState
                if done: break
            testScores.append(totalReward)
            #print(totalReward)
        return sum(testScores)/float(testEpisodes)


    def preprocess(self, observation):
        buckets = []
        n = len(observation)

        for index in range(n):
            buckets.append(getBucket(observation[index], self.bounds[index][0], self.bounds[index][1], self.nBuckets[index]))

        # map to a non-negative integer
        m = max(self.nBuckets)
        n = len(buckets)
        return int(sum([m**j * buckets[j] for j in range(n)]))

def eGreedy(q, epsilon):
    aSize = len(q)
    u = rand.random()
    greedy = np.argmax(q)

    # Decrease exploration over time
    # epsilon = max(minExploreRate, min(1, 2.544 - math.log10(time + 1)))
    return greedy if u > epsilon else int(rand.random()*aSize)

def greedy(q): return np.argmax(q)

def getBucket(value, lowerB, upperB, nBuckets):
    # Check whether out of bounds
    if value < lowerB: bucket = 0
    elif value > upperB: bucket = nBuckets - 1

    # assign buckets
    else:
        ran = upperB - lowerB
        bucket = math.floor(nBuckets*(value - lowerB) / ran)
    return bucket

def pause(): programPause = raw_input("Press the <ENTER> key to continue...")

#pole = ClassicControl(game = 'CartPole-v0', bounds = [[-2.4, 2.4], [-2, 2], [-0.21, 0.21], [-2, 2]], nBuckets = [1, 1, 20, 20], aSize = 2)
#pole.setLearningParameters(nEpisodes = 100, maxEpisodeLength = 200, discount = 0.99, minLearningRate = 0.05, minExploreRate = 0.1)
#print('Converged in {} episodes'.format(pole.learn(plot=True)))
#pause()
#pole.test()

car = ClassicControl(game = 'MountainCar-v0', bounds = [[-1.2, 1.2], [-0.07, 0.07]], nBuckets = [10, 30], aSize = 3)
car.setLearningParameters(nEpisodes = 10000, discount = 0.99, minLearningRate = 0.05, minExploreRate = 0.1)
car.learn(plot=True)
pause()
result = car.test()
print result
