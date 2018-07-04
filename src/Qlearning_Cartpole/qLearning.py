import numpy as np
import random as rand
import math
import cartpolePreprocessing as crtpl
import hyperparameters as _

nEPISODES = _.nEPISODES
MIN_EXPLORE_RATE = _.MIN_EXPLORE_RATE
MAXTIME = _.MAXTIME
MIN_LEARNING_RATE = _.MIN_LEARNING_RATE
DISCOUNT = _.DISCOUNT

INTERVAL = _.INTERVAL

# Takes array, returns e-greedy action
def eGreedy(q, time):
    A = len(q)
    u = rand.random()
    greedy = np.argmax(q)

    # Decrease exploration over time
    epsilon = max(MIN_EXPLORE_RATE, min(1, 2.544 - math.log10(time + 1)))
    return greedy if u > epsilon else int(rand.random()*A)

# Takes an array, returns greedy action
def greedy(q):
    return np.argmax(q)

def getStatespaceSize():
    m = max(crtpl.nBUCKETS)
    return (m - 1)*(1 + m + m**2 + m**3) + 1

def learn(env):
    S = getStatespaceSize()
    A = 2
    randomVector = np.random.normal(size = A*S)
    Q = [[randomVector[A*j + i] for i in range(A)] for j in range(S)]
    trainingScores = []

    # Iterating over episodes
    for episode in range(nEPISODES):
        observation = env.reset()
        initialState = crtpl.preprocess(observation)

        # Stepping through episode
        Q, length = runEpisode(env, Q, initialState)

        # Keeping track of training performance
        if episode%INTERVAL == 0:
            testScore = test(env, Q, visualise = False)
            trainingScores.append(testScore)
            if testScore > 180:
                print('Episode: ', episode)
                break
    return Q, trainingScores

def runEpisode(env, Q, state):
    for time in range(MAXTIME):
        action = eGreedy(Q[state], time)
        newObservation, reward, done, info = env.step(action)
        newState = crtpl.preprocess(newObservation)

        # Decrease learning rate over time
        learningRate = max(MIN_LEARNING_RATE, min(1, 2.544 - math.log10(time + 1)))

        # Performing Q-learning update
        Q[state][action] += learningRate*(reward + DISCOUNT*np.max(Q[newState]) - Q[state][action])
        state = newState
        if done: break
    return Q, time

# Given a fitted Q, visualise how well it does
def test(env, Q, nEpisodes = 10, maxTime = 250, visualise = True):
    testScores = []
    for episode in range(nEpisodes):
        observation = env.reset()
        state = crtpl.preprocess(observation)

        # Stepping through episode
        for time in range(maxTime):
            if visualise: env.render()
            action = greedy(Q[state])
            newObservation, reward, done, info = env.step(action)
            newState = crtpl.preprocess(newObservation)
            state = newState
            if done: break
        testScores.append(time)
    return sum(testScores)/float(nEpisodes)

