import numpy as np
import random as rand
import math
import cartpolePreprocessing as crtpl

MIN_EXPLORE_RATE = 0.05
nEPISODES = 2000
MAXTIME = 250
MIN_LEARNING_RATE = 0.05
DISCOUNT = 0.99

def eGreedy(q, time):
    A = len(q)
    u = rand.random()
    greedy = np.argmax(q)

    # Decrease exploration over time
    epsilon = max(MIN_EXPLORE_RATE, min(1, 2.544 - math.log10(time + 1)))
    return greedy if u > epsilon else int(rand.random()*A)

def greedy(q):
    return np.argmax(q)

def getStatespaceSize():
    m = max(crtpl.nBUCKETS)
    return (m - 1)*(1 + m + m**2 + m**3) + 1

def learn(env, visualise = False):
    S = getStatespaceSize()
    A = 2
    Q = [[rand.random() for i in range(A)] for j in range(S)]

    # Iterating over episodes
    for iterations in range(nEPISODES):
        observation = env.reset()
        state = crtpl.preprocess(observation)

        # Stepping through episode
        for time in range(MAXTIME):
            if visualise: env.render()
            action = eGreedy(Q[state], time)
            newObservation, reward, done, info = env.step(action)
            newState = crtpl.preprocess(newObservation)

            # Decrease learning rate over time
            learningRate = max(MIN_LEARNING_RATE, min(1, 2.544 - math.log10(time + 1)))
            # Performing Q-learning update
            Q[state][action] += learningRate*(reward + DISCOUNT*np.max(Q[newState]) - Q[state][action])
            state = newState
            if done: break
    return Q

# Given a fitted Q, visualise how well it does
def test(env, Q, nEpisodes = 10, maxTime = 250):
    for iterations in range(nEpisodes):
        observation = env.reset()
        state = crtpl.preprocess(observation)

        # Stepping through episode
        for time in range(maxTime):
            env.render()
            action = greedy(Q[state])
            newObservation, reward, done, info = env.step(action)
            newState = crtpl.preprocess(newObservation)
            state = newState
            if done: break


