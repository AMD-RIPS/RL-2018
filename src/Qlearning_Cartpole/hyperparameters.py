# Bounds for the state space, bounds for index 0 and 2 are as returned by env.observation_space.low
BOUNDS = [[-1, 1], [-2, 2], [-0.21, 0.21], [-2, 2]]
nBUCKETS = [3, 3, 30, 30]

MIN_EXPLORE_RATE = 0.05
nEPISODES = 3000
MAXTIME = 250
MIN_LEARNING_RATE = 0.05
DISCOUNT = 0.99

INTERVAL = 20