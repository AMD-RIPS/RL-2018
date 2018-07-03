import math

# Bounds for the state space, bounds for index 0 and 2 are as returned by env.observation_space.low
BOUNDS = [[-2.4, 2.4], [-2, 2], [-0.21, 0.21], [-2, 2]]
nBUCKETS = [2, 2, 30, 30]

# Returns a non-negative integer, representing the state the game is in
# In the cartpole game the state is represented as 4 real numbers, we map these 4 dimensional 
# vectors to a non-negative integer. 
def preprocess(observation):
	buckets = []
	n = len(observation)

	for index in range(n):
		buckets.append(getBucket(observation[index], BOUNDS[index][0], BOUNDS[index][1], nBUCKETS[index]))

	# map to a non-negative integer
	m = max(nBUCKETS)
	answer = int(buckets[0] + m*buckets[1] + m**2 *buckets[2] + m**3 *buckets[3])
	return answer

def getBucket(value, lowerB, upperB, nBuckets):
	# Check whether out of bounds
	if value < lowerB:
			bucket = 0
	elif value > upperB:
		bucket = nBuckets - 1

	# assign bucket
	else:
		ran = upperB - lowerB
		bucket = math.floor(nBuckets*(value - lowerB) / ran)
	return bucket

def pause():
    programPause = raw_input("Press the <ENTER> key to continue...")



