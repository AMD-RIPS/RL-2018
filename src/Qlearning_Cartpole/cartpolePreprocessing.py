import math
import hyperparameters as _

BOUNDS = _.BOUNDS
nBUCKETS = _.nBUCKETS

# Returns a non-negative integer, representing the state the game is in
# In the cartpole game the state is represented as 4 real numbers, we map these 4 dimensional 
# vectors to a non-negative integer. 
def preprocess(observation, bounds = BOUNDS, nBuckets = nBUCKETS):
	buckets = []
	n = len(observation)

	for index in range(n):
		buckets.append(getBucket(observation[index], bounds[index][0], bounds[index][1], nBuckets[index]))

	# map to a non-negative integer
	m = max(nBuckets)
	answer = int(buckets[0] + m*buckets[1] + m**2 *buckets[2] + m**3 *buckets[3])
	return answer

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

def pause():
    programPause = raw_input("Press the <ENTER> key to continue...")