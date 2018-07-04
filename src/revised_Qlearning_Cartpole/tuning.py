import cartpole

bucketXs = [1, 3, 6, 10]
bucketThetas = [6, 10, 20, 30]
minExploreRates = [0.05]
minLearningRates = [0.05]
discounts = [0.99]

BOUNDS = [[-2.4, 2.4], [-2, 2], [-0.21, 0.21], [-2, 2]]

for bucketX in bucketXs:
	for bucketTheta in bucketThetas:
		for minLearningRate in minLearningRates:
			for minExploreRate in minExploreRates:
				for discount in discounts:
					c = cartpole.Cartpole(BOUNDS = BOUNDS, nBUCKETS = [bucketX, bucketX, bucketTheta, bucketTheta], MIN_EXPLORE_RATE = minExploreRate, MIN_LEARNING_RATE = minLearningRate, DISCOUNT = discount)
					convergenceSpeed = 0
					for _ in range(5):
						convergenceSpeed += c.learn(nEPISODES = 1000)
					print('bucketX = {0}, bucketTheta = {1}, convergencespeed = {2}'.format(bucketX, bucketTheta, convergenceSpeed/5.0))

# Current optimal parameter values are bucketX = 1 or 3 and bucketTheta = 20 or 30, with convergence in well under 200 episodes