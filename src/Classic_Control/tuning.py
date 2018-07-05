from classicControl import ClassicControl

fixedExploreRates = [0.1, 0.2, 0.3]
fixedLearningRates = [0.001, 0.01, 0.1]

result = '5000 training episodes\n'
for learningRate in fixedLearningRates:
	for exploreRate in fixedExploreRates:
		cumscore = 0
		for _ in range(5):
			car = ClassicControl(game = 'MountainCar-v0', bounds = [[-1.2, 1.2], [-0.07, 0.07]], nBuckets = [10, 30], aSize = 3)
			car.setLearningParameters(minLearningRate = learningRate, minExploreRate = exploreRate, fixedERate = True, fixedLRate = True, nEpisodes = 5000)
			car.learn()
			cumscore += car.test(visualize = False)
		result += 'fixed Learning Rate = {0}, fixed Exploration Rate = {1}, average reward = {2}\n\n'.format(learningRate, exploreRate, cumscore/5.0)

file = open('TuningResults.txt', 'w')
file.write(result)
file.close()
