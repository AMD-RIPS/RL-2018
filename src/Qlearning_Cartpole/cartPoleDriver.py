import gym
import qLearning as QL
import cartpolePreprocessing as crtpl

env = gym.make('CartPole-v0')
fitQ, trainingScores = QL.learn(env)
print(trainingScores)
crtpl.pause()
testScores = QL.test(env, fitQ)
print(testScores)