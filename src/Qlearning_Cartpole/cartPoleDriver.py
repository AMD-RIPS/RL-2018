import gym
import qLearning as QL
import cartpolePreprocessing as crtpl

env = gym.make('CartPole-v0')
fitQ = QL.learn(env)
crtpl.pause()
QL.test(env, fitQ)
