import gym
import math
import numpy as np
import copy

# Initialize environment
env = gym.make('CartPole-v1')

# Greedy parameter
epsilon_max = .9
epsilon_min = .01

# Discount factor
gamma = .9

# Learning rate
alpha_max = .05
alpha_min = .001
# alpha = .05

# Set action parameters
ACTION_SIZE = env.action_space.n

# Set state space parameters
DISC_POS_SIZE = 3
DISC_V_SIZE = 10
DISC_ANGLE_SIZE = 30
DISC_AV_SIZE = 30

# Fixed values
STATE_SIZE = 4
BUCKETS = (DISC_POS_SIZE, DISC_V_SIZE, DISC_ANGLE_SIZE, DISC_AV_SIZE)
CUM_PROD = [1, np.prod(BUCKETS[0:1]), np.prod(BUCKETS[0:2]), np.prod(BUCKETS[0:3])]
SPACE_SIZE = ACTION_SIZE*DISC_POS_SIZE*DISC_V_SIZE*DISC_ANGLE_SIZE*DISC_AV_SIZE

# Get bounds of state space
# STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
STATE_BOUNDS = [[-4.8, 4.8], [-5, 5], [-.5, .5], [-5, 5]]


#####			      		#####
#####		Q-Function		#####
##### 						#####
# Takes an interval and discretizes it into 'size' buckets
def discretized_state_bucket(interval, size, value):
	interval_length = (interval[1]+0.0)/size - (interval[0]+0.0)/size
	return int(np.floor(value/interval_length - (0.0+interval[0])/interval_length))

# Given an action and state vector, returns corresponding index of q function
def get_q_index(action, state_array):
	index = action*SPACE_SIZE/ACTION_SIZE
	# buckets = np.zeros(4)
	for i in range(STATE_SIZE):
		bucket = discretized_state_bucket(STATE_BOUNDS[i],BUCKETS[i],state_array[i])
		index = index + CUM_PROD[i]*bucket
		# buckets[i] = bucket
	return index


def get_q_action(S):
	zero_index = get_q_index(0, S)
	one_index = get_q_index(1, S)
	if (Q[zero_index] > Q[one_index]):
		return 0
	else:
		return 1

def get_max_q_value(state):
	zero_index = get_q_index(0, state)
	one_index = get_q_index(1, state)
	if (Q[zero_index] > Q[one_index]):
		return Q[zero_index]
	else:
		return Q[one_index]

def get_action(state, eps):
	if (np.random.uniform() < eps):
		return np.random.binomial(1,.5)
	else:
		return get_q_action(state)

def compute_qnorm(Q_old):
	return np.linalg.norm(np.subtract(Q_old,Q), 2)

# Initialize q function
# Q = np.random.random([SPACE_SIZE])
Q = np.zeros(SPACE_SIZE)

num_epsidoes = 20000
eps_decay_rate = (epsilon_min - epsilon_max + 0.0)/num_epsidoes
alpha_decay_rate = (alpha_min - alpha_max + 0.0)/num_epsidoes
gamma_plot_array = np.zeros(num_epsidoes)
for i in range(num_epsidoes):
	# Initialize state
	# [Position, Velocity, Angle, Angular velocity]
	S = env.reset()
	done = False
	env.reset()
	Q_old = copy.copy(Q)
	while not done:
		action = get_action(S, epsilon_max + eps_decay_rate*i)
		new_state, reward, done, _ = env.step(action)
		index = get_q_index(action, S)
		alpha = alpha_max + alpha_decay_rate*i
		Q[index] += alpha*(reward + gamma*get_max_q_value(new_state) - Q[index])
		S = new_state
		if (i > 19990):
			env.render()
	gamma_plot_array[i] = compute_qnorm(Q_old)

# print env.observation_space.low
np.savetxt('gamma_values.csv', gamma_plot_array, delimiter=',')