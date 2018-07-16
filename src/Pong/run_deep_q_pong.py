import deep_q_pong as game

# Train Cartpole
steering = [-1, 0, 1]
acceleration = [0, 1]
deceleration = [0, 0.8]
print ('---- Pong ----')
pong = game.Playground('Pong-v0', 2, [24,24], .4, .01, .05, .99, 32, 10000)
pong.begin_training(1000, 4)