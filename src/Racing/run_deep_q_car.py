import deep_q_car

# Train Cartpole
steering = [-1, 0, 1]
acceleration = [0, 1]
deceleration = [0, 0.8]
print ('---- CarRacing ----')
cr = deep_q_car.Playground('CarRacing-v0', 2, [24,24], .4, .01, .05, .99, 32, 10000, steering, acceleration, deceleration)
cr.begin_training(1000, 4)