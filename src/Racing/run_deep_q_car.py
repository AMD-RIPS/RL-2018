import deep_q_car

# Train Cartpole
steering = [-1, -.5, 0, .5, 1]
acceleration = [0, .5, 1]
deceleration = [0, .5, 1]
print ('---- CarRacing ----')
cr = deep_q_car.Playground('CarRacing-v0', 2, [24,24], .4, .01, .05, .99, 32, 10000, steering, acceleration, deceleration)
print '------------------- HERE! ----------------------'
cr.begin_training(10000)
cr.test_Q(100)