import deep_q_car_gpu

# Train Cartpole
steering = [-1, 1]
acceleration = [0, 1]
deceleration = [0, 0.8]
print ('---- CarRacing ----')
cr = deep_q_car_gpu.Playground('CarRacing-v0', 2, [24,24], .4, .01, .05, .99, 32, 10000, steering, acceleration, deceleration)
cr.begin_training(10000)
cr.test_Q(100)