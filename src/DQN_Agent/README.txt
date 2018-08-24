This folder contains the necessary files to recreate the results in the final report submitted by RIPS AMD team. 

This folder contains the following files:
agent.py
	This file defines the initializers for hyperparameters and training functions used for updating the neural network.

environment.py
	This file defines the functions used for interactions with the envrionment.

main.py
	This is the main file to call to define training environement and start training.

replay_memory.py
	This file defines the experience replay memory functions, including the length and adding to the memory.

tester.py
	This file is a simple function to test the checkpoints saved during training.

utils.py
	This file contains image preprocessing functions and a pause function for debugging.

parameters/architecture.py
	This file defines the network architecture, 
	available architectures include: 
		-Basic_Architecture: two fully connected neural network layers with 32 neurons at each layer;
    	-Conv_1Layer: one convolutional layer with 16 8x8 kernels with strides 4x4 followed by one dense layer with dropout;
    	-Conv_2Layer: two convolutional layers, one with 16 8x8 kernels with strides 4x4, the second one with 32 4x4 kernels with strides 2x2, 		connected to one dense layer with dropout; 
    	-Atari_Paper: the same network architecture used in the paper published by DeepMind in 2013
    	-Nature_Paper: the same network architecture used in the nature paper published by DeepMind in 2015
    	-Nature_Paper_Batchnorm: based on the nature architecture, added batchnorm to each layer
    	-Nature_Paper_Dropout: based on the nature architecture, added dropout to the dense and output layer
    	-Nature_Paper_Conv_Dropout: based on the nature architecture, added dropout to the second convolutional layer.

parameters/explore_rates.py
	This file defines the explore rate decay
	avaliable explore rate dacay option:
		-Basic_Explore_Rate: decays from 0.5 to 0.1 linearly over num_episodes defined above
		-Atari_Explore_Rate: decays from 0.6 to 0.1 linearly over learning_rate_drop_frame_limit defined above
		-Decay_Explore_Rate: decays from 1 to 0.1 linearly over learning_rate_drop_frame_limit defined above
		-Fixed_Explore_Rate: fixed to be 0.1

parameters/learning_rates.py
	This file defines the learning rate
	available learning rate option:
		-Basic_Learning_Rate: decays from 0.005 to 0.001 over num_episodes defined above
		-Atari_Learning_Rate: fixed to be 0.00025

parameters/setup.py
	This file defines the hyperparameter setup

Environments/*
	This folder contains the environment files used for simulations. Pleae refer to Environments/new_env_instructions.txt for more instructions.

Follow the instruction in Environments/new_env_instructions.txt. There are three training environments available to choose from, which include random short tracks, fixed one track and fixed three track. The training environment can be defined in main.py. To train in random short tracks environment, uncomment line 19; to train in fixed one/three tracks environment, uncomment line 22 with seed = fixed_1track_seed/fixed_3track_seed. 

To start training, run the following command: 
python main.py <folder_name_for_saving_the_model>

The models and the tensorboard log will be saved in <folder_name_for_saving_the_model>. 