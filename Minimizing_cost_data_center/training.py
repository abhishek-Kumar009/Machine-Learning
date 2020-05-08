#IMPORTING THE LIBRARIES
import numpy as np
import os
import random as rn
import environment
import brain
import dqn
import matplotlib.pyplot as plt

#SETTING THE SEED FOR REPRODUCIBILITY
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)

#INITIALIZING THE PARAMETERES AND VARIABLES
max_memory = 3000
num_epochs = 100
epsilon = 0.3
num_of_actions = 5
batch_size = 512
direction_boundary = (num_of_actions - 1)/2
temperature_step = 2.5

#CREATING THE OBJECTS OF THE VARIOUS CLASSES

#ENVIRONMENT SETUP
env = environment.Environment(optimal_temperature = (18.0, 24.0), initial_num_users = 20, initial_data_rate = 30, initial_month = 0)
#BRAIN SETUP
brain = brain.Brain(learning_rate = 0.00001, number_of_actions = num_of_actions)
#SETTING UP THE DEEP Q-LEARNING NETWORK
dqn = dqn.DQN(max_memory = max_memory, discount = 0.9)

training = True
model = brain.model

#SET THE MODE
env.training = training

#EARLY STOPPING PARAMETERS
patience = 20
best_total_reward = -np.inf
patience_count = 0
early_stopping = True

if(env.training):
    loss_array = []
    epoch_list = []
    total_reward_list = []
    for epoch in range(1, num_epochs):
        total_reward = 0        
        timestep = 0
        current_state, _, _ = env.observe()
        game_over = False
        loss = 0
        new_month = np.random.randint(0, 12)
        
        #RESET THE ENVIRONMENT 
        env.reset_env(new_month = new_month)
        
        while((not game_over) and timestep <= 5 * 30 * 24 * 60):
            #EXPLORE OR EXPLOIT 
            if(np.random.rand() <= epsilon):
                #EXPLORE
                action = np.random.randint(0, num_of_actions)
                if(action - direction_boundary < 0):
                    #COOLING DOWN THE SYSTEM                    
                    direction = -1
                else:
                    #HEATING UP THE SYSTEM                    
                    direction = 1
                energy_ai = abs(action - direction_boundary) * temperature_step
            else:
                #EXPLOIT
                #Get the Q-values by passing the current state into the ANN
                Q_values = model.predict(current_state)
                action = np.argmax(Q_values[0])
                if(action - direction_boundary < 0):
                    #COOLING DOWN THE SYSTEM                    
                    direction = -1
                else:
                    #HEATING UP THE SYSTEM                    
                    direction = 1
                energy_ai = abs(action - direction_boundary) * temperature_step
            
            #UPDATE THE ENVIRONMENT AFTER TAKING THE ACTION
            next_state, reward, game_over = env.update_env(direction, energy_ai, int(timestep/(30*24*60)))
            total_reward += reward
            
            #RECORD THE TRANSITION IN THE MEMORY
            dqn.remember([current_state, action, reward, next_state], game_over)
            
            #MAKE BATCHES OF INPUTS AND TARGETS TO TRAIN WITH THE ANN
            inputs, targets = dqn.get_batch(model, batch_size = batch_size)
            #TRAIN THE ANN WITH THESE BATCHES OF DATA
            loss += model.train_on_batch(inputs, targets)
            
            timestep += 1
            current_state = next_state
        
        loss_array.append(loss)
        epoch_list.append(epoch)
        total_reward_list.append(total_reward)
        
        print("\n")
        print("Epoch: {:03d}/{:03d}".format(epoch, num_epochs))
        print("Energy spent with AI: {:0.0f}".format(env.total_energy_ai))
        print("Energy spent without AI: {:0.0f}".format(env.total_energy_noai))
        
        #EARLY STOPPING ALGORITHM
        print("Best Total Reward:", best_total_reward)
        print("Total Reward",total_reward)
        if(early_stopping):
            if(total_reward <= best_total_reward):
                patience_count += 1
            elif(total_reward > best_total_reward):
                best_total_reward = total_reward
                patience_count = 0
            if(patience_count >= patience):
                print("Early Stopping!")
                break
        
        
        #save the model parameters
        model.save("MinimizeCostModel(ES).h5")
print("Loss vs epochs curve: \n")

plt.figure(1)
plt.plot(epoch_list, loss_array)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

plt.figure(2)
plt.plot(epoch_list, total_reward_list)
plt.xlabel("Epoch")
plt.ylabel("Total Reward")
plt.show()
            
            
            
                
                
    

