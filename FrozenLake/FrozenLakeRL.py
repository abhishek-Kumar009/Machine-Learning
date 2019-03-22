
# coding: utf-8

# In[27]:


import numpy as np
import gym
import random
import time
from IPython.display import clear_output

env = gym.make("FrozenLake-v0")

action_space_size = env.action_space.n
state_space_size = env.observation_space.n


q_table = np.zeros((state_space_size, action_space_size))

print(q_table)

total_episodes = 10000
max_steps_in_episode = 100
discount_rate = 0.99
learning_rate = 0.1
exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

rewards_all_episodes = []
for episode in range(total_episodes):
    state = env.reset()
    done = False
    reward_current_episode = 0
    for steps in range(max_steps_in_episode):
        #Decide either to exploit or explore
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            #exploit
            action = np.argmax(q_table[state, :])
        else:
            #explore
            action = env.action_space.sample()
        
        new_state, reward, done, info = env.step(action)
                
        #Update q_table
        q_table[state, action] = (1 - learning_rate)*q_table[state, action] + learning_rate*(reward + discount_rate*np.max(q_table[new_state, :]))
        
        state = new_state
        reward_current_episode += reward
        
        if done == True:
            break
    rewards_all_episodes.append(reward_current_episode)
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate)*(np.exp(-exploration_decay_rate*episode))
rewards_per_th_eps = np.split(np.array(rewards_all_episodes), total_episodes/1000)
count = 1000
for i in rewards_per_th_eps:
    print(count, " :",sum(i/1000),"\n")
    count+= 1000

        


# In[30]:


type(q_table)


# In[35]:


max_steps_in_episode = 100
for episode in range(3):
    env.reset()
    print("Episode:", episode + 1,"\n")
    time.sleep(1)
    done = False
    for step in range(max_steps_in_episode):
        clear_output(wait = True)
        env.render()
        time.sleep(0.3)
        action = np.argmax(q_table[state, :])
        new_state, reward, done, info = env.step(action)
        
        if done:
            
            clear_output(wait = True)
            env.render()
            time.sleep(1)
            if reward == 1:
                print("You made it to the freesbe!!")
                time.sleep(3)
            else:
                print("You fell into a hole!!")
                time.sleep(3)
            clear_output(wait = True)
            break
        state = new_state
env.close()
        
            
        
        

