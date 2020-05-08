import numpy as np
import os
import random as rn
import environment
from keras.models import load_model

#SET THE SEED FOR REPRODUCIBILITY
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(38)
rn.seed(12348)

#SETTING UP THE PARAMETERS AND VARIABLES
num_actions = 5
temperature_step = 2.5
direction_boundary = (num_actions - 1)/2

#SETTING UP THE ENVIRONMENT
env = environment.Environment(optimal_temperature = (18.0, 24.0), initial_num_users = 20, initial_data_rate = 30, initial_month = 0)
#LOADING THE PRE-TRAINED BRAIN
model = load_model("MinimizeCostModel(ES).h5")

#SETTING UP THE MODE
train = False
env.training = train

current_state, _, _ = env.observe()

for timestep in range(0, 12 * 30 * 24 * 60):
    Q_values = model.predict(current_state)
    action = np.argmax(Q_values[0])
    if(action - direction_boundary <= 0):
        #AI COOLING DOWN THE SERVER
        direction = -1
    elif(action - direction_boundary > 0):
        #AI HEATING UP THE SYSTEM
        direction = 1
    energy_ai = abs(action - direction_boundary) * temperature_step
        
    #UPDATE THE ENVIRONMENT
    next_state, reward, game_over = env.update_env(direction, energy_ai, int(timestep/(30 * 24 * 60)))
        
    current_state = next_state
    print("Total Energy AI:", env.total_energy_ai)
    print("Total Energy no AI:", env.total_energy_noai)

energy_saved = ((env.total_energy_noai - env.total_energy_ai)/env.total_energy_noai) * 100
print("\nTotal Energy spent by AI: {:0.3f}".format(env.total_energy_ai))
print("Total Energy spent without AI: {:0.3f}".format(env.total_energy_noai))
print("Energy saved by AI: {:0.3f}%".format(energy_saved))

