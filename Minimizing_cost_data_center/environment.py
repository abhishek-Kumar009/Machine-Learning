#Building the Environment
import numpy as np

class Environment(object):
  #INTRODUCING THE PARAMETERES AND VARIABLES AND INITIALIZING THEM
  def __init__(self, optimal_temperature = (18.0, 24.0), initial_num_users = 10, initial_data_rate = 60, initial_month = 0):

    self.optimal_temperature = optimal_temperature
    self.monthly_avg_temperature = [1.0, 5.0, 7.0, 10.0, 11.0, 20.0,3.0, 24.0, 22.0, 10.0, 5.0, 1.0]
    self.initial_month = initial_month
    self.atmos_temperature =self.monthly_avg_temperature[initial_month]

    self.initial_num_users = initial_num_users
    self.current_num_users = initial_num_users
    self.max_num_users = 100
    self.min_num_users = 10
    self.max_update_users = 10

    self.initial_data_rate = initial_data_rate
    self.current_data_rate = initial_data_rate
    self.max_data_rate = 300.0
    self.min_data_rate = 20.0
    self.max_update_data_rate = 20.0   
    
    self.min_temperature_server = -20.0
    self.max_temperature_server = 80.0
    self.reward = 0.0
    self.game_over = 0
    self.training = True

    self.intrinsic_temperature = self.atmos_temperature + 1.25*self.initial_num_users + 1.25*self.initial_data_rate
    self.temperature_ai = self.intrinsic_temperature
    self.temperature_noai = (self.optimal_temperature[0] + self.optimal_temperature[1])/2.0
    self.total_energy_ai = 0.0
    self.total_energy_noai = 0.0
  #UPDATING THE ENVIRONMENT AFTER AN ACTION HAS BEEN PLAYED BY THE AI
  def update_env(self, direction, energy_ai, month):
    #GETTING THE REWARD
    
    energy_noai = 0
    #Computing the energy spent by the internal cooling system when there is no AI
    if(self.temperature_noai < self.optimal_temperature[0]):
      energy_noai = self.optimal_temperature[0] - self.temperature_noai
      self.temperature_noai = self.optimal_temperature[0]
    elif(self.temperature_noai > self.optimal_temperature[1]):
      energy_noai = self.temperature_noai - self.optimal_temperature[1]
      self.temperature_noai = self.optimal_temperature[1]
    #Calculating the reward
    self.reward = energy_noai - energy_ai
    #Scaling the reward
    self.reward = 1e-3 * self.reward

    #GETTING THE NEXT STATE

    #Updating the atmospheric temperature
    self.atmos_temperature = self.monthly_avg_temperature[month]
    #Updating the number of users
    self.current_num_users += np.random.randint(-self.max_update_users, self.max_update_users)
    if(self.current_num_users > self.max_num_users):
      self.current_num_users = self.max_num_users
    elif(self.current_num_users < self.min_num_users):
      self.current_num_users = self.min_num_users
    #Updating the rate of data     
    self.current_data_rate += np.random.randint(-self.max_update_data_rate, self.max_update_data_rate)
    if(self.current_data_rate > self.max_data_rate):
      self.current_data_rate = self.max_data_rate
    elif(self.current_data_rate < self.min_data_rate):
      self.current_data_rate = self.min_data_rate   
    #Computing the delta of intrinsic temperature
    previous_intrinsic_temperature = self.intrinsic_temperature
    self.intrinsic_temperature = self.atmos_temperature + 1.25 * self.current_num_users + 1.25 * self.current_data_rate
    delta_intrinsic_temperature = self.intrinsic_temperature - previous_intrinsic_temperature
    #Updating the Server's temperature when AI is present
    if(direction == -1):
      self.temperature_ai += delta_intrinsic_temperature - energy_ai
    elif(direction == 1):
      self.temperature_ai += delta_intrinsic_temperature + energy_ai
    #Updating the Server's temperature when AI is absent
    self.temperature_noai += delta_intrinsic_temperature
    
    #GAME OVER
    if(self.temperature_ai > self.max_temperature_server):
      if(self.training == 1):
        self.game_over = 1
      else:        
        self.total_energy_ai += self.temperature_ai - self.optimal_temperature[1]
        self.temperature_ai = self.optimal_temperature[1]
    elif(self.temperature_ai < self.min_temperature_server):
      if(self.training == 1):
        self.game_over = 1
      else:        
        self.total_energy_ai += self.optimal_temperature[0] - self.temperature_ai
        self.temperature_ai = self.optimal_temperature[0]
    
    #UPDATING THE SCORES
    #Updating the total energy spent by AI
    self.total_energy_ai += energy_ai
    #Updating the total energy spent by server's internal cooling system
    self.total_energy_noai += energy_noai

    #RETURN THE STATE, REWARD AND GAME_OVER

    scaled_temperature_ai = (self.temperature_ai - self.min_temperature_server)/(self.max_temperature_server - self.min_temperature_server)
    scaled_current_num_users = (self.current_num_users - self.min_num_users)/(self.max_num_users - self.min_num_users)
    scaled_current_data_rate = (self.current_data_rate - self.min_data_rate)/(self.max_data_rate - self.min_data_rate)
    next_state = np.matrix([scaled_temperature_ai, scaled_current_num_users ,scaled_current_data_rate])
    #RETURNING NEXT STATE, REWARD, GAME OVER
    return next_state, self.reward, self.game_over
  
  #METHOD THAT RESETS THE ENVIRONMENT

  def reset_env(self, new_month):
    self.atmos_temperature = self.monthly_avg_temperature[new_month]
    self.initial_month = new_month
    self.current_num_users = self.initial_num_users
    self.current_data_rate = self.initial_data_rate
    self.intrinsic_temperature = self.atmos_temperature + 1.25 * self.current_num_users + 1.25 * self.current_data_rate
    self.temperature_ai = self.intrinsic_temperature
    self.temperature_noai = (self.optimal_temperature[0] + self.optimal_temperature[1])/2.0
    self.total_energy_ai = 0.0
    self.total_energy_noai = 0.0
    self.game_over = 0
    self.training = 1
    self.reward = 0.0
  
  #METHOD TO RETURN CURRENT STATE, LAST REWARD AND GAME OVER STATUS

  def observe(self):    
    scaled_temperature_ai = (self.temperature_ai - self.min_temperature_server)/(self.max_temperature_server - self.min_temperature_server)
    scaled_current_num_users = (self.current_num_users - self.min_num_users)/(self.max_num_users - self.min_num_users)
    scaled_current_data_rate = (self.current_data_rate - self.min_data_rate)/(self.max_data_rate - self.min_data_rate)
    current_state = np.matrix([scaled_temperature_ai, scaled_current_num_users ,scaled_current_data_rate])

    return current_state, self.reward, self.game_over
  
  

