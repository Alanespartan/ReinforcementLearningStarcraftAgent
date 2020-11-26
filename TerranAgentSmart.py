'''
Copyright (C) 2020 Juan Arturo Cruz Cardona - Intelligent Systems Class Project
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License 
as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see https://www.gnu.org/licenses/.
'''
# USEFUL LINKS FOR THE PROJECT
# Units PYSC2: https://github.com/deepmind/pysc2/blob/master/pysc2/lib/units.py 
# Functions PYSC2: https://github.com/deepmind/pysc2/blob/master/pysc2/lib/actions.py 
import random
import numpy as np # Mathematical functions
import pandas as pd  # Manipulate and analize of data
import os
from absl import app
from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
from pysc2.env import sc2_env, run_loop

# Reinforcment Learning Algorithm
class QLearningTable: 
  def __init__(self, actions, learning_rate=0.01, reward_decay=0.9):
    self.actions = actions
    self.learning_rate = learning_rate
    self.reward_decay = reward_decay
    self.count = 0
    self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

  # 90% Chooses preferred action and 10% randomly for extra possibilities
  def choose_action(self, observation, e_greedy=0.9):
    self.check_state_exist(observation)
    if np.random.uniform() < e_greedy:
      state_action = self.q_table.loc[observation, :]
      action = np.random.choice(
          state_action[state_action == np.max(state_action)].index)
    else:
      action = np.random.choice(self.actions)
    return action

  # Takes the state and action and update table accordingly to learn over time
  def learn(self, s, a, r, s_):
    self.check_state_exist(s_)
    q_predict = self.q_table.loc[s, a] # Get the value that was given for taking the action when we were first in the state
    # Determine the maximum possible value across all actions in the current state
    # and then discount it by the decay rate (0.9) and add the reward we received (can be terminal or not)
    if s_ != 'terminal':
      q_target = r + self.reward_decay * self.q_table.loc[s_, :].max()
    else: # Reward from last step of game is better
      q_target = r 
    self.q_table.loc[s, a] += self.learning_rate * (q_target - q_predict)

  def check_state_exist(self, state): # Check to see if the state is in the QTable already, and if not it will add it with a value of 0 for all possible actions.
    if state not in self.q_table.index:
      self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))

# Simple Agent definition (both random and learning agent use this)
class Agent(base_agent.BaseAgent):
  # Base actions both Smart and Random Agent can perform
  actions = ("do_nothing", "harvest_minerals", "build_supply_depot", "build_barracks", "train_marine", "attack")
  
  # HELPER FUNCTIONS
  def get_my_units_by_type(self, obs, unit_type):
    return [unit for unit in obs.observation.raw_units
            if unit.unit_type == unit_type 
            and unit.alliance == features.PlayerRelative.SELF]
  
  def get_enemy_units_by_type(self, obs, unit_type):
    return [unit for unit in obs.observation.raw_units
            if unit.unit_type == unit_type 
            and unit.alliance == features.PlayerRelative.ENEMY]
  
  def get_my_completed_units_by_type(self, obs, unit_type):
    return [unit for unit in obs.observation.raw_units
            if unit.unit_type == unit_type 
            and unit.build_progress == 100
            and unit.alliance == features.PlayerRelative.SELF]
    
  def get_enemy_completed_units_by_type(self, obs, unit_type):
    return [unit for unit in obs.observation.raw_units
            if unit.unit_type == unit_type 
            and unit.build_progress == 100
            and unit.alliance == features.PlayerRelative.ENEMY]

  def get_distances(self, obs, units, xy):
    units_xy = [(unit.x, unit.y) for unit in units]
    return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1) # Normalize the array

  # AGENT ACTIONS 

  def step(self, obs):
    super(Agent, self).step(obs)
    if obs.first():
      command_center = self.get_my_units_by_type(obs, units.Terran.CommandCenter)[0]
      self.base_top_left = (command_center.x < 32)

  def do_nothing(self, obs):
    return actions.RAW_FUNCTIONS.no_op()
  
  def harvest_minerals(self, obs):
    scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
    idle_scvs = [scv for scv in scvs if scv.order_length == 0]
    if len(idle_scvs) > 0:
      mineral_patches = [unit for unit in obs.observation.raw_units
                         if unit.unit_type in [
                           units.Neutral.BattleStationMineralField,
                           units.Neutral.BattleStationMineralField750,
                           units.Neutral.LabMineralField,
                           units.Neutral.LabMineralField750,
                           units.Neutral.MineralField,
                           units.Neutral.MineralField750,
                           units.Neutral.PurifierMineralField,
                           units.Neutral.PurifierMineralField750,
                           units.Neutral.PurifierRichMineralField,
                           units.Neutral.PurifierRichMineralField750,
                           units.Neutral.RichMineralField,
                           units.Neutral.RichMineralField750
                         ]]
      scv = random.choice(idle_scvs)
      distances = self.get_distances(obs, mineral_patches, (scv.x, scv.y))
      mineral_patch = mineral_patches[np.argmin(distances)] 
      return actions.RAW_FUNCTIONS.Harvest_Gather_unit("now", scv.tag, mineral_patch.tag)
    return actions.RAW_FUNCTIONS.no_op()
  
  def build_supply_depot(self, obs):
    supply_depots = self.get_my_units_by_type(obs, units.Terran.SupplyDepot)
    scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
    if (len(supply_depots) == 0 and obs.observation.player.minerals >= 100 and len(scvs) > 0):
      supply_depot_xy = (22, 26) if self.base_top_left else (35, 42)
      distances = self.get_distances(obs, scvs, supply_depot_xy)
      scv = scvs[np.argmin(distances)]
      return actions.RAW_FUNCTIONS.Build_SupplyDepot_pt( "now", scv.tag, supply_depot_xy)
    return actions.RAW_FUNCTIONS.no_op()
    
  def build_barracks(self, obs):
    completed_supply_depots = self.get_my_completed_units_by_type(obs, units.Terran.SupplyDepot)
    barrackses = self.get_my_units_by_type(obs, units.Terran.Barracks)
    scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
    if (len(completed_supply_depots) > 0 and len(barrackses) == 0 and obs.observation.player.minerals >= 150 and len(scvs) > 0):
      barracks_xy = (22, 21) if self.base_top_left else (35, 45)
      distances = self.get_distances(obs, scvs, barracks_xy)
      scv = scvs[np.argmin(distances)]
      return actions.RAW_FUNCTIONS.Build_Barracks_pt("now", scv.tag, barracks_xy)
    return actions.RAW_FUNCTIONS.no_op()
    
  def train_marine(self, obs):
    completed_barrackses = self.get_my_completed_units_by_type(obs, units.Terran.Barracks)
    free_supply = (obs.observation.player.food_cap - obs.observation.player.food_used)
    if (len(completed_barrackses) > 0 and obs.observation.player.minerals >= 100 and free_supply > 0):
      barracks = self.get_my_units_by_type(obs, units.Terran.Barracks)[0]
      if barracks.order_length < 5:
        return actions.RAW_FUNCTIONS.Train_Marine_quick("now", barracks.tag)
    return actions.RAW_FUNCTIONS.no_op()
  
  def attack(self, obs):
    marines = self.get_my_units_by_type(obs, units.Terran.Marine)
    if len(marines) > 0:
      attack_xy = (38, 44) if self.base_top_left else (19, 23)
      distances = self.get_distances(obs, marines, attack_xy)
      marine = marines[np.argmax(distances)]
      x_offset = random.randint(-4, 4)
      y_offset = random.randint(-4, 4)
      return actions.RAW_FUNCTIONS.Attack_pt("now", marine.tag, (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
    return actions.RAW_FUNCTIONS.no_op()

# Takes the list of actions of the base agent, chooses one at random and then execute it
class RandomAgent(Agent):
  def step(self, obs):
    super(RandomAgent, self).step(obs)
    action = random.choice(self.actions)
    return getattr(self, action)(obs)

# Similar to Random Agent but also initialize the QLearning Table to know which actions can perform and learn
class SmartAgent(Agent):
  def __init__(self):
    super(SmartAgent, self).__init__()
    self.qtable = QLearningTable(self.actions)
    self.new_game()

  def reset(self):
    super(SmartAgent, self).reset()
    print(self.qtable.q_table)
    self.qtable.count += 1
    if self.qtable.count == 100:
      self.qtable.q_table.to_excel(r'QLearningTable.xlsx', sheet_name='QLearningTable', index = False)
    self.new_game()
  
  # Start the new game and store actions and states for the reinforcement learning
  def new_game(self):
    self.base_top_left = None
    self.previous_state = None
    self.previous_action = None

  # Takes all the values of the game we find important and then returning those in a tuple to feed into our machine learning algorithm
  def get_state(self, obs):
    scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
    idle_scvs = [scv for scv in scvs if scv.order_length == 0]
    command_centers = self.get_my_units_by_type(obs, units.Terran.CommandCenter)
    supply_depots = self.get_my_units_by_type(obs, units.Terran.SupplyDepot)
    completed_supply_depots = self.get_my_completed_units_by_type(obs, units.Terran.SupplyDepot)
    barrackses = self.get_my_units_by_type(obs, units.Terran.Barracks)
    completed_barrackses = self.get_my_completed_units_by_type(obs, units.Terran.Barracks)
    marines = self.get_my_units_by_type(obs, units.Terran.Marine)
    
    queued_marines = (completed_barrackses[0].order_length if len(completed_barrackses) > 0 else 0)
    
    free_supply = (obs.observation.player.food_cap - obs.observation.player.food_used)
    can_afford_supply_depot = obs.observation.player.minerals >= 100
    can_afford_barracks = obs.observation.player.minerals >= 150
    can_afford_marine = obs.observation.player.minerals >= 100
    
    enemy_scvs = self.get_enemy_units_by_type(obs, units.Terran.SCV)
    enemy_idle_scvs = [scv for scv in enemy_scvs if scv.order_length == 0]
    enemy_command_centers = self.get_enemy_units_by_type(obs, units.Terran.CommandCenter)
    enemy_supply_depots = self.get_enemy_units_by_type(obs, units.Terran.SupplyDepot)
    enemy_completed_supply_depots = self.get_enemy_completed_units_by_type(obs, units.Terran.SupplyDepot)
    enemy_barrackses = self.get_enemy_units_by_type(obs, units.Terran.Barracks)
    enemy_completed_barrackses = self.get_enemy_completed_units_by_type(obs, units.Terran.Barracks)
    enemy_marines = self.get_enemy_units_by_type(obs, units.Terran.Marine)
    
    # Return tuple 
    return (len(command_centers),
            len(scvs),
            len(idle_scvs),
            len(supply_depots),
            len(completed_supply_depots),
            len(barrackses),
            len(completed_barrackses),
            len(marines),
            queued_marines,
            free_supply,
            can_afford_supply_depot,
            can_afford_barracks,
            can_afford_marine,
            len(enemy_command_centers),
            len(enemy_scvs),
            len(enemy_idle_scvs),
            len(enemy_supply_depots),
            len(enemy_completed_supply_depots),
            len(enemy_barrackses),
            len(enemy_completed_barrackses),
            len(enemy_marines))

  # Gets the current state of the game, feeds the state into the QLearningTable and the QLearningTable chooses an action
  def step(self, obs):
    super(SmartAgent, self).step(obs)
    state = str(self.get_state(obs))
    action = self.qtable.choose_action(state)
    if self.previous_action is not None:
      self.qtable.learn(self.previous_state, self.previous_action, obs.reward, 'terminal' if obs.last() else state)
    self.previous_state = state
    self.previous_action = action
    return getattr(self, action)(obs)

# Run the game, create agents, set up instructions for the game
def main(unused_argv):
  agent1 = SmartAgent()
  agent2 = RandomAgent()
  try:
    with sc2_env.SC2Env(
        map_name="Simple64", # Choose the map
        players=[sc2_env.Agent(sc2_env.Race.terran), 
                 sc2_env.Agent(sc2_env.Race.terran)],
        agent_interface_format=features.AgentInterfaceFormat(
            action_space=actions.ActionSpace.RAW,
            use_raw_units=True,
            raw_resolution=64,
        ),
        step_mul=128, # How fast it runs the game
        disable_fog=True, # Too see everything in the minimap
    ) as env:
      run_loop.run_loop([agent1, agent2], env, max_episodes=1000) # Control both agents instead of one
  except KeyboardInterrupt:
    pass

if __name__ == "__main__":
  app.run(main)