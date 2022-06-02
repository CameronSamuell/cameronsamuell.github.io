from calendar import c
from env.joadia import Joadia
from agents.agent import WeightedHybridAgent
from agents.random import RandomLegal, RandomPolicy
from agents.heuristic import GeneralHeuristic, RedHeuristic
from agents.rl import RLAgent
from env.observation import Observation
from env.player import Player
import numpy as np
import copy

import matplotlib 
import matplotlib.pyplot as plt
from numpy import genfromtxt

## viz setup
plt.close('all')
cmap_terrain = matplotlib.colors.ListedColormap(['cornflowerblue', 'lime','lime', 'darkgreen', 'green', 'limegreen', 'orangered'])


############################
########## Run game ########
############################

blue_agent = GeneralHeuristic()
blue_agent_name = 'General Heuristic'

#blue_agent = RandomLegal()
#blue_agent_name = 'Random Legal'

#blue_agent = RLAgent(model_name="models/blue/example_blue_model.zip")
#blue_agent_name = 'Example Blue Model'

red_agent = WeightedHybridAgent(agents=[RedHeuristic(), RandomLegal()], weights=[0.5, 0.5])
Joadia.INCLUDE_RED = True
verbose = True

test_env_obs_length = len(Joadia.get_joadia_environment().get_observation())
agent_env_obs_length = len(RLAgent().env.get_observation())

env = Joadia.get_joadia_environment()

turns = []; rescues= []; deaths = []; score = []
status = {"id":[], "healthy":[], "injured":[], "supply":[]}
status_blue = {"id":[], "healthy":[], "injured":[], "supply":[]}

env.reset()
done = env.is_done()

while not done:
  # play one game to completion

  obs = env.get_observation()

  current_player = obs[Observation.CURRENT_PLAYER]
  if current_player == Player.RED and env.INCLUDE_RED:
      action_id = red_agent.act(obs)
  elif current_player == Player.BLUE:
      action_id = blue_agent.act(obs)

  _, reward, done, info = env.step(action_id)
  # Get 2D-resolved turn status

  struct_obs = env.get_structured_observation(obs)
  if (struct_obs.current_phase == 0) and (struct_obs.current_unit_id == 9):
    status_single = struct_obs.territories
    status_blue['id'].append([status_single[territory].get_territory_id() for territory in status_single])
    status_blue['healthy'].append([status_single[territory].get_healthy() for territory in status_single])
    status_blue['injured'].append([status_single[territory].get_injured() for territory in status_single])
    status_blue['supply'].append([status_single[territory].get_supply() for territory in status_single])

  env_copy = copy.copy(env)
  obs_copy = env_copy.get_observation(player=Player.WHITE)
  struct_obs = env_copy.get_structured_observation(obs_copy)
  if (struct_obs.current_phase == 0) and (struct_obs.current_unit_id == 9):
    status_single = struct_obs.territories
    status['id'].append([status_single[territory].get_territory_id() for territory in status_single])
    status['healthy'].append([status_single[territory].get_healthy() for territory in status_single])
    status['injured'].append([status_single[territory].get_injured() for territory in status_single])
    status['supply'].append([status_single[territory].get_supply() for territory in status_single])


  #Get 1D status
  if env.current_unit_id == 0:
    turns.append(env.current_turn)
    rescues.append(env.rescues)
    deaths.append(env.deaths)
    score.append(env.get_score())

for key in status:
  status[key] = np.array(status[key])


############################
###### Generate Plots ######
############################

# remap territories to 6x8 array
territory_id = np.full((6,9), np.nan)
terrain = np.full((6,9), 0)
healthy = np.full((6,9),np.nan)
injured = np.full((6,9),np.nan)
supply = np.full((6,9),np.nan)
healthy_blue = np.full((6,9),np.nan)
injured_blue = np.full((6,9),np.nan)
supply_blue = np.full((6,9),np.nan)


territory_mapping ={
  1:(0,0), 2:(0,3), 3:(1,1), 4:(1,2), 5:(1,4), 6:(1,5),  7:(1,6), 8:(1,7), 9:(2,1),
  10:(2,4), 11:(2,5), 12:(3,1), 13:(3,2), 14:(3,4), 15:(4,4),  16:(4,5), 17:(4,6), 18:(5,0),
  19:(5,2), 20:(5,3), 21:(5,5), 22:(5,6), 23:(5,7), 24:(0,4),  25:(0,6), 26:(2,0), 27:(3,5),
  28:(3,6), 29:(4,7), 0:(0,8)
}


for turn in range(7):
#for turn in [6]:
  for i, territory in enumerate(status['id'][turn]):

    territory_id[territory_mapping[territory]] = status['id'][turn][i]
    healthy[territory_mapping[territory]] = status['healthy'][turn][i]
    injured[territory_mapping[territory]] = status['injured'][turn][i]
    supply[territory_mapping[territory]] = status['supply'][turn][i]
    healthy_blue[territory_mapping[territory]] = status_blue['healthy'][turn][i]
    injured_blue[territory_mapping[territory]] = status_blue['injured'][turn][i]
    supply_blue[territory_mapping[territory]] = status_blue['supply'][turn][i]


  healthy[healthy<0]=np.nan
  injured[injured<0]=np.nan
  supply[supply<0]=np.nan
  healthy_blue[healthy_blue<0]=np.nan
  injured_blue[injured_blue<0]=np.nan
  supply_blue[supply_blue<0]=np.nan

  for k, v in env.PRIMARY_TERRITORIES.items():
    terrain[territory_id == k] = v

  plt.figure()

  plt.subplot(4,4,1)
  #plt.imshow(territory_id < 24, cmap = cmap_terrain)
  plt.imshow(terrain, cmap = cmap_terrain)
  plt.axis('on')
  plt.gca().set_xticks([])
  plt.gca().set_yticks([])
  plt.gca().set_xticklabels([])
  plt.gca().set_yticklabels([])
  plt.title('terrain')

  plt.subplot(4,4,2)
  plt.imshow(healthy)
  plt.clim([0,6])
  plt.gca().set_xticks([])
  plt.gca().set_yticks([])
  plt.gca().set_xticklabels([])
  plt.gca().set_yticklabels([])
  plt.title('healthy',weight='normal')
  plt.ylabel('full view',weight='normal')

  plt.subplot(4,4,3)
  plt.imshow(injured)
  plt.clim([0,6])
  plt.gca().set_xticks([])
  plt.gca().set_yticks([])
  plt.gca().set_xticklabels([])
  plt.gca().set_yticklabels([])
  plt.title('injured',weight='normal')

  plt.subplot(4,4,4)
  plt.imshow(supply)
  plt.clim([0,30])
  plt.gca().set_xticks([])
  plt.gca().set_yticks([])
  plt.gca().set_xticklabels([])
  plt.gca().set_yticklabels([])
  plt.title('supply',weight='normal')


  plt.subplot(4,4,6)
  plt.imshow(healthy_blue)
  plt.clim([0,6])
  plt.gca().set_xticks([])
  plt.gca().set_yticks([])
  plt.gca().set_xticklabels([])
  plt.gca().set_yticklabels([])
  #plt.title('healthy')
  plt.ylabel('blue view',weight='normal')

  plt.subplot(4,4,7)
  plt.imshow(injured_blue)
  plt.clim([0,6])
  plt.gca().set_xticks([])
  plt.gca().set_yticks([])
  plt.gca().set_xticklabels([])
  plt.gca().set_yticklabels([])
  #plt.title('injured')

  plt.subplot(4,4,8)
  plt.imshow(supply_blue)
  plt.clim([0,30])
  plt.gca().set_xticks([])
  plt.gca().set_yticks([])
  plt.gca().set_xticklabels([])
  plt.gca().set_yticklabels([])
  #plt.title('supply')



  plt.subplot(2,2,3)
  plt.plot(turns[:turn+1],rescues[:turn+1], '-o')
  plt.xlim([0,8])
  plt.ylim([0,100])
  plt.title('rescues',weight='normal')
  plt.xlabel('turn',weight='normal')


  plt.subplot(2,2,4)
  plt.plot(turns[:turn+1],deaths[:turn+1], '-o')
  plt.xlim([0,8])
  plt.ylim([0,100])
  plt.title('deaths',weight='normal')
  plt.xlabel('turn',weight='normal')

  
  plt.suptitle(blue_agent_name, weight='bold')
  plt.tight_layout()
  plt.savefig('./results/single_run/'+blue_agent_name.replace(" ", "_").lower()+f'/{turn}.png',dpi=150)
  #plt.show(block=False)
  

