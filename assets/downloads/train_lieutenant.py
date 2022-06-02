from agents.agent import WeightedHybridAgent
from env.joadia import Joadia
from agents.random import RandomLegal, RandomPolicy
from agents.heuristic import GeneralHeuristic, RedHeuristic
from agents.rl_cs import RLAgent, RLAgentCallback
from env.observation import Observation
from env.player import Player
from env.simple_log import SimpleLog
from gym import spaces
import numpy as np
from tqdm import tqdm
import random
import time

from pretrain_fun import ExpertDataSet
from torch.utils.data.dataset import random_split

from pretrain_fun import ExpertDataSet, pretrain_agent

from stable_baselines3.common.evaluation import evaluate_policy

from torch.optim.lr_scheduler import StepLR

if __name__ == "__main__":

    tic = time.time()
    verbose = False
    SimpleLog.setup(dir_name="results/training")

    red_agent = WeightedHybridAgent(agents=[RedHeuristic(), RandomLegal()], weights=[0.5, 0.5])
    Joadia.INCLUDE_RED = True
    Joadia.ENV_RED_AGENT = red_agent

    train_player = Player.BLUE


    ################################################################################
    ##### MAIN RUN PARAMS
    ################################################################################
    step_multiplier = 1/100 #first we gotta go fast. 1/1000 still shows some improvement from the PPO reoutine
    learn_timesteps = int(10000000*step_multiplier)  
    log_timesteps = int(100000*step_multiplier)      
    
    num_repeats = int(10000*step_multiplier)       


    ################################################################################
    ###### MAIN AGENT SETUP
    ################################################################################
    RLAgent.USE_SIMPLIFIED_ENV = False  
    blue_agent = RLAgent(agent_type="PPO")  
    blue_agent.learn_timesteps = learn_timesteps
    callback = RLAgentCallback(log_timesteps=log_timesteps, log_dir=SimpleLog.dir_name)

    SimpleLog.log("verbose={}\nblue_agent={}\nred_agent={}\ntraining={}\nlearn_timesteps={}\nlog_timesteps={}\nnum_repeats={}\n".format(
        verbose, blue_agent, red_agent, Player.get_str(train_player), learn_timesteps, log_timesteps, num_repeats))

    
    ################################################################################
    ###### DEFINE EXPERT AGENT AND GENERATE INTERACTION SET
    ################################################################################

    expert_agent = GeneralHeuristic()
    expert_agent.force_deterministic = True

    environ = Joadia.get_joadia_environment(gym_env=True)
    
    num_interactions = int(1E7) # This is a goal state, we'll run the number of games required to get to this
  

    regenerate_observations = False
    if regenerate_observations:
      expert_observations = np.empty((num_interactions,) + environ.observation_space.shape)
      expert_actions = np.empty((num_interactions,) + environ.action_space.shape)

      i=0
      num_games = 0
      with tqdm(total=num_interactions) as progress:
      
        while i < num_interactions:

          
          environ.reset()
          done = environ.is_done()

          while (not done) and (i < num_interactions):
              obs = environ.get_observation()
              struct_obs = environ.get_structured_observation(obs)

              current_player = obs[Observation.CURRENT_PLAYER]
              if current_player == Player.RED and environ.INCLUDE_RED:
                  action_id = red_agent.act(obs)
              elif current_player == Player.BLUE:
                  action_id = expert_agent.act(obs)
                  if struct_obs.current_turn > 0:
                    expert_observations[i] = obs
                    expert_actions[i] = action_id
                    i+=1
                    progress.update(i)
              if verbose:
                  SimpleLog.log(environ.describe_action(action_id), include_print=False)

              _, reward, done, info = environ.step(action_id)
          num_games+=1


        np.savez_compressed(
            "expert_data",
            expert_actions=expert_actions,
            expert_observations=expert_observations,
        )


    else: 
      expert_data = np.load("expert_data.npz")
      expert_observations = expert_data["expert_observations"]
      expert_actions = expert_data["expert_actions"]
    
    expert_dataset = ExpertDataSet(expert_observations, expert_actions)

    #Generate train/test split
    train_size = int(0.8 * len(expert_dataset))
    test_size = len(expert_dataset) - train_size
    train_expert_dataset, test_expert_dataset = random_split(
        expert_dataset, [train_size, test_size]
    )
    


    ################################################################################
    ###### PRE-TRAINING AGENT TO IMITATE EXPERT 
    ################################################################################


    blue_agent.policy = \
      pretrain_agent(
        blue_agent.model,
        environ,
        train_expert_dataset,
        test_expert_dataset,
        epochs=100,
        scheduler_gamma=0.7,
        learning_rate=0.1,
        log_interval=10000,
        no_cuda=True,
        seed=1,
        batch_size=64,
        test_batch_size=1000,
      )   


    blue_agent.save(name="results/training/pre_trained_student")
    blue_agent.save(name="{}/{}-{}_steps".format(SimpleLog.dir_name, blue_agent.agent_type, learn_timesteps))

    
