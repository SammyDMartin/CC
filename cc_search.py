from gym.spaces import Discrete
from ray.rllib.agents.ppo import PPOTorchPolicy, PPOTrainer
from ray.rllib.agents.dqn import DQNTorchPolicy,DQNTrainer
from ray.rllib.agents.a3c import A3CTrainer
from ray.rllib.agents.a3c.a3c_torch_policy import A3CTorchPolicy
from ray import tune
from cc_tools import add_support_checkpoint_composition
from cc_tools import logger,LOAD_FROM_CONFIG_KEY
import os
from tqdm import tqdm
from cc_tools import IteratedPrisonersDilemma, IteratedChicken,IteratedStagHunt, IteratedBoS, MutalismCoordination, PureCoordination,IteratedAsymBoS
import pandas as pd
import numpy as np

##############################################
##Main test Functions
####################################

def make_selfplay_checkpoint(environment,RL_alg, steps=10):
  policy,trainer = RL_alg

  C_original = {'env': environment,
    'env_config': {'get_additional_info': True,
      'max_steps': 100,
      'players_ids': ['player_row', 'player_col']},
    'framework': 'torch',
    'lr': 0.0001,
    'multiagent': {'policies': {'player_col': (policy,
        Discrete(5),
        Discrete(2),
        {}),
      'player_row': (policy,
        Discrete(5),
        Discrete(2),
        {})},
      'policies_to_train': ['player_row','player_col'],
      'policy_mapping_fn': lambda agent_id : agent_id,
    },
    'model': {
      "fcnet_hiddens": [64],
      "fcnet_activation": "relu",
    },
    "framework": "torch",
    "num_workers": 0,
  }

  if RL_alg == (PPOTorchPolicy,PPOTrainer):
    C_original.update({
      "train_batch_size": 128,
      "sgd_minibatch_size": 128,
      "shuffle_sequences": True,
      "num_sgd_iter": 10,
    })
  elif RL_alg == (DQNTorchPolicy,DQNTrainer):
    C_original.update({
      "learning_starts": 0,
      "timesteps_per_iteration": 100,
      "target_network_update_freq": 100,
      "hiddens": [48],
    })
  elif RL_alg == (A3CTorchPolicy,A3CTrainer):
    C_original.update({
      "num_workers": 2,
    })
  else:
    #raise NotImplementedError
    print("No Params!")

  stop_config = {"training_iteration":steps}
  
  tune_analysis = tune.run(
    trainer,
    config=C_original,
    stop=stop_config,
    checkpoint_at_end=True,
    name="test", 
    verbose=0,
    )
  
  checkpoint = tune_analysis.get_last_checkpoint(list(tune_analysis.trial_dataframes.keys())[0])
  tune_analysis.get_all_configs()

  return checkpoint, tune_analysis

def get_rllib_config_eval(seeds, 
                          steps, 
                          game, 
                          debug=False, 
                          stop_iters=200, 
                          policy_classes=None,
                          checkpoint_paths=[]):
    stop_config = {
        "training_iteration": 2 if debug else stop_iters,
    }

    # add option to compose policies from checkpoints
    
    if checkpoint_paths == []:
      modified_policy_classes = policy_classes
      checkpoint_paths = [None,None]
    elif len(checkpoint_paths) == 1:
      modified_policy_classes = [add_support_checkpoint_composition(policy_classes[0]),None]
      checkpoint_paths.append(None)
    else:   
      modified_policy_classes = [add_support_checkpoint_composition(policy_class) for policy_class in policy_classes]

    env_config = {
        "players_ids": ["player_row", "player_col"],
        "max_steps": steps,
        "get_additional_info": True,
    }

    rllib_config = {
        "env": game,
        "env_config": env_config,
        "multiagent": {
            # add a fake policy name so that the list is not empty
            "policies_to_train": ["None"],

            "policies": {
                env_config["players_ids"][0]: (
                    modified_policy_classes[0], 
                    game.OBSERVATION_SPACE,
                    game.ACTION_SPACE, 
                    # Added the info to specify which checkpoint to load from and which policy to load inside it
                    {
                        LOAD_FROM_CONFIG_KEY:(checkpoint_paths[0], env_config["players_ids"][0]),         
                    }
                ),
                env_config["players_ids"][1]: (
                    modified_policy_classes[1], 
                    game.OBSERVATION_SPACE,
                    game.ACTION_SPACE, 
                    # Added the info to specify which checkpoint to load from and which policy to load inside 
                    {
                        LOAD_FROM_CONFIG_KEY:(checkpoint_paths[1], env_config["players_ids"][1])
                    }
                ),
            },
            "policy_mapping_fn": lambda agent_id, **kwargs: agent_id,
        },
        "seed": tune.grid_search(seeds),
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "framework": 'torch',
        'model': {
          # Number of hidden layers for fully connected net
          "fcnet_hiddens": [64],
          # Nonlinearity for fully connected net (tanh, relu)
          "fcnet_activation": "relu",
        },
        "num_workers": 0,
        "explore": False,
        "exploration_config": {
          "type": "SoftQ",
          # Parameters for the Exploration class' constructor:
          "temperature": 1.0,
        },
    }


    # We need to make the networks of the eval equal to those used for the training 
    for i, policy_class in enumerate(policy_classes):
      policy_id = env_config["players_ids"][i]

      if policy_class == DQNTorchPolicy:
        rllib_config["multiagent"]["policies"][policy_id][3].update({                     
            "hiddens": [48],
            "model":{"vf_share_layers":True},
        })
      elif policy_class == PPOTorchPolicy:
          rllib_config["multiagent"]["policies"][policy_id][3].update({    
            "model":{"vf_share_layers":False},
            "use_critic": True,
            "use_gae": True,                 
          }) 

    return rllib_config, stop_config
  
def assess_agent_vs_agent(policies=[],checkpoints=[],environment=None,iters=None):
  seeds = [1]

  rllib_config, stop_config = get_rllib_config_eval(
      seeds,
      iters,
      game=environment,
      debug=False,
      stop_iters=iters,
      policy_classes=policies,
      checkpoint_paths=checkpoints
  )

  tune_analysis = tune.run(
      # Works with any Trainer but then using PGTrainer is more safe (simple training loop)... but PGTrainer doesn't work here for some reasons...
      DQNTrainer,
      # PPOTrainer,
      config=rllib_config,
      stop=stop_config,
      checkpoint_at_end=True, verbose = 0
  )

  return tune_analysis

def assess_agent_vs_others(agent_trainer,opponents=[],strategies=[],steps=10):

  full_outputs={}

  games = ['IteratedPrisonersDilemma','IteratedChicken','IteratedStagHunt','IteratedBoS','MutalismCoordination','PureCoordination','IteratedAsymBos']
  games_actual = [IteratedPrisonersDilemma, IteratedChicken,IteratedStagHunt, IteratedBoS, MutalismCoordination, PureCoordination,IteratedAsymBoS]

  other_policies = strategies+opponents
  
  tests = (len(other_policies)*len(games)) + ((1+len(opponents))*len(games))

  pbar = tqdm(total=tests)

  result = {}

  train_steps = int(steps/2)
  test_steps = int(steps)

  for game_num in range(len(games)):
    game_outputs = {}

    name = games[game_num]
    game = games_actual[game_num]

    if agent_trainer[1] == None:
      check = None
      train_res = "no_train"
      cps = []
    else:
      check,train_res = make_selfplay_checkpoint(game,agent_trainer,train_steps)
      cps = [check]
    pbar.update(1)

    game_outputs["train"] = train_res

   
    for itr, other_policy in enumerate(other_policies):
      if agent_trainer[0] in strategies:
        #strategy is the main player
        strat_func = agent_trainer[0]
        print(strat_func,game)
        agent = strat_func(game,0)
      else:
        agent = agent_trainer[0]

      if other_policy in opponents:
        #other_policy is an RL agent
        opp_agent_trainer = other_policy
        check_opp,opp_train = make_selfplay_checkpoint(game,opp_agent_trainer,train_steps)
        pbar.update(1)
        other_policy = other_policy[0]
        opponent_name = str(other_policy.__name__)
        cps = [check,check_opp]

        train_name = "train" + opponent_name
        game_outputs[train_name] = opp_train


      elif other_policy in strategies:
        #other_policy is a non-RL strat
        try:
          opponent_name = str(other_policy.__name__)+str(other_policy(game,1).__name__)
        except Exception:
          print("Cannot use policy, {}".format(other_policy.__name__))
        other_policy = other_policy(game,1)

      pols = [agent,other_policy]

      names = lambda lst : [str(x.__name__) for x in lst]

      print("Game:", game.__name__)
      print("\n\nPolicies:", names(pols))
      print("Checkpoints:", cps)

      output = assess_agent_vs_agent(policies=pols, checkpoints = cps,environment=game,iters=test_steps)

      output_df = output.trial_dataframes[list(output.trial_dataframes.keys())[0]]

      
      game_outputs[opponent_name] = output_df

      pbar.update(1)

      epslen = output_df['episode_len_mean'][0]
      #number of timesteps per episode

      rewards = output_df['policy_reward_mean/player_row'].to_numpy()
      rewards = rewards/epslen
      if itr == 0:
        total = rewards
      else:
        total += rewards

    result[name] = rewards/float(len(other_policies))
    full_outputs[name] = game_outputs

  row_player_mean_reward_per_timestep = pd.DataFrame.from_dict(result)


  return row_player_mean_reward_per_timestep,full_outputs
