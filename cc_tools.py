from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pickle5

import gym
import numpy as np
# Print the versions of the libraries that we are using:

import ray
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.rllib.examples.env import matrix_sequential_social_dilemma
from tqdm import tqdm, trange
from ray.tune.logger import pretty_print
print("ray version:", ray.__version__)

import random

from ray.rllib.examples.env.matrix_sequential_social_dilemma import \
    IteratedPrisonersDilemma, IteratedChicken, \
    IteratedStagHunt, IteratedBoS
from ray.rllib.examples.env.matrix_sequential_social_dilemma import MatrixSequentialSocialDilemma
from ray.rllib.examples.env.utils.mixins import TwoPlayersTwoActionsInfoMixin
from gym.spaces import Discrete

from ray.rllib.examples.env.matrix_sequential_social_dilemma import \
    IteratedPrisonersDilemma, IteratedChicken, \
    IteratedStagHunt, IteratedBoS, IteratedAsymBoS

import ray
import pandas as pd
from ray import tune
import os
from ray.rllib.agents.dqn import DQNTorchPolicy
#from ray.rllib.agents.a3c import A3CTorchPolicy
from ray.rllib.agents.ppo import PPOTorchPolicy
from ray.rllib.agents.pg import PGTrainer, PGTorchPolicy
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.policy.policy import Policy

import nashpy as nash

from ray.rllib.agents.pg import PGTrainer
from ray.rllib.agents.dqn import DQNTrainer, DEFAULT_CONFIG
from ray.rllib.agents.a3c import A3CTrainer
from ray.rllib.contrib.maddpg import MADDPGTrainer

import ray
import pandas as pd
from ray import tune
import os
from ray.rllib.agents.dqn import DQNTorchPolicy
from ray.rllib.agents.a3c.a3c_torch_policy import A3CTorchPolicy
from ray.rllib.agents.a3c import A3CTrainer
from ray.rllib.agents.ppo import PPOTorchPolicy
from ray.rllib.agents.pg import PGTrainer, PGTorchPolicy
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.policy.policy import Policy

from ray.rllib.agents.pg import PGTrainer
from ray.rllib.agents.dqn import DQNTrainer, DEFAULT_CONFIG
from ray.rllib.contrib.maddpg import MADDPGTrainer
from ray.rllib.examples.env.matrix_sequential_social_dilemma import MatrixSequentialSocialDilemma
from ray.rllib.examples.env.utils.mixins import TwoPlayersTwoActionsInfoMixin
from gym.spaces import Discrete
from ray.rllib.examples.env.matrix_sequential_social_dilemma import \
    IteratedPrisonersDilemma, IteratedChicken, \
    IteratedStagHunt, IteratedBoS, IteratedAsymBoS

# From https://github.com/longtermrisk/marltoolbox/blob/anonymization/marltoolbox/utils/restore.py
# The marltoolbox uses an 'old' version of RLLib thus for speed I copied the code here

import logging
import os
import pickle
from typing import List
from ray.tune.analysis import ExperimentAnalysis






#######################################################





## Specify New Learning Games


class PureCoordination(TwoPlayersTwoActionsInfoMixin,
                              MatrixSequentialSocialDilemma):
    """
    A two-agent environment for the pure coordination game
    """
    NUM_AGENTS = 2
    NUM_ACTIONS = 2
    NUM_STATES = NUM_ACTIONS**NUM_AGENTS + 1
    ACTION_SPACE = Discrete(NUM_ACTIONS)
    OBSERVATION_SPACE = Discrete(NUM_STATES)
    PAYOUT_MATRIX = np.array([[[+1, +1], [0, 0]], [[0, 0], [+1, +1]]])
    NAME = "IPC"

class MutalismCoordination(TwoPlayersTwoActionsInfoMixin,
                              MatrixSequentialSocialDilemma):
    """
    A two-agent environment for the byproduct mutualism game
    """
    NUM_AGENTS = 2
    NUM_ACTIONS = 2
    NUM_STATES = NUM_ACTIONS**NUM_AGENTS + 1
    ACTION_SPACE = Discrete(NUM_ACTIONS)
    OBSERVATION_SPACE = Discrete(NUM_STATES)
    PAYOUT_MATRIX = np.array([[[+1, +1], [0, 0]], [[0, 0], [-1, -1]]])
    NAME = "IMC"
    
####################################


#games = [IteratedPrisonersDilemma, IteratedChicken,IteratedStagHunt, IteratedBoS, MutalismCoordination, PureCoordination,IteratedAsymBoS]
#games_names = ['IteratedPrisonersDilemma','IteratedChicken','IteratedStagHunt','IteratedBoS','MutalismCoordination','PureCoordination','IteratedAsymBos']

def nash_from_RLLIB(game):
  """Returns nashpy game from input RLLIB env array

  Args:
      game ([type]): [description]

  Returns:
      [type]: [description]
  """
  matrix_row = [[i[0] for i in item] for item in game.PAYOUT_MATRIX]
  matrix_col = [[i[1] for i in item] for item in game.PAYOUT_MATRIX]
  game = nash.Game(matrix_row,matrix_col)
  return game

def games_test(games):
  for idx,g in enumerate(games):
    game = nash_from_RLLIB(g)

    print(games[idx].__name__)
    print(game)

def calculate_cc_aid(calculator,game_list,names,epslen,player=None):
  """Function to calculate ideal/maximin outcome of games for a CC demo

  Args:
      calculator ([type]): [description]
      game_list ([type]): [description]
      names ([type]): [description]
      epslen ([type]): [description]
      player ([type], optional): [description]. Defaults to None.

  Returns:
      [type]: [description]
  """
  eps_utils = []
  for idx,g in enumerate(game_list):
    row_mm,col_mm = calculator(g)
    game = nash_from_RLLIB(g)
    if player == 'row':
      u = np.sum(game[row_mm,col_mm][0])*epslen
    elif player == 'col':
      u = np.sum(game[row_mm,col_mm][1])*epslen
    else:
      u = np.sum(game[row_mm,col_mm])*epslen
    col = [u for count in range(int(epslen))]
    eps_utils.append(col)
  return np.array(eps_utils).T

def eps_utils(game_list,names,player=None):
  """Function for CC demo

  Args:
      game_list ([type]): [description]
      names ([type]): [description]
      player ([type], optional): [description]. Defaults to None.

  Returns:
      [type]: [description]
  """
  summary = {}
  sum_names = ['maximin','maximax','ideal_selfplay']
  for idn,calculator in enumerate([maximin,maximax,ideal_selfplay]):
    eps_utils = []
    for idx,g in enumerate(game_list):
      row_mm,col_mm = calculator(g)
      game = nash_from_RLLIB(g)
      if player == 'row':
        u = np.sum(game[row_mm,col_mm][0])*epslen
      elif player == 'col':
        u = np.sum(game[row_mm,col_mm][1])*epslen
      else:
        u = np.sum(game[row_mm,col_mm])*epslen
      eps_utils.append(u)
    cal_name = sum_names[idn]

    summary[cal_name] = eps_utils
    
  return pd.DataFrame(summary,index=names).T

#eps_utils(games,games_names)

class coop_equilibrium():
  """Cooperative equilibrium in iterated prisoners dilemma calculation

  Returns:
      [type]: [description]
  """
  #currently only handles pure strategies, i.e. when to switch from cooperate to defect on prisoners dilemma, will need to look at some more
  #https://docs.google.com/document/d/13hfSs_2mWwbhZ5ypcTb-JnZ6EOauuBEBZPPxf6X317E/edit#heading=h.wo12rmt74pda 
  def __init__(self,val_cc,val_cd,val_dd,nash_act):
    self.t = 1
    self.k = 0
    self.val_cc = val_cc
    self.val_cd = val_cd
    self.val_dd = val_dd
    self.u_table = [0,0]
    self.nash_act = [float(x) for x in nash_act]
    self.threshold_payoff = 0
  def tau_cc(self):
    return (self.k+1)/(self.t+1)
  def tau_cd(self):
    return 1-self.tau_cc()
  def tau_dd(self):
    return 1
  def val_cooperate(self):
    return self.val_cc*self.tau_cc() + self.val_cd*self.tau_cd()
  def val_defect(self):
    return self.val_dd*self.tau_dd()
  def update_table(self):
    #self.step(obs)
    self.u_table[0]=float(self.val_cooperate())
    self.u_table[1]=float(self.val_defect())
    self.threshold_payoff = max(self.u_table)
  def step(self,obs):
    self.t += 1
    self.k += obs
    self.update_table()
  def act_vector(self):
    res = [float(x) == self.threshold_payoff for x in self.u_table]
    return [x/sum(res) for x in res]
  def action(self):
    if self.act_vector() == [0.5,0.5]:
      return self.nash_act
    else: 
      return self.act_vector


def maximin(game):
  """Calculates which of always 1, always 2 or 1/2 50/50 is closest to maximin

  Args:
      game ([type]): [description]

  Returns:
      [type]: [description]
  """
  ngame = nash_from_RLLIB(game)

  always_C = np.array([1,0])
  always_D = np.array([0,1])

  pure_strats = [always_C,always_D]

  #print(ngame)

  

  utils_1 = [[],[]]
  utils_2 = [[],[]]

  for ids1,s1 in enumerate(pure_strats):
    for ids2,s2 in enumerate(pure_strats):
      utils = ngame[s1,s2]
      if ids1 == 0:
        utils_1[0].append(utils[0])
      elif ids1 == 1:
        utils_1[1].append(utils[0])
      if ids2 == 0:
        utils_2[0].append(utils[1])
      elif ids2 == 1:
        utils_2[1].append(utils[1])

  #print("Row player always action 1:{}, action 2:{}".format(utils_1[0],utils_1[1]))
  #print("Col player always action 1:{}, action 2:{}".format(utils_2[0],utils_2[1]))

  min_vals = lambda util : [min(outcome) for outcome in util]
  maximin1,maximin2 = np.array([int(m1 == max(min_vals(utils_1))) for m1 in min_vals(utils_1)]),np.array([int(m2 == max(min_vals(utils_2))) for m2 in min_vals(utils_2)])

  return maximin1/np.sum(maximin1), maximin2/np.sum(maximin2)

def maximax(game):
    #print(game)
    ngame = nash_from_RLLIB(game)

    always_C = np.array([1,0])
    always_D = np.array([0,1])

    pure_strats = [always_C,always_D]

    
    utils_1 = [[],[]]
    utils_2 = [[],[]]

    #print(ngame)

    for ids1,s1 in enumerate(pure_strats):
      for ids2,s2 in enumerate(pure_strats):
        utils = ngame[s1,s2]
        if ids1 == 0:
          utils_1[0].append(utils[0])
        elif ids1 == 1:
          utils_1[1].append(utils[0])
        if ids2 == 0:
          utils_2[0].append(utils[1])
        elif ids2 == 1:
          utils_2[1].append(utils[1])

    #print("Row player always action 1:{}, action 2:{}".format(utils_1[0],utils_1[1]))
    #print("Col player always action 1:{}, action 2:{}".format(utils_2[0],utils_2[1]))

    max_vals = lambda util : [max(outcome) for outcome in util]

    mmax1 = max_vals(utils_1)
    mmax2 = max_vals(utils_2)

    #print(mmax1,mmax2)
    
    mmax1 = np.array([int(m1 == max(mmax1)) for m1 in mmax1])
    
    mmax2 = np.array([int(m2 == max(mmax2)) for m2 in mmax2])
    
    #print(mmax1,mmax2)

    return mmax1/np.sum(mmax1), mmax2/np.sum(mmax2)

def ideal_selfplay(game):
  """"Ideal agent behaviour"

  Args:
      game ([type]): [description]

  Returns:
      [type]: [description]
  """
  if game == IteratedPrisonersDilemma:
    return np.array([1,0]), np.array([1,0])
  elif game == IteratedChicken:
    return np.array([1,0]), np.array([1,0])
  elif game == IteratedStagHunt:
    return np.array([1,0]), np.array([1,0])
  elif game == IteratedBoS:
    return np.array([1,0]), np.array([1,0])
  elif game == MutalismCoordination:
    return np.array([1,0]), np.array([1,0])
  elif game == PureCoordination:
    return np.array([0,1]), np.array([0,1])
  elif game == IteratedAsymBoS:
    return np.array([0,1]), np.array([0,1]) 
    
    
##############################################


class RandPolicy(RandomPolicy):
  def update_target(self):
    pass
  def get_weights(self):
    pass


class CoopPolicy(RandomPolicy):
  def update_target(self):
    pass
  def get_weights(self):
    pass
  def compute_actions(self,
                      obs_batch,
                      state_batches=None,
                      prev_action_batch=None,
                      prev_reward_batch=None,
                      **kwargs):
      # Alternatively, a numpy array would work here as well.
      # e.g.: np.array([random.choice([0, 1])] * len(obs_batch))
    act_zero_one = 0
    return np.array([act_zero_one] * len(obs_batch)), \
            [], {}

class DefectPolicy(CoopPolicy):
  def update_target(self):
    pass
  def get_weights(self):
    pass
  def compute_actions(self,
                      obs_batch,
                      state_batches=None,
                      prev_action_batch=None,
                      prev_reward_batch=None,
                      **kwargs):
      # Alternatively, a numpy array would work here as well.
      # e.g.: np.array([random.choice([0, 1])] * len(obs_batch))
    act_zero_one = 1
    #len(obs_batch)
    return np.array([act_zero_one] * len(obs_batch)), \
            [], {}

class TFTAverage(RandomPolicy):
    """Play the move that would beat the last move of the opponent."""

    def update_target(self):
      pass
    def get_weights(self):
      pass
    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        **kwargs):
      input = obs_batch[0]

      mean_act_his = np.mean(input)
      if mean_act_his > 0.5:
        output = 1
      else:
        output = 0


      return np.array([output]), \
              [], {}



def maximin_policy(game, player_num):
  vector = maximin(game)[player_num]
  if vector[0] == 1:
    return CoopPolicy
  if vector[1] == 1:
    return DefectPolicy
  else:
    return RandomPolicy

def ideal_selfplay_policy(game, player_num):
  vector = ideal_selfplay(game)[player_num]
  if vector[0] == 1:
    return CoopPolicy
  if vector[1] == 1:
    return DefectPolicy
  else:
    return RandomPolicy

random_agent = lambda x,y: RandPolicy



################################


logger = logging.getLogger(__name__)

LOAD_FROM_CONFIG_KEY = "checkpoint_to_load_from"

def before_loss_init_load_policy_checkpoint(
    policy, observation_space=None, action_space=None, trainer_config=None
):
    """
    This function is to be given to a policy template(a policy factory)
    (to the 'after_init' argument).
    It will load a specific policy state from a given checkpoint
    (instead of all policies like what does the restore option provided by
    RLLib).

    The policy config must contain the tuple (checkpoint_path, policy_id) to
    load from, stored under the LOAD_FROM_CONFIG_KEY key.

    Finally, the checkpoint_path can be callable, in this case it must
    return a path (str) and accept the policy config as the only argument.
    This last feature allows to dynamically select checkpoints
    for example in multistage training or experiments
    Example: determining the checkpoint to load conditional on the current seed
    (when doing a grid_search over random seeds and with a multistage training)
    """




    checkpoint_path, policy_id = policy.config.get(
        LOAD_FROM_CONFIG_KEY, (None, None)
    )

    if callable(checkpoint_path):
        checkpoint_path = checkpoint_path(policy.config)

    if checkpoint_path is not None:
        load_one_policy_checkpoint(policy_id, policy, checkpoint_path)
        msg = (
            f"submission restore: checkpoint found for policy_id: "
            f"{policy_id}"
        )
        logger.debug(msg)
    else:
        msg = (
            f"submission restore: NO checkpoint found for policy_id:"
            f" {policy_id} and policy {policy}."
            f"Not found under the config key: {LOAD_FROM_CONFIG_KEY}"
        )
        logger.warning(msg)


def load_one_policy_checkpoint(
    policy_id, policy, checkpoint_path, using_Tune_class=False
):
    """

    :param policy_id: the policy_id of the policy inside the checkpoint that
        is going to be loaded into the policy provided as 2nd argument
    :param policy: the policy to load the checkpoint into
    :param checkpoint_path: the checkpoint to load from
    :param using_Tune_class: to be set to True in case you are loading a
        policy from a Tune checkpoint
        (not a RLLib checkpoint) and that the policy you are loading into was
        created by converting your Tune trainer
        into frozen a RLLib policy
    :return: None
    """

    logger = logging.getLogger(__name__)

    LOAD_FROM_CONFIG_KEY = "checkpoint_to_load_from"

    if using_Tune_class:
        # The provided policy must implement load_checkpoint.
        # This is only intended for the policy class:
        # FrozenPolicyFromTuneTrainer
        policy.load_checkpoint(checkpoint_tuple=(checkpoint_path, policy_id))
    else:
        checkpoint_path = os.path.expanduser(checkpoint_path)
        logger.debug(f"checkpoint_path {checkpoint_path}")
        checkpoint = pickle.load(open(checkpoint_path, "rb"))

        assert "worker" in checkpoint.keys()
        assert "optimizer" not in checkpoint.keys()
        objs = pickle5.loads(checkpoint["worker"])

        # TODO Should let the user decide to load that too
        # self.sync_filters(objs["filters"])
        logger.warning("restoring ckpt: not loading objs['filters']")
        found_policy_id = False
        for p_id, state in objs["state"].items():
            if p_id == policy_id:
                logger.debug(
                    f"going to load policy {policy_id} "
                    f"from checkpoint {checkpoint_path}"
                )
                print(
                    f"going to load policy {policy_id} "
                    f"from checkpoint {checkpoint_path}"
                )
                
                
                del state["_optimizer_variables"]

                policy.set_state(state)
                found_policy_id = True
                break
        if not found_policy_id:
            logger.debug(
                f"policy_id {policy_id} not in "
                f'checkpoint["worker"]["state"].keys() '
                f'{objs["state"].keys()}'
            )
            print(
                f"policy_id {policy_id} not in "
                f'checkpoint["worker"]["state"].keys() '
                f'{objs["state"].keys()}'
            )
            
            
#####################################



def add_support_checkpoint_composition(policy_class):

    if policy_class == DQNTorchPolicy:
      from ray.rllib.agents.dqn.dqn_torch_policy import before_loss_init
      def merged_before_loss_init(*arg, **kwargs):
          before_loss_init(*arg, **kwargs)
          before_loss_init_load_policy_checkpoint(*arg, **kwargs)

      MyPolicyClass = policy_class.with_updates(
        before_loss_init=merged_before_loss_init,
      )
    elif policy_class == PPOTorchPolicy:
      class MyPPOPolicy(PPOTorchPolicy):
        def __init__(self, observation_space, action_space, config):
          super().__init__(observation_space, action_space, config)
          before_loss_init_load_policy_checkpoint(self)

      MyPolicyClass = MyPPOPolicy
    
    elif policy_class in [CoopPolicy,DefectPolicy,TFTAverage,RandPolicy]:
      MyPolicyClass = policy_class

    else: 
      #raise NotImplementedError()
      try:
        class MyPolicyClass(policy_class):
          def __init__(self, observation_space, action_space, config):
            super().__init__(observation_space, action_space, config)
            before_loss_init_load_policy_checkpoint(self)
      except Exception as E:
        print(E)
        raise NotImplementedError

    return MyPolicyClass

###########################################

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