import torch
from torch.nn.functional import mse_loss
from torch.autograd import Variable
import torch.optim as optim
import random
import glob
import os
from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from inspect import signature
from typing import Union
import numpy as np
from gym import Space
import torch
import torch.nn as nn
from itertools import count
import math
from emu_agents.rdqn.rainbow_dqn import DQN
from emu_agents.rdqn.replay_buffer import *
from emu_agents.ppo.AgentPPO import RewardScaling, Normalization
import argparse

parser_rdqn = argparse.ArgumentParser("Hyperparameter Setting for DQN")
parser_rdqn.add_argument("--max_train_steps", type=int,
                         default=int(4e5), help=" Maximum number of training steps")
parser_rdqn.add_argument("--evaluate_freq", type=float, default=1e3,
                         help="Evaluate the policy every 'evaluate_freq' steps")
parser_rdqn.add_argument("--evaluate_times", type=float,
                         default=3, help="Evaluate times")

parser_rdqn.add_argument("--buffer_capacity", type=int,
                         default=int(1e5), help="The maximum replay-buffer capacity ")
parser_rdqn.add_argument("--batch_size", type=int,
                         default=256, help="batch size")
parser_rdqn.add_argument("--hidden_dim", type=int, default=256,
                         help="The number of neurons in hidden layers of the neural network")
parser_rdqn.add_argument("--lr", type=float, default=1e-4,
                         help="Learning rate of actor")
parser_rdqn.add_argument("--gamma", type=float,
                         default=0.99, help="Discount factor")
parser_rdqn.add_argument("--epsilon_init", type=float,
                         default=0.5, help="Initial epsilon")
parser_rdqn.add_argument("--epsilon_min", type=float,
                         default=0.1, help="Minimum epsilon")
parser_rdqn.add_argument("--epsilon_decay_steps", type=int, default=int(1e5),
                         help="How many steps before the epsilon decays to the minimum")
parser_rdqn.add_argument("--tau", type=float, default=0.005,
                         help="soft update the target network")
parser_rdqn.add_argument("--use_soft_update", type=bool,
                         default=True, help="Whether to use soft update")
parser_rdqn.add_argument("--target_update_freq", type=int, default=200,
                         help="Update frequency of the target network(hard update)")
parser_rdqn.add_argument("--n_steps", type=int, default=5, help="n_steps")
parser_rdqn.add_argument("--alpha", type=float,
                         default=0.6, help="PER parameter")
parser_rdqn.add_argument("--beta_init", type=float, default=0.4,
                         help="Important sampling parameter in PER")
parser_rdqn.add_argument("--use_lr_decay", type=bool,
                         default=True, help="Learning rate Decay")
parser_rdqn.add_argument("--grad_clip", type=float,
                         default=10.0, help="Gradient clip")

parser_rdqn.add_argument("--use_double", type=bool, default=True,
                         help="Whether to use double Q-learning")
parser_rdqn.add_argument("--use_dueling", type=bool,
                         default=True, help="Whether to use dueling network")
parser_rdqn.add_argument("--use_noisy", type=bool,
                         default=True, help="Whether to use noisy network")
parser_rdqn.add_argument("--use_per", type=bool,
                         default=True, help="Whether to use PER")
parser_rdqn.add_argument("--use_n_steps", type=bool, default=True,
                         help="Whether to use n_steps Q-learning")

parser_rdqn.add_argument("--share_model", type=bool, default=True,
                         help="MARL - share model")
parser_rdqn.add_argument("--share_memory", type=bool, default=True,
                         help="MARL - share replay buffer")

parser_rdqn.add_argument("--use_state_norm", type=bool,
                         default=False, help="Trick 2:state normalization")
parser_rdqn.add_argument("--use_reward_norm", type=bool,
                         default=False, help="Trick 3:reward normalization")
parser_rdqn.add_argument("--use_reward_scaling", type=bool,
                         default=False, help="Trick 4:reward scaling")
shared_resources = {}


def get_memory(args):
    replay_buffer = None
    if args.use_per and args.use_n_steps:
        replay_buffer = N_Steps_Prioritized_ReplayBuffer(args)
    elif args.use_per:
        replay_buffer = Prioritized_ReplayBuffer(args)
    elif args.use_n_steps:
        replay_buffer = N_Steps_ReplayBuffer(args)
    else:
        replay_buffer = ReplayBuffer(args)
    return replay_buffer


class AgentRDQN():
    def __init__(self, env, obs_set, action_set, agent_name, device='cpu'):
        self.device = torch.device(device)
        self.env = env
        self.agent_name = agent_name
        self.obs_set = obs_set
        self.obs_set_number = len(obs_set)
        self.action_set = action_set
        self.action_number = action_set.n

        args = parser_rdqn.parse_args()
        args.state_dim = self.obs_set_number
        args.action_dim = self.action_number
        self.args = args

        self.model: DQN
        if self.args.share_model:
            if 'model' not in shared_resources:
                shared_resources['model'] = DQN(args)
            self.model = shared_resources['model']
        else:
            self.model = DQN(args)

        if self.args.share_memory:
            if not 'memory' in shared_resources:
                shared_resources['memory'] = get_memory(self.args)
            self.replay_buffer = shared_resources['memory']
        else:
            self.replay_buffer = get_memory(self.args)

        self.state_norm = Normalization(
            shape=self.args.state_dim)
        self.reward_scale = RewardScaling(
            shape=1, gamma=self.args.gamma)
        self.reward_norm = Normalization(shape=1)

        if args.use_noisy:
            self.epsilon = 0
        else:
            self.epsilon = self.args.epsilon_init
            self.epsilon_min = self.args.epsilon_min
            self.epsilon_decay = (
                self.args.epsilon_init - self.args.epsilon_min) / self.args.epsilon_decay_steps

    def get_action_with_prob(self, state):
        # for training purpose
        if self.args.use_state_norm:
            state = self.state_norm(state)

        state = torch.Tensor(state).to(self.device)
        action = self.model.choose_action(state, self.epsilon)

        # it does not need log prob
        return action, 0

    def get_action(self, state, action_space):
        # for evaluation purpose
        if self.args.use_state_norm:
            state = self.state_norm(state)

        state = torch.Tensor(state).to(self.device)
        return self.model.choose_action(state, 0)

    def train(
        self,
        agent_state,
        agent_state_next,
        agent_action,
        agent_action_log_prob,
        agent_reward,
        agent_done,
        agent_dw,
        total_steps,
    ):

        if self.args.use_reward_norm:
            agent_reward = self.reward_norm(agent_reward)
        elif self.args.use_reward_scaling:
            agent_reward = self.reward_scale(agent_reward)

        if self.args.use_state_norm:
            agent_state = self.state_norm(agent_state)
            agent_state_next = self.state_norm(agent_state_next)

        self.replay_buffer.store_transition(
            agent_state, agent_action,
            agent_reward, agent_state_next, agent_dw, agent_done)

        if self.replay_buffer.count == self.args.batch_size:
            self.model.learn(
                self.replay_buffer, total_steps)
            return True
        return False

    def end_episode(self):
        pass
