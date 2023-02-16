from pathlib import Path
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
from emu_agents.ppo.ppo_discrete import PPO_discrete
from emu_agents.ppo.replaybuffer import ReplayBuffer
import argparse

parser_ppo = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
parser_ppo.add_argument("--max_train_steps", type=int,
                        default=int(2e5), help=" Maximum number of training steps")
parser_ppo.add_argument("--evaluate_freq", type=float, default=5e3,
                        help="Evaluate the policy every 'evaluate_freq' steps")
parser_ppo.add_argument("--save_freq", type=int,
                        default=20, help="Save frequency")
parser_ppo.add_argument("--batch_size", type=int,
                        default=2048, help="Batch size")
parser_ppo.add_argument("--mini_batch_size", type=int,
                        default=64, help="Minibatch size")
parser_ppo.add_argument("--hidden_width", type=int, default=64,
                        help="The number of neurons in hidden layers of the neural network")
parser_ppo.add_argument("--lr_a", type=float, default=3e-4,
                        help="Learning rate of actor")
parser_ppo.add_argument("--lr_c", type=float, default=3e-4,
                        help="Learning rate of critic")
parser_ppo.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
parser_ppo.add_argument("--lamda", type=float,
                        default=0.95, help="GAE parameter")
parser_ppo.add_argument("--epsilon", type=float, default=0.2,
                        help="PPO clip parameter")
parser_ppo.add_argument("--K_epochs", type=int,
                        default=10, help="PPO parameter")
parser_ppo.add_argument("--use_adv_norm", type=bool, default=True,
                        help="Trick 1:advantage normalization")
parser_ppo.add_argument("--use_state_norm", type=bool,
                        default=True, help="Trick 2:state normalization")
parser_ppo.add_argument("--use_reward_norm", type=bool,
                        default=False, help="Trick 3:reward normalization")
parser_ppo.add_argument("--use_reward_scaling", type=bool,
                        default=True, help="Trick 4:reward scaling")
parser_ppo.add_argument("--entropy_coef", type=float,
                        default=0.01, help="Trick 5: policy entropy")
parser_ppo.add_argument("--use_lr_decay", type=bool, default=False,
                        help="Trick 6:learning rate Decay")
parser_ppo.add_argument("--use_grad_clip", type=bool,
                        default=True, help="Trick 7: Gradient clip")
parser_ppo.add_argument("--use_orthogonal_init", type=bool,
                        default=True, help="Trick 8: orthogonal initialization")
parser_ppo.add_argument("--set_adam_eps", type=float, default=True,
                        help="Trick 9: set Adam epsilon=1e-5")
parser_ppo.add_argument("--use_tanh", type=float, default=True,
                        help="Trick 10: tanh activation function")
parser_ppo.add_argument("--share_model", type=bool, default=True,
                        help="MARL - share model")
parser_ppo.add_argument("--share_memory", type=bool, default=True,
                        help="MARL - share replay buffer")


class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)


class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x


class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)


shared_resources = {}

# cuda:0, cpu, or mps


class AgentPPO():
    def __init__(self, env, obs_set, action_set, agent_name, device='cpu'):
        self.device = torch.device(device)
        self.env = env
        self.agent_name = agent_name
        self.obs_set = obs_set
        self.obs_set_number = len(obs_set)
        self.action_set = action_set
        self.action_number = action_set.n

        args = parser_ppo.parse_args()
        args.state_dim = self.obs_set_number
        args.action_dim = self.action_number
        self.args = args

        self.model: PPO_discrete
        if self.args.share_model:
            if 'model' not in shared_resources:
                shared_resources['model'] = PPO_discrete(args)
            self.model = shared_resources['model']
        else:
            self.model = PPO_discrete(args)

        if self.args.share_memory:
            if not 'memory' in shared_resources:
                shared_resources['memory'] = ReplayBuffer(self.args)
            self.replay_buffer = shared_resources['memory']
        else:
            self.replay_buffer = ReplayBuffer(self.args)

        self.state_norm = Normalization(
            shape=self.args.state_dim)
        self.reward_scale = RewardScaling(
            shape=1, gamma=self.args.gamma)
        self.reward_norm = Normalization(shape=1)

    def get_action_with_prob(self, state):
        if self.args.use_state_norm:
            state = self.state_norm(state)

        state = torch.Tensor(state).to(self.device)
        action, action_log_prob = self.model.choose_action(state)
        return action, action_log_prob

    def get_action(self, state, action_space):
        return self.get_action_with_prob(state)[0]

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

        self.replay_buffer.store(
            agent_state, agent_action, agent_action_log_prob,
            agent_reward, agent_state_next, agent_dw, agent_done)

        if self.replay_buffer.count == self.args.batch_size:
            self.model.update(
                self.replay_buffer, total_steps)
            self.replay_buffer.count = 0
            return True
        return False

    def end_episode(self):
        pass
