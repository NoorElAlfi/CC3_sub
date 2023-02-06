import torch
from torch.nn.functional import mse_loss
from torch.autograd import Variable
import torch.optim as optim
import random
import glob
import os
from .config import Config
from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from .model import Model
from inspect import signature
from typing import Union
import numpy as np
from gym import Space
from submission.Agent.Memory import Memory
import torch
import torch.nn as nn 
from itertools import count
import math

BATCH_SIZE = 16 #64
TAU = 0.005
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, env, obs_set ,action_set, agent_name):
        self.env = env
        self.agent_name = agent_name
        self.obs_set = obs_set[agent_name]
        self.obs_set_number = len(obs_set)
        self.action_set = action_set
        self.action_number = action_set.n
        self.epsilon = EPS_START
        self.steps_done = 0
        self.build_network()

    def build_network(self):
        self.Q_network = Model(self.action_number)#.cuda()
        self.target_network = Model(self.action_number)#.cuda()
        self.optimizer = optim.Adam(self.Q_network.parameters(), lr=Config.lr)
        self.memory = Memory(10000)
    
    def update_target_network(self):
        # copy current_network to target network
        self.target_network.load_state_dict(self.Q_network.state_dict())
    
    def get_action(self, state):
        state = torch.Tensor(state)#.to(device)
        with torch.no_grad():
            values = self.Q_network(state)

        # select a random action wih probability eps
        if random.random() <= self.epsilon:
            action = np.random.randint(0, self.action_number)
        else:
            action = np.argmax(values.numpy())

        return action
       
    def update_epsilon(self):
        if self.epsilon > Config.min_epsilon:
            self.epsilon -= Config.epsilon_discount_rate
    
    def stop_epsilon(self):
        self.epsilon_tmp = self.epsilon        
        self.epsilon = 0        
    
    def restore_epsilon(self):
        self.epsilon = self.epsilon_tmp        
    
    def save(self, step, logs_path):
        os.makedirs(logs_path, exist_ok=True)
        model_list =  glob.glob(os.path.join(logs_path, '*.pth'))
        if len(model_list) > Config.maximum_model - 1 :
            min_step = min([int(li.split('/')[-1][6:-4]) for li in model_list]) 
            os.remove(os.path.join(logs_path, 'model-{}.pth' .format(min_step)))
        logs_path = os.path.join(logs_path, 'model-{}.pth' .format(step))
        self.Q_network.save(logs_path, step=step, optimizer=self.optimizer)
        print('=> Save {}' .format(logs_path)) 
    
    def restore(self, logs_path):
        self.Q_network.load(logs_path)
        self.target_network.load(logs_path)
        print('=> Restore {}' .format(logs_path)) 
        
    def set_initial_values(self, action_space):
        if type(action_space) is dict:
            self.action_params = {action_class: signature(action_class).parameters for action_class in action_space['action'].keys()}
            
    def train(self):

        states, actions, next_states, rewards, is_done = self.memory.sample(BATCH_SIZE)

        q_values = self.Q_network(states)

        next_q_values = self.Q_network(next_states)
        next_q_state_values = self.target_network(next_states)

        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = rewards + TAU * next_q_value * (1 - is_done)

        loss = (q_value - expected_q_value.detach()).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def evaluate(self, env, repeats):
        """
        Runs a greedy policy with respect to the current Q-Network for "repeats" many episodes. Returns the average
        episode reward.
        """
        self.Q_network.eval()
        perform = 0
        for _ in range(repeats):
            state = env.reset()
            done = False
            while not done:
                state = torch.Tensor(state).to(device)
                with torch.no_grad():
                    values = self.Q_network(state)
                action = np.argmax(values.cpu().numpy())
                state, reward, done, _ = env.step(action)
                perform += reward
        self.train()
        return perform/repeats

    def update_parameters(current_model, target_model):
        target_model.load_state_dict(current_model.state_dict())

    def end_episode(self):
        pass
        #self.scan_state = np.zeros(10)
        #self.start_actions = [51, 116, 55]
        #self.agent_loaded = False