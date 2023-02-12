import random
import torch
import numpy as np


class Memory:
    def __init__(self, max_len, device):
        self.rewards = []
        self.state = []
        self.action = []
        self.is_done = []
        self.max_len = max_len
        self.device = device

    def update(self, state, action, reward, done):
        # if the episode is finished we do not save to new state. Otherwise we have more states per episode than rewards
        # and actions whcih leads to a mismatch when we sample from memory.
        # if not done:
        self.state.append(state)
        self.action.append(action)
        self.rewards.append(reward)
        self.is_done.append(done)

    def sample(self, batch_size):
        """
        sample "batch_size" many (state, action, next state, reward, is_done) datapoints.
        """
        # print(self.state)
        n = len(self.is_done) - 2

        # print(len(self.is_done))
        idx = random.sample(range(0, n-1), batch_size)

        sampled_state = torch.Tensor(np.array([
            self.state[i] for i in idx
        ])).to(self.device)

        sampled_action = torch.Tensor(np.array([
            self.action[i] for i in idx
        ])).type(torch.int64).to(self.device)

        sampled_next_state = torch.Tensor(np.array([
            self.state[i+1] for i in idx
        ])).to(self.device)

        sampled_reward = torch.Tensor(np.array([
            self.rewards[i+1] for i in idx
        ])).to(self.device)

        sampled_is_done = torch.Tensor(np.array([
            self.is_done[i] for i in idx
        ])).to(self.device)

        return (sampled_state, sampled_action,
                sampled_next_state, sampled_reward,
                sampled_is_done)

        # return torch.Tensor(self.state)[idx].to(device), torch.LongTensor(self.action)[idx].to(device), \
        #     torch.Tensor(self.state)[1+np.array(idx)].to(device), torch.Tensor(self.rewards)[idx].to(device), \
        #     torch.Tensor(self.is_done)[idx].to(device)

    def reset(self):
        self.rewards.clear()
        self.state.clear()
        self.action.clear()
        self.is_done.clear()

    def __len__(self):
        return len(self.state)
