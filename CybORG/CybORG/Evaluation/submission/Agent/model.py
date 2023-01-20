import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, state_dim, action_num):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_num
        '''
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(8, 8), stride=4),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=2),
            nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1),
            nn.ReLU())
        '''
        self.fc1 = nn.Sequential(
            nn.Linear(108, 64),
            nn.ReLU())
        self.fc2 = nn.Sequential( 
            nn.Linear(64, 64),
            nn.ReLU())
        self.fc3 = nn.Linear(64, action_num)
    
    def forward(self, observation):
        #out1 = self.conv1(observation)
        #out2 = self.conv2(out1)        
        #out3 = self.conv3(out2)
        out1 = self.fc1(observation)        
        out2 = self.fc2(out1)
        out = self.fc3(out2)
        
        return out
    

    def save(self, path, step, optimizer):
        torch.save({
            'step': step,
            'state_dict': self.state_dict(),
            'optimizer': optimizer.state_dict()
        }, path)
            
    def load(self, checkpoint_path, optimizer=None):
        checkpoint = torch.load(checkpoint_path)
        step = checkpoint['step']
        self.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])