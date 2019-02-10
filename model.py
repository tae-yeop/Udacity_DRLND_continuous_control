import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
import numpy as np

device = ("cuda:0" if torch.cuda.is_available() else "cpu")



class Actor(nn.Module):
    def __init__(self, state_size, action_size, params):

        super().__init__()
        
        seed = params['SEED']
        fc1_units = params['ACTOR_FC1']
        fc2_units = params['ACTOR_FC2']

        torch.manual_seed(seed)
        self.bn0 = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, fc1_units,bias=False)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units,bias=False)
        self.bn2 = nn.BatchNorm1d(fc2_units)
        self.fc3 = nn.Linear(fc2_units,action_size)

        self.reset_parameters()

    def reset_parameters_xavier(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                I.xavier_normal_(m.weight)

    def get_fan_in(self, layer):
        """
        Get the fan-in in each layer.
        """
        lim = 1/np.sqrt(layer.in_features)
        return (-lim, lim)
        
    def reset_parameters(self):
        """
        Initialize weights and bais in each layer
        """
        I.uniform_(self.fc1.weight, *self.get_fan_in(self.fc1))
        I.uniform_(self.fc2.weight, *self.get_fan_in(self.fc2))
        I.uniform_(self.fc3.weight, -3*1e-3, 3e-3)

    def forward(self, state):
        x = self.bn0(state)
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.fc3(x)
        return F.tanh(x)


class Critic(nn.Module):

    def __init__(self, state_size, action_size, params):
        super().__init__()

        seed = params['SEED']
        fc1_units = params['FC1']
        fc2_units = params['FC2']
        
        torch.manual_seed(seed)
        
        self.bn0 = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, fc1_units,bias=False)

        self.fc_merged = nn.Linear(fc1_units+action_size, fc2_units)
        self.fc2 = nn.Linear(fc2_units, 1)

        self.reset_parameters()

    def reset_parameters_xavier(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                I.xavier_normal_(m.weight)

    def get_fan_in(self, layer):
        """
        Get the fan-in in each layer.
        """
        fan_in = 1/np.sqrt(layer.in_features)
        return (-fan_in, fan_in)

    def reset_parameters(self):
        """
        Initialize weights and bais in each layer
        """
        I.uniform_(self.fc1.weight, *self.get_fan_in(self.fc1))
        I.uniform_(self.fc_merged.weight, *self.get_fan_in(self.fc_merged))
        I.uniform_(self.fc2.weight, -3*1e-3, 3e-3)   

    def forward(self, state, action):

        x = self.bn0(state)
        x = F.relu(self.fc1(x))
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.fc_merged(x))
        x = self.fc2(x)
      
        return x

class Critic2(nn.Module):

    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=64):
        super().__init__()

        torch.manual_seed(seed)
        
        self.fc_s1 = nn.Linear(state_size, fc1_units,bias=False)
        self.bn_s = nn.BatchNorm1d(fc1_units)
        self.fc_s2 = nn.Linear(fc1_units, fc2_units)

        self.fc_a1 = nn.Linear(action_size, fc1_units,bias=False)
        self.bn_a = nn.BatchNorm1d(fc1_units)
        self.fc_a2 = nn.Linear(fc1_units, fc2_units)

        self.fc_merged = nn.Linear(fc2_units, 1)

        self.reset_parameters()

    def reset_parameters_xavier(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                I.xavier_normal_(m.weight)

    def reset_parameters(self):
        """
        Initialize weights and bais in each layer
        """
        I.uniform_(self.fc_s1.weight, *self.get_fan_in(self.fc_s1))
        I.uniform_(self.fc_s2.weight, *self.get_fan_in(self.fc_s2))
        I.uniform_(self.fc_a1.weight, *self.get_fan_in(self.fc_a1))
        I.uniform_(self.fc_a2.weight, *self.get_fan_in(self.fc_a2))
        I.uniform_(self.fc_merged.weight, -3*1e-3, 3e-3)

    def get_fan_in(self, layer):
        """
        Get the fan-in in each layer.
        """
        fan_in = 1/np.sqrt(layer.in_features)
        return (-fan_in, fan_in)

    def forward(self, state, action):
        x = F.relu(self.fc_s1(state))
        x = self.bn_s(x)
        x = F.relu(self.fc_s2(x))

        y = F.relu(self.fc_a1(action))
        y = self.bn_a(y)
        y = F.relu(self.fc_a2(y))

        net = F.relu(x+y)
        net = self.fc_merged(net)
        return net

class Critic3(nn.Module):

    def __init__(self, state_size, action_size, params):
        super().__init__()

        seed = params['SEED']
        fc1_units = params['CRITIC_FC1']
        fc2_units = params['CRITIC_FC2']
        fc3_units = params['CRITIC_FC2']
        torch.manual_seed(seed)
 
        
        self.bn0 = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, fc1_units,bias=False)

        self.fc_merged = nn.Linear(fc1_units+action_size, fc2_units)
        self.bn1 = nn.BatchNorm1d(fc2_units)

        self.fc2 = nn.Linear(fc2_units, fc3_units)
        self.bn2 = nn.BatchNorm1d(fc3_units)

        self.fc3 = nn.Linear(fc3_units, 1)

        self.reset_parameters()

    def reset_parameters_xavier(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                I.xavier_normal_(m.weight)

    def reset_parameters(self):
        """
        Initialize weights and bais in each layer
        """
        I.uniform_(self.fc1.weight, *self.get_fan_in(self.fc1))
        I.uniform_(self.fc_merged.weight, *self.get_fan_in(self.fc_merged))
        I.uniform_(self.fc2.weight, *self.get_fan_in(self.fc2))
        I.uniform_(self.fc3.weight, -3*1e-3, 3e-3)

    def get_fan_in(self, layer):
        """
        Get the fan-in in each layer.
        """
        fan_in = 1/np.sqrt(layer.in_features)
        return (-fan_in, fan_in)

    def forward(self, state, action):

        x = self.bn0(state)
        x = F.relu(self.fc1(x))
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.fc_merged(x))
        x = F.relu(self.fc2(self.bn1(x)))
        x = self.fc3(self.bn2(x))
       
        return x

class Critic4(nn.Module):

    def __init__(self, state_size, action_size, params):
        super().__init__()

        seed = params['SEED']
        fc1_units = params['CRITIC_FC1']
        fc2_units = params['CRITIC_FC2']
        fc3_units = params['CRITIC_FC2']
        torch.manual_seed(seed)
 
        
        
        self.fc1 = nn.Linear(state_size, fc1_units,bias=False)
        self.bn1 = nn.BatchNorm1d(fc1_units)

        self.fc_merged = nn.Linear(fc1_units+action_size, fc2_units)
        self.bn_merged = nn.BatchNorm1d(fc2_units)

        self.fc2 = nn.Linear(fc2_units, fc3_units)
        self.bn2 = nn.BatchNorm1d(fc3_units)

        self.fc3 = nn.Linear(fc3_units, 1)

        self.reset_parameters()

    def reset_parameters_xavier(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                I.xavier_normal_(m.weight)

    def reset_parameters(self):
        """
        Initialize weights and bais in each layer
        """
        I.uniform_(self.fc1.weight, *self.get_fan_in(self.fc1))
        I.uniform_(self.fc_merged.weight, *self.get_fan_in(self.fc_merged))
        I.uniform_(self.fc2.weight, *self.get_fan_in(self.fc2))
        I.uniform_(self.fc3.weight, -3*1e-3, 3e-3)

    def get_fan_in(self, layer):
        """
        Get the fan-in in each layer.
        """
        fan_in = 1/np.sqrt(layer.in_features)
        return (-fan_in, fan_in)

    def forward(self, state, action):

        x = F.relu(self.fc1(state))
        x = self.bn1(x)

        x = torch.cat((x, action), dim=1)
        x = F.relu(self.fc_merged(x))
        x = self.bn_merged(x)

        x = F.relu(self.fc2(x))
        x = self.bn2(x)

        x = self.fc3(x)
       
        return x


