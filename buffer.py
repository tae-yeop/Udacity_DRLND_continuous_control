from collections import deque, namedtuple
import random
import torch
import numpy as np

device = ("cuda:0" if torch.cuda.is_available() else "cpu")
class ReplayBuffer():
    def __init__(self, buf_size, batch_size, seed):
        """
        Set memory and bacth_size
        Params
        =====
        buf_size (int): size of memory
        batch_size (int): number of samples to be sampled
        seed (int): random seed
        """
        # When the replay buffer was full, the oldest sample needs to be discarded.
        # So deque is suitalbe data structure. 
        self.memory = deque(maxlen=buf_size)
        random.seed(seed)
        self.batch_size = batch_size
        self.experience = namedtuple('Trajectory', field_names=["state", "action", "reward", "next_state", "done"])
        
    def __len__(self):
        """
        Return the size of memory
        """
        return len(self.memory)
    def add(self, state, action, reward, next_state, done):
        """
        Add the agent's experiences at eacy time to the memory

        Params
        ======
        state (numpy array) [state_size,]
        action (numpy array) [action_size,]
        reward (float)
        next_state (numpy array) [state_size,]
        done (bool) 
        """
        # Instantiate new experience with custom nemaedTuple
        e = self.experience(state, action, reward, next_state, done)
        # Add the tuple to the memory
        self.memory.append(e)
        
    def sample(self):
        """
        Draw a sample.
        Since the sample data will be used by pytorch model, 
        It needs to be converted to a torch Tensor.
        
        Returns
        =====
        A tuple of torch tensor :
            Each tenosr's outermost dimension is batch_size.
            for example, states shape is [bacth_size, state_size]
            dones shape is [bact_size,]
        """
        # list of sampled experience namedtuple of size of self.batch_size
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        # dones is needed to calculated the Q-value. At terminal state(dones=1), the Q-value should be just latest rewards.
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones)
        