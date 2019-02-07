import torch
import numpy as np
from model import Actor, Critic, Critic2
from buffer import ReplayBuffer
import torch.nn.functional as F

device = ("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyper parameters
BUFFER_SIZE = int(5e4)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3        # learning rate of the actor 
LR_CRITIC = 1e-3       # learning rate of the critic
WEIGHT_DECAY = 1e-4     # L2 weight decay for the critic
TRAIN_FREQ = 20
UPDATE_COUNT = 20
NOISE_SIGMA = 0.05
TRUE_BATCH_SIZE = 128
PRIORITY_SELECTION = False
EPSILON = 0.5

class DDPG():
    def __init__(self, state_size, action_size, seed, critic_type, lr_a, lr_c, tau, gamma, w_decay):
        
        torch.manual_seed(seed)

        self.actor = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_opti = torch.optim.Adam(self.actor.parameters(), lr=lr_a)


        if critic_type==0:
            self.critic = Critic(state_size, action_size, seed).to(device)
            self.critic_target = Critic(state_size, action_size, seed).to(device)

        elif critic_type==1:
            self.critic = Critic2(state_size, action_size, seed).to(device)
            self.critic_target = Critic2(state_size, action_size, seed).to(device)
        self.critic_opti = torch.optim.Adam(self.critic.parameters(), lr=lr_c, weight_decay=w_decay)


        self.hard_copy(self.actor, self.actor_target)
        self.hard_copy(self.critic, self.critic_target)

        
        self.noise = OUNoise(action_size, seed)
        self.gamma = gamma
        self.tau = tau
        self.epsilon = EPSILON

    def hard_copy(self, local, target):

        for (local_param, target_param) in zip(local.parameters(), target.parameters()):
            target_param.data.copy_(local_param.data)
        
    def soft_copy(self, local, target, tau):

        for (local_param, target_param) in zip(local.parameters(), target.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def act(self, state):
        """
        Select action based on the current actor policy and exploration.
        Params
        =====
        state (numpy array) [state_size,]
        Returns
        =====
        action (numpy array) [action_size,]
        """
        # state shape : [1,state_size] : batch norm1d need 2d or 3d shape tensor
        state = torch.tensor(state).unsqueeze(0).float().to(device)
        # enable batch norm eval mode : disable updating running mean, variance
        self.actor.eval()
        # deactivate autograd engine
        with torch.no_grad():
            # from GPU memory to main memory if device is cuda
            action = self.actor(state).cpu().data.numpy().squeeze(0)
        # batch norm back to train mode
        self.actor.train()

        
        action = action + self.noise.sample() # [action_size,]

        return np.clip(action, -1, 1)

    def control_noise(self):
        noise_reset_decision = False
        if np.random.randn(1) < self.epsilon:
            noise_reset_decision = True
        return noise_reset_decision

    def reset(self):
        """
        Reset noise
        """
        #if self.control_noise():
        self.noise.reset()

    def update(self, e):
        # dones : [batch_size,]
        # rewards : [batch_size,]
        states, actions, rewards, next_states, dones = e

        # ---------Update local critic----------------#
        
        # target critic's parameters are not updated
        # So its gradient computation is not needed
        #with torch.no_grad():
        # next_action shape [batch_size, action_size]
        next_actions = self.actor_target(next_states)
        # q_next shape : [batch_size, 1]
        q_next = self.critic_target(next_states, next_actions)
        # y shape : [batch_size, 1]
        y = rewards.view(-1,1) + self.gamma*q_next*(1-dones.view(-1,1))

        # [batch_size, 1]
        self.critic.train()
        q_value = self.critic(states, actions)

        #hubber_loss = torch.nn.SmoothL1Loss()
        critic_loss = F.mse_loss(q_value, target=y.to(device))#F.smooth_l1_loss(q_value, target=y)
        self.critic_opti.zero_grad() # Clear gradients
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_opti.step() # perform single optimization step


        # ---------Update local critic----------------#
        predicted_action = self.actor(states)
        self.critic.eval()
        actor_loss = -self.critic(states, predicted_action).mean()
        self.actor_opti.zero_grad() 
        actor_loss.backward() 
        self.actor_opti.step()
        self.critic.train()

        # ---------Update target networks----------------#
        self.soft_copy(self.actor, self.actor_target, self.tau)
        self.soft_copy(self.critic, self.critic_target, self.tau)


    def priority_update(self, e):
        """
        Get batch_size of eperiences, but select the expericens of TRUE_BATCH_SIZE
        This is indirect priority buffer implementation.
        Since each agents may have different TD error,
        The priority can vary among the agents.
        So calculate the proiorty for each individual agent.

        TRUE_BATCH_SIZE must be smaller than BATCH_SIZE
        """
        # states : [batch_size, state_size]
        # actions : [batch_size, action_size]
        # rewards : [batch_size,]
        # next_states : [batch_size, state_size]
        # dones : [batch_size,]
        states, actions, rewards, next_states, dones = e

        # ---------Update local critic----------------#
        self.critic_opti.zero_grad() # Clear gradients
        # target critic's parameters are not updated
        # So its gradient computation is not needed
        with torch.no_grad():
            # next_action shape [batch_size, action_size]
            next_actions = self.actor_target(next_states)
            # q_next shape : [batch_size, 1]
            q_next = self.critic_target(next_states, next_actions)
            # y shape : [batch_size, 1]
            y = rewards.view(-1,1) + self.gamma*q_next*(1-dones.view(-1,1))

        # [batch_size, 1]
        q_value = self.critic(states, actions)

        # Select 
        # td_error's requires_grad is False
        td_error = y-q_value.detach().clone()
        priority = torch.abs(td_error)
        prob = priority/sum(priority)
        prob_numpy = prob.cpu().numpy()
        index = np.random.choice(len(prob_numpy),TRUE_BATCH_SIZE, False, prob_numpy.reshape(-1))

        #hubber_loss = torch.nn.SmoothL1Loss()
        critic_loss = F.mse_loss(q_value[index], target=y[index])#F.smooth_l1_loss(q_value, target=y)
        critic_loss.backward()
        self.critic_opti.step() # perform single optimization step


        # ---------Update local critic----------------#
        self.actor_opti.zero_grad() 
        predicted_action = self.actor(states)
        actor_loss = -self.critic(states, predicted_action).mean()
        actor_loss.backward() 
        self.actor_opti.step()

        # ---------Update target networks----------------#
        self.soft_copy(self.actor, self.actor_target, self.tau)
        self.soft_copy(self.critic, self.critic_target, self.tau)

class MultiAgent():

    def __init__(self, state_size, action_size, num_agents, seed, critic_type=0, lr_a=LR_ACTOR, lr_c=LR_CRITIC, tau=TAU, gamma=GAMMA, w_decay=WEIGHT_DECAY, buf_size=BUFFER_SIZE, batch_size=BATCH_SIZE):

        self.batch_size = batch_size
        self.num_agents = num_agents
        self.agents = []
        for i in range(self.num_agents):
            self.agents.append(DDPG(state_size, action_size, seed, critic_type, lr_a, lr_c, tau, gamma, w_decay))

        self.memory = ReplayBuffer(buf_size, batch_size, seed)
        self.step_count = 0

    def act(self, states):
        """
        Select agents' actions to interact with the environment
        Params
        =====
        states : (numpy array) [num_agents, state_size]

        Returns
        =====
        actions : (numpy array) [num_agents, action_size]
        """
        agents_actions = [agent.act(state) for agent, state in zip(self.agents, states)]
        # actions = np.stack(agents_actions)
        return agents_actions
    
    def train_condition_check(self):
        memory_check = False
        train_check = False
        if len(self.memory)>=self.batch_size:
            memory_check = True

        if self.step_count % TRAIN_FREQ == 0:
           train_check = True
        # if both condition are True, then True
        return np.all([memory_check, train_check])

    def step(self, states, actions, rewards, next_states, dones):
        """
        Params
        =====
            states (numpy array) [num_agents, state_size]
            actions (numpy array) [num_agetns, action_size]
            rewards (float list) [num_agents]
            next_states (numpy array) [num_agents, state_size]
            dones (boolean list) [num_agents]
        """
        # add each experiences gathered by agents to the buffer.
        for i in range(self.num_agents):
            # state[i] shape : [state_size]
            # actions[i] shape : [action_size]
            # dones[i] shape : (bool)
            self.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])

        self.step_count += 1
        #if len(self.memory)>=self.batch_size:
        if self.train_condition_check():
            for agent in self.agents:
                for i in range(UPDATE_COUNT):
                    e = self.memory.sample()
                    # Use prioirty buffer method
                    if PRIORITY_SELECTION:
                        agent.priority_update(e)
                    else:
                        agent.update(e)


    def reset(self):
        for agent in self.agents:
            agent.reset()

    def save_model(self, path='data/model/'):
        """
        Save model parameters.

        """
        for i in range(self.num_agents):
            torch.save(self.agents[i].actor.state_dict(), path+'agent{}_actor.pth'.format(i))
            torch.save(self.agents[i].critic.state_dict(), path+'agent{}_critic.pth'.format(i))
            torch.save(self.agents[i].actor_target.state_dict(), path+'agent{}_actor_target.pth'.format(i))
            torch.save(self.agents[i].critic_target.state_dict(), path+'agent{}_critic_target.pth'.format(i))
   
    def load_model(self, path='data/model/'):
        """
        Load model parameters
        """
        if torch.cuda.is_available():
            for i in range(self.num_agents):
                self.agents[i].actor.load_state_dict(torch.load(path+'agent{}_actor.pth'.format(i)))
                self.agents[i].critic.load_state_dict(torch.load(path+'agent{}_critic.pth'.format(i)))
                self.agents[i].actor_target.load_state_dict(torch.load(path+'agent{}_actor_target.pth'.format(i)))
                self.agents[i].critic.load_state_dict(torch.load(path+'agent{}_critic_target.pth'.format(i)))
        else:
            for i in range(self.num_agents):
                self.agents[i].actor.load_state_dict(torch.load(path+'agent{}_actor.pth'.format(i), map_location='cpu'))
                self.agents[i].critic.load_state_dict(torch.load(path+'agent{}_critic.pth'.format(i), map_location='cpu'))
                self.agents[i].actor_target.load_state_dict(torch.load(path+'agent{}_actor_target.pth'.format(i),map_location='cpu'))
                self.agents[i].critic.load_state_dict(torch.load(path+'agent{}_critic_target.pth'.format(i), map_location='cpu'))
            
class OUNoise():
    def __init__(self, action_size, seed, mu=.0, theta=0.15, sigma=NOISE_SIGMA):
        """
        Set initial random process state.
        
        Params
        =====
            action_size (int)
            seed (int) : For determinsitc random process. 
            mu (float) : center that noise will move around.
            sacle (flaot) : scale factor
            theta(flaot) : parameter for the process
            sigma(float) : parameter for the process
        """
        #self.noise_state = np.ones(action_size)*mu
        self.mu = np.ones(action_size)*mu # shape : [action_size,]
        self.theta = theta
        self.sigma = sigma
        np.random.seed(seed)
        self.reset()
    
    def reset(self):
        """
        Reset the noise state to the mu.
        """
        self.noise_state = self.mu

    def sample(self):
        """
        Returns
        =====
            noise_state (numpy array) [action_size,]
        """
        x = self.noise_state
        dx = self.theta*(self.mu-x) + self.sigma*(np.random.randn(len(x)))
        self.noise_state = x+ dx
        return (self.noise_state)

    