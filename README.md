[//]: # (Image References)

[image1]:
https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif
"Trained Agent"

# Project 2 : Continuous Control

### Project Details

In this project, You can train double-jointed arm agents to move to target locations.

![Trained Agent][image1]

A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

There are two versions of reacher environemnt; One agent version and Twenty agents version. To solve the problem, the agents must get average episode score of +30 over 100 conscutive episodes. In twenty agents case, the single episode score is calculated by averaging the all of the scores which each agent have. Then this single episode score is averaged over 100 cosecutive episodes. 

---
### Getting Started

To run this project, You need several python packages, Unity ML-Agents Toolkit and the environment.

- numpy(>=1.11)
- pytorch(>=0.4)
- matplotlib(>=1.11)

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```
2. Clone the udacity nanodegree repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies. This is for installing [Unity ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents) and all the needed python packages.
    ```bash
    git clone https://github.com/udacity/deep-reinforcement-learning.git
    cd deep-reinforcement-learning/python
    pip install .
    ```
3. Download the unity environment from one of the links below. In this case you will download the Reacher environment.

    1. Version 1 : One(1) Agent
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)
    
    2. Version 2 : Twenty(20) Agents
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the environment.

4. Place the file in the data folder, and uzip the file.
5. When you try to open the unity environment in the jupyter notebook, You need to specify the path of the environment.

4. Place the file in the data folder, and uzip the file.
---
### Instructions

There are 7 main elements in this project. 

- Report.ipynd
- model.py
- buffer.py
- agent.py
- model_checkpoint.pth (in ./model)
- params.json (in ./model)
- scores.json (in ./model)

*Report.ipynd* includes simple summary of the algorithm and codes for training the agent, visualizing the rewards graphs and running the agent. You can try experiments by setting different hyperparameters in this Report.ipynd file.

The code is fully compatible with both one agent and twenty agents versions. Please specify the path of unity environemnt properly.

You can modify the actor and critic network model via *model.py*. 
Currently 4 different critic models are included.

*buffer.py* includes replay buffer class.

*agent.py* includes base ddpg model and noise process model. To make it compatible to twenty agent version, multiagent class is implemented. this multiagent class handles the individual ddpg agent and shared replay buffer. 

*model_checkpoint.pth* is parameters of the agent's networks. You can check the all the checkpoint.pth in ./model folder. Instead of training, You can use this checkpoint directly to see how the agents interact with the each other. To see how the agent behave, Run the cell in Report.ipynd

*params.json* is the log file for the hyperparmeters of particular experiment model.

*scores.json* is the log file for the particular experiment model.

After several experiments, I've found that model_35_single is the best model. Please check the ./model folder.

---
### References

- Lillicrap et al., [Continuous control with deep reinforcement learning](http://arxiv.org/abs/1509.02971)

---
### License

This project is covered under the [MIT License.](./LICENSE)