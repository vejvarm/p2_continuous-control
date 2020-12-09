[//]: # (Image References)

[image1]: https://siasky.net/_BmMxqys6J0ykCLIZWCLs9srIkWwSGFwIzym5BszVwXE3A "Trained Agent"


# Project 2: Continuous Control

### 1 Introduction

This project focuses on the  [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

### 2 The Environment
The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The selected Reacher Unity environment contains 20 identical agents, each with its own copy of the environment.  

#### 2.1 Solving the Environment

The task is episodic. One episode lasts for 1000 steps.

The barrier for solving the second version of the environment is taking into account the presence of many agents.  In particular, the agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).  Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores. 
- This yields an **average score** for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 

### 3 Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the repository folder and unzip (or decompress) the file. 

#### 3.1 Requirements
To install required dependencies for running the code using Anaconda please follow the instructions bellow.
1. Make a new environment e.g. \
    `conda create -n drlnd-p2`
    
2. Activate the environment
    - `activate drlnd-p2` (for Windows)
    - `conda activate drlnd-p2` (elsewhere)
    
3. Install python version 3.6 (newer versions have dependency problems with unityagents) \
    `conda install python=3.6`
    
4. Install pytorch based on your preferences (please refer to the Install table on [pytorch site](https://pytorch.org/))
    - for example installing PyTorch witch CUDA 10.2 support: \
        `conda install pytorch torchvision cudatoolkit=10.2 -c pytorch`
        
5. Install jupyter notebook package \
    `conda install notebook`
    
6. Install Unity Machine Learning Agents package using pip \
    `pip install unityagents`

Congratulations! If everything went well, you should now have an environment which is ready to run the 
code in this repository. 

### 4 Instructions
To run the notebook, you just have to activate the previously created conda repository from Anaconda console and start a jupyter notebook server in the local clone root.
1. Activate the environment
    - `activate drlnd-p2` (for Windows)
    - `conda activate drlnd-p2` (elsewhere)

2. Change current console folder to the root of the cloned repository \
    `cd \d path/to/local/repo/clone/` 

3. Run jupyter notebook server \
    `jupyter notebook`
    
4. Open `Continuous_Control.ipynb` and follow the instructions

