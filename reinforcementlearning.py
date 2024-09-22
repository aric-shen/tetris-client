'''
Based on:
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
'''


import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from game import Board
from piece import Piece
from config import AI

from sys import exit

from RLhelper import *


is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(64, 1, kernel_size=(2, 2))
        self.conv3 = nn.Conv2d(128, 128, kernel_size=(3, 3))
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3))


        self.layer1 = nn.Linear(n_observations, 128)
        #self.layer1 = nn.Linear(50, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 128)
        self.layer4 = nn.Linear(128, 128)
        self.finalLayer = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        '''
        x = x.view(-1, 1, 246)
        next = x[:, :, :6]
        board = x[:, :, 6:]
        board = board.view(-1, 1, 10, 24)
        
        board = F.relu(self.conv1(board))
        board = F.relu(self.conv2(board))
        board = F.max_pool2d(board, (2, 2))
        board = F.dropout(board, p=0.2, training=self.training)
        board = torch.flatten(board, start_dim=2)
        

        x = torch.cat((next, board), dim=2)
        '''
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        return self.finalLayer(x)
    
# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 512
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 10000
TAU = 0.005
LR = 1e-4
MID_LAYERS = 3

# Calculate number of actions
# Actions count from bottom left corner, count across and then up.
# First 7*24 actions for horizontal I pieces
# Next 10*21 actions represent vertical I pieces
# Next 8*23 actions for state 0 J piece
# 9*22 for s1 J
# 8*23 for s2 J
# 9*22 for s3 J
# duplicate for L, T pieces
# 9*23 spots for O piece
# 8*23 for SZ s0,s2
# 9*22 for SZ s1,s3
# total 3641 in layout: IJLOSTZ
n_actions = 3641


# Get the number of state observations
# First five elements of state are the next queue, the rest are the board array
b = Board()
state = b.toList()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0


def select_action(boardState):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    state = torch.tensor(boardState.toList(), dtype = torch.float32, device=device)
    validActions = getValidActions(boardState)
    '''
    mask = []
    for i in range(n_actions):
        if i in validActions:
            mask.append(1)
        else:
            mask.append(0)
    mt = torch.tensor(mask) * policy_net(state)
    for i in validActions:
        print(str(i) + " is " + str(mt[i]))
    print(mt.argmax())
    '''
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            mask = []
            for i in range(n_actions):
                if i in validActions:
                    mask.append(1)
                else:
                    mask.append(0)
            mt = torch.tensor(mask, device=device) * policy_net(state)

            return mt.argmax()
    else:
        #print("Naive Choice")
        chosen, choice = getNaiveChoice(boardState, validActions)
        if(chosen):
            return choice
        else:
            #print("Warning: Random choice taken")
            return random.choice(validActions)

episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.1)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = [s for s in batch.next_state if s is not None]
    if(len(non_final_next_states) == 0):
        print("Can't train!")
        return
    non_final_next_states = torch.cat(non_final_next_states)
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

if(AI):
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        num_episodes = 600
    else:
        print(torch.version.cuda)
        raise Exception("No GPU")
        num_episodes = 50

    for i_episode in range(num_episodes):
        # Initialize the environment and get its state
        state = Board()
        for t in count():
            stateTensor = torch.tensor(state.toList(), dtype=torch.float32, device=device).unsqueeze(0)
            actionNum = select_action(state)
            #observation, reward, terminated, truncated, _ = env.step(action.item())

            
            action = numToAction(actionNum)
            success, naiveReward = placePiece(state, action[1], action[2], action[3], action[4])
            
            reward = state.lastScore
            rewardNum = reward
            reward = torch.tensor([reward], device=device)
            terminated = state.finished
            done = terminated
            observation = state.toList()
            

            if terminated or (not success):
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            action = np.ndarray(shape=(1, 1,), dtype=float)
            action[0][0] = actionNum
            action = torch.tensor(action, device=device, dtype=torch.long)
            memory.push(stateTensor, action, next_state, reward)

            # Move to the next state done automatically

            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done or (not success):
                episode_durations.append(state.score)
                if(i_episode % 2 == 0):
                    plot_durations()
                    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
                    print(eps_threshold)
                    print(steps_done)
                reset()
                break

    print('Complete')
    plot_durations(show_result=True)
    plt.ioff()
    plt.show()

