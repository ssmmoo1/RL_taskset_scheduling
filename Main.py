from Processor import Processor
from Scheduler import Scheduler, EDF_Scheduler
from Task import Task
from OS import OS
from time import time
from Model import TaskGCN
import torch
from collections import deque
import torch.optim as optim
import numpy as np

#Create set of processors
pset = Processor.create_homogeneous_pset(1,1)

#Create set of tasks
task1 = Task.create_non_runnable(10, 5)
task2 = Task.create_non_runnable(20, 5)
task3 = Task.create_non_runnable(30, 5)
tset = [task1,task2,task3]
#Create OS
os = OS(tset, pset)


model = TaskGCN(2, 10, 5, ["processing", "not_processing", "will_process"])
optimizer = optim.Adam(model.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()

def select_action(state):
    edge_scores = model(state)

    #TODO convert edge scores to probablities

    return [{0:0, 1:1, 2:2}] #map processsor IDs to task IDs

def finish_episode():
    R = 0
    policy_loss = []
    returns = deque()
    for r in model.rewards[::-1]:
        R = r + 0.01 * R
        returns.appendleft(R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(model.saved_log_probs, returns):
        policy_loss.append(-log_prob * R) #TODO determine if shape of log_prob will get messed up with variable action space
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del model.rewards[:]
    del model.saved_log_probs[:]


def main():
    running_reward = 10
    for episode in range(10):

        state = os.reset()
        ep_reward = 0

        for t in range(1,10000):
            action = select_action(state)
            state, reward = os.step(action)
            model.rewards.append(reward)
            ep_reward += reward

        running_reward = 0.05 * ep_reward + (1-0.05) * running_reward
        finish_episode()

        print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
            episode, ep_reward, running_reward))