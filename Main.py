from Processor import Processor
from Scheduler import Scheduler, EDF_Scheduler, Model_Scheduler
from Task import Task
from OS import OS
from time import time
from Model import TaskGCN_MLP, TaskGCN_Dot
import torch
from collections import deque
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from Graph import create_task_graph



#Create set of processors
pset = Processor.create_homogeneous_pset(1,1)

#Create set of tasks
task1 = Task.create_non_runnable(10, 2)
task2 = Task.create_non_runnable(5, 3)
task3 = Task.create_non_runnable(3, 1)

tset = [task1,task2,task3]
#Create OS
os = OS(tset, pset)


model = TaskGCN_MLP(2, 5, 5, ["processing", "not_processing", "will_process","processing_r", "not_processing_r", "will_process_r"])
optimizer = optim.Adam(model.parameters(), lr=1e-3)
eps = np.finfo(np.float32).eps.item()
GAMMA = 0.99
MODEL_PATH = "model_bad.pth"
NUM_EPISODES = 2
TIME_STEPS = 100
EP_RESET = True

def select_action(state):
    pt_feats = state.nodes["previous_task"].data["features"]
    pc_feats = state.nodes["processor"].data["features"]
    rt_feats = state.nodes["ready_task"].data["features"]
    node_features = {"previous_task":pt_feats, "processor":pc_feats, "ready_task":rt_feats}
    edge_probs = model(state, node_features, "will_process")
    m = Categorical(edge_probs)
    action = m.sample()
    model.saved_log_probs.append(m.log_prob(action))
    return action.item()

def finish_episode():
    R = 0
    policy_loss = []
    returns = deque()
    for r in model.rewards[::-1]:
        R = r + GAMMA * R
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

    return policy_loss



def main_train_gnn():
    running_reward = 0
    x = list(range(NUM_EPISODES))
    rewards = []
    losses = []

    state = os.reset()

    for episode in range(NUM_EPISODES):

        ep_reward = 0

        if EP_RESET is True:
            state = os.reset()

        for t in range(1,TIME_STEPS):
            #if no tasks are ready then don't run the scheduler
            if state is None:
                state, reward, _ = os.step(None)
                continue

            state = create_task_graph(*state)
            action = select_action(state) #returns index of ready task to run
            state, reward, _ = os.step(action)
            if reward is None:
                continue
            model.rewards.append(reward)
            ep_reward += reward

        rewards.append(ep_reward)

        running_reward = 0.05 * ep_reward + (1-0.05) * running_reward
        loss = finish_episode()
        losses.append(loss.detach().numpy())



        print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
            episode, ep_reward, running_reward))

    torch.save(model.state_dict(), MODEL_PATH)

    plt.subplot(1,2,1)
    plt.plot(x, rewards)
    plt.title("Rewards")

    plt.subplot(1,2,2)
    plt.plot(x,losses)
    plt.title("Loss")
    plt.show()

def inference(scheduler):

    state = os.reset()
    deadlines_missed = 0
    rewards = 0
    for t in range(1, TIME_STEPS):

        action = None
        if state is not None:
            action = scheduler(*state)

        state, reward, deadline_missed = os.step(action)

        rewards+= reward
        deadlines_missed += deadline_missed

    print(f"Total rewards: {rewards}, Missed Deadlines: {deadlines_missed}")

def main_compare():
    inference(EDF_Scheduler)

    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    Model_Scheduler.model = model

    inference(Model_Scheduler)


if __name__ == "__main__":
    #main_train_gnn()
    main_compare()


