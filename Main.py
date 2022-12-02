from Processor import Processor
from Scheduler import Scheduler, EDF_Scheduler, Model_Scheduler
from Task import Task, TaskSet
from OS import OS
from time import time
from Model import TaskGCN
import torch
from collections import deque
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from Graph import create_task_graph
from random import randint
import pickle
from tqdm import tqdm

#RL and Graph Constants
RELATIONS =  ["processing", "not_processing", "will_process","processing_r", "not_processing_r", "will_process_r"]
eps = np.finfo(np.float32).eps.item()
GAMMA = 0.99
MODEL_PATH = "model_bad.pth"
NUM_EPISODES = 300
TIME_STEPS = 50
EP_RESET = True

#Given the gnn model and state of OS, returns mapping of processors to ready tasks
def select_action(model, state):
    pt_feats = state.nodes["previous_task"].data["features"]
    pc_feats = state.nodes["processor"].data["features"]
    rt_feats = state.nodes["ready_task"].data["features"]
    node_features = {"previous_task":pt_feats, "processor":pc_feats, "ready_task":rt_feats}
    edge_probs = model(state, node_features, "will_process")
    m = Categorical(edge_probs)
    action = m.sample()
    model.saved_log_probs.append(m.log_prob(action))

    return action.tolist() #index is the processor, value is the index of the ready task to run

#Calculates and applies change in weights based on saved action and reward buffers
def finish_episode(model, optimizer):
    R = 0
    policy_loss = []
    returns = deque()
    for r in model.rewards[::-1]:
        R = r + GAMMA * R
        returns.appendleft(R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(model.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del model.rewards[:]
    del model.saved_log_probs[:]

    return policy_loss


#Trains a gnn model
def main_train_gnn(os, model, optimizer):
    running_reward = 0
    x = list(range(NUM_EPISODES))
    rewards = []
    missed_deadlines = []
    context_switches = []

    state = os.reset()

    for episode in range(NUM_EPISODES):

        ep_reward = 0

        if EP_RESET is True:
            state = os.reset()

        for t in range(1,TIME_STEPS):
            #if no tasks are ready then don't run the scheduler
            if state is None:
                state, reward = os.step(None)
                continue

            state = create_task_graph(*state)
            action = select_action(model, state) #returns index of ready task to run
            state, reward = os.step(action)
            if reward is None:
                continue
            model.rewards.append(reward)
            ep_reward += reward

        running_reward = 0.05 * ep_reward + (1-0.05) * running_reward
        finish_episode(model, optimizer)

        Model_Scheduler.model = model
        model.eval()
        reward, md, cs = inference(Model_Scheduler)
        rewards.append(reward)
        missed_deadlines.append(md)
        context_switches.append(cs)
        model.train()

        print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
            episode, ep_reward, running_reward))

    torch.save(model.state_dict(), MODEL_PATH)

    return rewards, missed_deadlines, context_switches

    # plt.subplot(1,2,1)
    # plt.plot(x, rewards)
    # plt.title("Rewards")
    # plt.subplot(1, 2, 2)
    # plt.plot(x, missed_deadlines)
    # plt.title("Missed Deadlines")
    #
    # plt.show()"

#simulate a task set using the given scheduler function and return performance statistics
def inference(scheduler, os, time_steps):

    state = os.reset()
    rewards = 0
    for t in range(1, time_steps):

        action = None
        if state is not None:
            action = scheduler(*state)

        state, reward = os.step(action)

        rewards+= reward

    print(f"Total rewards: {rewards}, Missed Deadlines: {os.deadlines_missed}, Context Switches: {os.context_switches}")
    return rewards, os.deadlines_missed, os.context_switches


def main_compare():
    inference(EDF_Scheduler, os, TIME_STEPS)

    model = TaskGCN(3, 8, 4, RELATIONS, mlp=True)

    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    Model_Scheduler.model = model

    inference(Model_Scheduler, os, TIME_STEPS)


def plot_training(rewards, deadlines, context_switches):
    plt.subplot(1,3,1)
    plt.plot(list(range(len(rewards))), rewards)
    plt.title("Rewards")
    plt.subplot(1, 3, 2)
    plt.plot(list(range(len(deadlines))), deadlines)
    plt.title("Missed Deadlines")
    plt.subplot(1, 3, 3)
    plt.plot(list(range(len(context_switches))), context_switches)
    plt.title("Missed Deadlines")
    plt.show()

def main_search():
    plot_counter = 1
    plt.figure(figsize=(16,200))
    for hidden_feats in range(2,11,2):
        for out_feats in range(1,5):
            for mlp in [True, False]:
                model = TaskGCN(2, hidden_feats, out_feats, RELATIONS, mlp=mlp)
                optimizer = optim.Adam(model.parameters(), lr=1e-3)
                rewards, missed_deadlines = main_train_gnn(model, optimizer)

                plt.subplot(100,2,plot_counter)
                plt.plot(list(range(NUM_EPISODES)), rewards)
                plt.title(f"Rewards:Hidden Feats {hidden_feats}, Out Feats {out_feats}, MLP {mlp}")
                plt.subplot(100, 2, plot_counter+1)
                plt.plot(list(range(NUM_EPISODES)), missed_deadlines)
                plt.title(f"Missed Deadlines:Hidden Feats {hidden_feats}, Out Feats {out_feats}, MLP {mlp}")


                model.eval()
                Model_Scheduler.model = model
                inference(Model_Scheduler)

                plot_counter +=4

    plt.savefig("compare_plots.png")

def train_episode(os, time_steps, model, optimizer):
    state = os.reset()
    ep_reward = 0

    for t in range(1, time_steps):
        # if no tasks are ready then don't run the scheduler
        if state is None:
            state, reward = os.step(None)
            continue

        state = create_task_graph(*state)
        action = select_action(model, state)  # returns index of ready task to run
        state, reward = os.step(action)
        if reward is None:
            continue
        model.rewards.append(reward)
        ep_reward += reward

    finish_episode(model, optimizer)

#single core
def experiment_1():
    model_path = "trained_model.pth"
    train_path = "small_datasets/small_train.pkl"
    test_path = "small_datasets/small_test.pkl"
    epochs = 2

    #load taskets
    with open(train_path, "rb") as file:
        train_task_sets = pickle.load(file)

    # load taskets
    with open(test_path, "rb") as file:
        test_task_sets = pickle.load(file)

    #create model
    model = TaskGCN(3, 8, 4, RELATIONS, mlp=True)
    #model.load_state_dict(torch.load(model_path))
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    #Training Stats
    test_rewards = []
    test_md = []
    test_cs = []

    #Train model on train set
    for epoch in range(epochs): #each epoch train on all task_sets

        for i, task_set in enumerate(train_task_sets):
            #create OS
            os = OS(task_set.tasks, Processor.create_homogeneous_pset(1,1))
            train_episode(os, task_set.lcm+1, model, optimizer)
            print(f"Epoch {epoch}, Step {i}/{len(train_task_sets)}")


            Model_Scheduler.model = model
            print(f"Epoch {epoch} Starting Validation")
            model.eval()
            reward, md, cs = benchmark(test_task_sets, Model_Scheduler, Processor.create_homogeneous_pset(1,1))
            model.train()
            print(f"Epoch {epoch} Finished Validation")
            test_rewards.append(reward)
            test_md.append(md)
            test_cs.append(cs)
            torch.save(model.state_dict(), model_path)
            print(f"Epoch {epoch} saved model checkpoint")

        print(f"Epoch {epoch} completed")

    model.eval()
    #Benchmark trained model and heuristic
    reward_m, md_m, cs_m = benchmark(test_task_sets, Model_Scheduler, Processor.create_homogeneous_pset(1,1))
    reward_h, md_h, cs_h = benchmark(test_task_sets, EDF_Scheduler, Processor.create_homogeneous_pset(1, 1))
    print(f"Trained Model Based Benchmark\n\tRewards: {reward_m} \n\tMissed Deadlines: {md_m}\n\t Context Switches: {cs_m}")
    print(f"EDF Benchmark\n \t Rewards: {reward_h} \n\tMissed Deadlines: {md_h}\n\t Context Switches: {cs_h}")

    plot_training(test_rewards, test_md, test_cs)

def benchmark(tasksets, scheduler, pset):
    # with open(dataset_path, "rb") as file:
    #     tasksets = pickle.load(file)

    #TaskSet.plot_tasksets_stats(tasksets)

    total_reward = 0
    total_md = 0
    total_cs = 0
    for i in tqdm(range(len(tasksets))):
        taskset = tasksets[i]
        #print(taskset.cpu_util())
        os = OS(taskset.tasks, pset)
        reward, md, cs = inference(scheduler, os, taskset.lcm+1)
        total_reward+=reward
        total_md+= md
        total_cs += cs


    print(f"Benchmark Stats: Reward:{total_reward}, Missed Deadlines: {total_md}, Context Switches: {total_cs}")
    return total_reward, total_md, total_cs

if __name__ == "__main__":
    # model = TaskGCN(3, 8, 4, RELATIONS, mlp=True)
    # model.load_state_dict(torch.load(MODEL_PATH))
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # rewards, deadlines = main_train_gnn(model, optimizer)
    # plot_training(rewards, deadlines)
    # main_compare()
    #main_search()
    #main_tasksets()

    #benchmark("small_datasets/small_train.pkl", EDF_Scheduler, Processor.create_homogeneous_pset(1,1))
    main_train_benchmark()