from Processor import Processor
from Scheduler import Scheduler, EDF_Scheduler, Model_Scheduler
from Task import generate_datasets
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
from os import makedirs

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
def reload_training_plot():
    save_path = "experiment_3/"
    with open(save_path + "rewards.pkl", "rb") as file:
        rewards = pickle.load(file)
    with open(save_path + "deadlines.pkl", "rb") as file:
        deadlines = pickle.load(file)
    with open(save_path + "context_switches.pkl", "rb") as file:
        context_switches = pickle.load(file)

    plot_training(rewards, deadlines, context_switches)

def plot_training(rewards, deadlines, context_switches, save_path=None):
    plt.clf()

    plt.subplot(1,3,1)
    plt.grid(visible=True)
    plt.plot(list(range(len(rewards))), rewards)
    plt.title("Rewards")
    plt.xlabel("Epochs")
    plt.ylabel("Reward")

    plt.subplot(1, 3, 2)
    plt.grid(visible=True)
    plt.plot(list(range(len(deadlines))), deadlines)
    plt.xlabel("Epochs")
    plt.ylabel("Missed Deadlines")
    plt.title("Missed Deadlines")

    plt.subplot(1, 3, 3)
    plt.grid(visible=True)
    plt.plot(list(range(len(context_switches))), context_switches)
    plt.title("Context Switches")
    plt.xlabel("Epochs")
    plt.ylabel("Context Switches")

    plt.tight_layout()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path+"training_plots.png")
        with open(save_path+"rewards.pkl", "wb") as file:
            pickle.dump(rewards, file)
        with open(save_path+"deadlines.pkl", "wb") as file:
            pickle.dump(deadlines, file)
        with open(save_path+"context_switches.pkl", "wb") as file:
            pickle.dump(context_switches, file)


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
def experiment_1(epochs):
    model_path = "experiment_1/trained_model.pth"
    train_path = "datasets/single_core_train_set.pkl"
    test_path = "datasets/single_core_test_set.pkl"

    processors = Processor.create_homogeneous_pset(1, 1)

    #load taskets
    with open(train_path, "rb") as file:
        train_task_sets = pickle.load(file)

    # load taskets
    with open(test_path, "rb") as file:
        test_task_sets = pickle.load(file)

    #create model
    model = TaskGCN(3, 8, 4, RELATIONS, mlp=True)
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
            os = OS(task_set.tasks, processors)
            train_episode(os, task_set.lcm+1, model, optimizer)
            print(f"Epoch {epoch}, Step {i}/{len(train_task_sets)}")


            Model_Scheduler.model = model
            print(f"Epoch {epoch} Starting Validation")
            model.eval()
            reward, md, cs = benchmark(test_task_sets, Model_Scheduler, processors)
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
    reward_m, md_m, cs_m = benchmark(test_task_sets, Model_Scheduler, processors)
    reward_h, md_h, cs_h = benchmark(test_task_sets, EDF_Scheduler, processors)
    print(f"Trained Model Based Benchmark\n\tRewards: {reward_m} \n\tMissed Deadlines: {md_m}\n\t Context Switches: {cs_m}")
    print(f"EDF Benchmark\n \t Rewards: {reward_h} \n\tMissed Deadlines: {md_h}\n\t Context Switches: {cs_h}")

    with open("experiment_1/benchmark_result.csv", "w+") as file:
        file.write("scheduler, reward, missed_deadlines, context_switches\n")
        file.write(f"trained model, {reward_m}, {md_m}, {cs_m}\n")
        file.write(f"EDF, {reward_h}, {md_h}, {cs_h}\n")

    plot_training(test_rewards, test_md, test_cs, save_path="experiment_1/")

def experiment_2(epochs):
    model_path = "experiment_2/trained_model.pth"
    train_path = "datasets/homogeneous_quad_train_set.pkl"
    test_path = "datasets/homogeneous_quad_test_set.pkl"

    processors = Processor.create_homogeneous_pset(4,1)

    #load taskets
    with open(train_path, "rb") as file:
        train_task_sets = pickle.load(file)

    # load taskets
    with open(test_path, "rb") as file:
        test_task_sets = pickle.load(file)

    #create model
    model = TaskGCN(3, 8, 4, RELATIONS, mlp=True)
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
            os = OS(task_set.tasks, processors)
            train_episode(os, task_set.lcm+1, model, optimizer)
            print(f"Epoch {epoch}, Step {i}/{len(train_task_sets)}")


        Model_Scheduler.model = model
        print(f"Epoch {epoch} Starting Validation")
        model.eval()
        reward, md, cs = benchmark(test_task_sets, Model_Scheduler, processors)
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
    reward_m, md_m, cs_m = benchmark(test_task_sets, Model_Scheduler, processors)
    reward_h, md_h, cs_h = benchmark(test_task_sets, EDF_Scheduler, processors)
    print(f"Trained Model Based Benchmark\n\tRewards: {reward_m} \n\tMissed Deadlines: {md_m}\n\t Context Switches: {cs_m}")
    print(f"EDF Benchmark\n \t Rewards: {reward_h} \n\tMissed Deadlines: {md_h}\n\t Context Switches: {cs_h}")

    with open("experiment_2/benchmark_result.csv", "w+") as file:
        file.write("scheduler, reward, missed_deadlines, context_switches\n")
        file.write(f"trained model, {reward_m}, {md_m}, {cs_m}\n")
        file.write(f"EDF, {reward_h}, {md_h}, {cs_h}\n")

    plot_training(test_rewards, test_md, test_cs,save_path="experiment_2/")


def experiment_3(epochs):
    model_path = "experiment_3/trained_model.pth"
    train_path = "datasets/heterogeneous_train_set.pkl"
    test_path = "datasets/heterogeneous_test_set.pkl"

    processors = Processor.create_hetero_pset([2,2,2,2,1,1,1,1])

    #load taskets
    with open(train_path, "rb") as file:
        train_task_sets = pickle.load(file)

    # load taskets
    with open(test_path, "rb") as file:
        test_task_sets = pickle.load(file)

    #create model
    model = TaskGCN(3, 8, 4, RELATIONS, mlp=True)
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
            os = OS(task_set.tasks, processors)
            train_episode(os, task_set.lcm+1, model, optimizer)
            print(f"Epoch {epoch}, Step {i}/{len(train_task_sets)}")


        Model_Scheduler.model = model
        print(f"Epoch {epoch} Starting Validation")
        model.eval()
        reward, md, cs = benchmark(test_task_sets, Model_Scheduler, processors)
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
    reward_m, md_m, cs_m = benchmark(test_task_sets, Model_Scheduler, processors)
    reward_h, md_h, cs_h = benchmark(test_task_sets, EDF_Scheduler, processors)
    print(f"Trained Model Based Benchmark\n\tRewards: {reward_m} \n\tMissed Deadlines: {md_m}\n\t Context Switches: {cs_m}")
    print(f"EDF Benchmark\n \t Rewards: {reward_h} \n\tMissed Deadlines: {md_h}\n\t Context Switches: {cs_h}")

    #Save benchmark data to csv file
    with open("experiment_3/benchmark_result.csv", "w+") as file:
        file.write("scheduler, reward, missed_deadlines, context_switches\n")
        file.write(f"trained model, {reward_m}, {md_m}, {cs_m}\n")
        file.write(f"EDF, {reward_h}, {md_h}, {cs_h}\n")



    plot_training(test_rewards, test_md, test_cs,save_path="experiment_3/")

def benchmark_schedulers():
    with open("datasets/homogeneous_quad_test_set.pkl", "rb") as file:
        task_sets = pickle.load(file)

    model = TaskGCN(3, 8, 4, RELATIONS, mlp=True)
    model.load_state_dict(torch.load("trained_model.pth"))
    model.eval()
    Model_Scheduler.model = model

    reward_h, md_h, cs_h = benchmark(task_sets, EDF_Scheduler, Processor.create_homogeneous_pset(4, 1))
    reward_m, md_m, cs_m = benchmark(task_sets, Model_Scheduler, Processor.create_homogeneous_pset(4, 1))
    print(f"Trained Model Based Benchmark\n\tRewards: {reward_m} \n\tMissed Deadlines: {md_m}\n\t Context Switches: {cs_m}")
    print(f"EDF Benchmark\n \t Rewards: {reward_h} \n\tMissed Deadlines: {md_h}\n\t Context Switches: {cs_h}")


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


def run_experiments():
    #make directories for output data
    makedirs("datasets", exist_ok=True) #folder for full datasets
    makedirs("experiment_1", exist_ok=True)
    makedirs("experiment_2", exist_ok=True)
    makedirs("experiment_3", exist_ok=True)

    generate_datasets("datasets/", 80, 20) #generate train and test sets for 3 experiments with 80 train sets, 20 test sets

    experiment_1(5) #single core
    experiment_2(50) #quad core homogeneous
    experiment_3(50) #octacore heterogeneous


if __name__ == "__main__":
    run_experiments()