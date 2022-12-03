from dataclasses import dataclass
import math
from random import randint
import pickle
import numpy as np
from random import sample
import matplotlib.pyplot as plt
from time import time

@dataclass
class Task:
    period: int
    exec_time: int
    deadline: int
    runnable: bool
    instance_id: int

    # Returns Task object that is runnable (ie exec_time is modifiable to simulate execution)
    # Takes in a non-runnable task an input
    @classmethod
    def create_runnable(cls, task, instance_id):
        return cls(task.period, task.exec_time, task.period, True, instance_id)

    # Returns a Task object that is not runnable, exec_time is fixed and is used to generate runnable tasks by the OS
    @classmethod
    def create_non_runnable(cls, period, exec_time):
        return cls(period, exec_time, -1, False, -1)

    @staticmethod
    def task_lcm(tasks):
        periods = [t.period for t in tasks]
        return math.lcm(*periods)

    @staticmethod
    def generate_task(max_period, min_period=1, min_exec_time=1):
        period = randint(min_period,max_period)
        exec_time = randint(1,period)

        return Task.create_non_runnable(period,exec_time)



@dataclass
class TaskSet:
    tasks: list[Task]
    lcm: int

    @staticmethod
    # max_lcm is inclusive
    def generate_taskset_random(num_tasks, max_lcm=1000):
        tasks = []
        done = False
        while not done:
            tasks = [Task.generate_task(randint(1, max_lcm)) for x in range(num_tasks)]
            if Task.task_lcm(tasks) <= max_lcm:
                done = True

        return tasks

    def cpu_util(self):
        util = 0
        for t in self.tasks:
            util+=t.exec_time / t.period

        return util

    #util is a percentage 1-100 inclusive
    @staticmethod
    def generate_taskset(util, min_tasks=3, max_tasks=10, max_lcm=200):
        possible_periods = []
        for i in range(2,max_lcm+1):
            if max_lcm % i == 0:
                possible_periods.append(i)

        util = min(100, util)
        numtasks = randint(min_tasks, max_tasks)
        task_utils = np.random.dirichlet(np.ones(numtasks))

        task_utils*=util
        task_utils = np.round(task_utils)
        task_utils = list(task_utils[task_utils != 0])


        tasks = []
        for t in task_utils:
            done = False
            period = None
            exec_time = None
            while not done:
                period = sample(possible_periods,1)[0]
                exec_time = int(period * (t/100))
                if exec_time != 0:
                    done = True
            tasks.append(Task.create_non_runnable(period, exec_time))
        return TaskSet(tasks, Task.task_lcm(tasks))

    @classmethod
    def combine_tasksets(cls, tasksets):

        tasks = []
        for ts in tasksets:
            tasks.extend(ts.tasks)
        lcm = Task.task_lcm(tasks)

        return cls(tasks, lcm)

    @staticmethod
    def generate_mutlicore_taskset(util, min_tasks, max_tasks, max_lcm):
        num_tasksets = util//100
        leftover_util = util % 100

        task_sets = []
        for ts in range(num_tasksets):
            task_sets.append(TaskSet.generate_taskset(100, min_tasks, max_tasks, max_lcm))

        task_sets.append(TaskSet.generate_taskset(leftover_util, min_tasks, max_tasks, max_lcm))

        combined_ts = TaskSet.combine_tasksets(task_sets)
        print(f"Combined task set cpu util: {combined_ts.cpu_util()}")
        return combined_ts

    @staticmethod
    def plot_tasksets_stats(tasksets):
        utils = []
        lcms = []
        n_tasks = []

        for taskset in tasksets:
            utils.append(taskset.cpu_util())
            lcms.append(taskset.lcm)
            n_tasks.append(len(taskset.tasks))

        fig, axs = plt.subplots(1, 3)
        axs[0].hist(utils)
        axs[0].title.set_text("CPU Utilization %")
        axs[1].hist(lcms)
        axs[1].title.set_text("Task Period LCM")
        axs[2].hist(n_tasks)
        axs[2].title.set_text("Number of Tasks")
        plt.show()



def generate_task_dataset(num_tasksets, min_tasks, max_tasks, min_lcm, max_lcm, min_util, max_util):

    tsets = []
    start = time()
    for t in range(num_tasksets):
        tset = TaskSet.generate_mutlicore_taskset(randint(min_util, max_util), min_tasks=min_tasks, max_tasks=max_tasks, max_lcm=randint(min_lcm, max_lcm))
        tsets.append(tset)
        if len(tset.tasks) == 0:
            raise Exception("Zero length Task Set!")

    print(f"Generated {num_tasksets} in {time() -start} seconds ")

    return tsets

def main_test():
    t1 = Task(10,10)
    t2 = Task(20, 10)
    t3 = Task(30, 10)

    tasks = [t1,t2,t3]
    lcm = Task.task_lcm(tasks)
    print(lcm)


    tasks = TaskSet.generate_taskset(5, 1000)
    print(tasks)

def generate_datasets(folder_path, train_tasks, test_tasks):

    #single core train and test
    task_sets = generate_task_dataset(train_tasks, 3, 25, 100, 300, 100, 100)
    with open(folder_path + "single_core_train_set.pkl", "wb") as file:
        pickle.dump(task_sets, file)

    task_sets = generate_task_dataset(test_tasks, 3, 25, 100, 300, 100, 100)
    with open(folder_path +"single_core_test_set.pkl", "wb") as file:
        pickle.dump(task_sets, file)

    #multicore homogenous train and test
    task_sets = generate_task_dataset(train_tasks, 3, 25, 100, 300, 400, 400) #quad-core
    with open(folder_path +"homogeneous_quad_train_set.pkl", "wb") as file:
        pickle.dump(task_sets, file)

    task_sets = generate_task_dataset(test_tasks, 3, 25, 100, 300, 400, 400) #quad-core
    with open(folder_path +"homogeneous_quad_test_set.pkl", "wb") as file:
        pickle.dump(task_sets, file)

    #multicore heterogenous train and test
    task_sets = generate_task_dataset(train_tasks, 3, 25, 100, 300, 400, 800)
    with open(folder_path +"heterogeneous_train_set.pkl", "wb") as file:
        pickle.dump(task_sets, file)

    task_sets = generate_task_dataset(test_tasks, 3, 25, 100, 300, 400, 800)
    with open(folder_path +"heterogeneous_test_set.pkl", "wb") as file:
        pickle.dump(task_sets, file)



if __name__ == "__main__":

    generate_datasets("datasets/", 80, 20)
    generate_datasets("small_datasets/", 8, 2)

