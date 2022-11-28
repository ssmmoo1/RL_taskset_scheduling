from Task import Task
from Scheduler import Scheduler
from Graph import create_task_graph
from copy import deepcopy

class OS:
    #Prememption will occur at every time step so time slice will always be 1
    def __init__(self, taskset, pset):
        self.taskset = taskset #list of readonly tasks that define the task set
        self.pset = pset    #list of processor objects
        self.tp_mapping = [] #array of diciontaries {"processor":p, "task":t}
        self.ready_tasks = [] #hold tasks that are ready to run but not mapped
        self.previous_tasks = []
        self.time = 0

    #calling run simulates 1 time step
    #each time step, do the following
    #map tasks to processors
    #add tasks to ready list
    #map tasks to processors
    #run the tasks on the processors

    def calc_reward(self):

        if len(self.ready_tasks) == 0:
            return 0, 0

        reward = 0
        deadlines_missed = 0
        for task in self.ready_tasks:
            if task.period >= 0:
                reward +=1
            else:
                reward -=5
                deadlines_missed+=1

        return reward/len(self.ready_tasks), deadlines_missed

    def reset(self):
        self.ready_tasks = []
        self.previous_tasks = []
        self.time = 0
        self.tp_mapping = []
        for t in self.taskset:
            self.ready_tasks.append(Task.create_runnable(t))

        return (self.previous_tasks, self.ready_tasks, self.pset, self.tp_mapping)

    def step(self, action):
        self.time+=1

        if action is not None:
            self.tp_mapping = [{"processor":self.pset[0], "task":self.ready_tasks[action]}]#TODO need to add on to this to make it get the correct object references
        else:
            self.tp_mapping = []

        #self.previous_tasks = deepcopy(self.ready_tasks)
        self.previous_tasks = self.ready_tasks.copy()

        #First schedule tasks that are ready to run
        for t in self.taskset:
            # at each period for the task schedule it
            if self.time % t.period == 0:
                self.ready_tasks.append(Task.create_runnable(t))

        #Now execute mapped tasks
        for tp_map in self.tp_mapping:
            if tp_map["task"] is not None:
                task_done = tp_map["processor"].process(tp_map["task"]) #execute tasks ie decreases its execution time
                if task_done is True: #if the task is fully processed then remove it from the ready_tasks list
                    self.ready_tasks.remove(tp_map["task"])
                    #may need to remove from mapping too

        #Decrement deadline for all tasks
        for task in self.ready_tasks:
            task.period-=1

        reward, deadlines_missed = self.calc_reward() #calculate reward

        print(f"Time Step: {self.time}")
        print(self.ready_tasks)
        print(self.tp_mapping)
        print(f"OS reward {reward}")
        print("")

        if len(self.ready_tasks) == 0:
            return None, 0, 0

        return (self.previous_tasks, self.ready_tasks, self.pset, self.tp_mapping), reward, deadlines_missed





