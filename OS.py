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
        self.task_counter = 0

        #metrics
        self.context_switches = 0
        self.deadlines_missed = 0

    #calling run simulates 1 time step
    #each time step, do the following
    #map tasks to processors
    #add tasks to ready list
    #map tasks to processors
    #run the tasks on the processors

    def calc_reward(self):

        if len(self.ready_tasks) == 0:
            return 0

        reward = 0
        deadlines_missed = 0

        #Check for missed deadlines and reduce reward
        for task in self.ready_tasks:
            if task.deadline >= 0:
                reward +=5
            else:
                reward -=20
                deadlines_missed+=1

        #Check for unused prcoessses
        #unused_processors = len(self.pset) - len(self.tp_mapping)
        #unmapped_tasks = len(self.ready_tasks) - len(self.tp_mapping)
        #could_map = min(unused_processors, unmapped_tasks) #calculate tasks that could have been mapped

        #Check for context switches
        context_switches = 0
        for p in self.pset:
            if p.context_switched:
                context_switches+=1
        reward -= context_switches #add small penalty for context switching

        #update OS metrics
        self.context_switches += context_switches
        self.deadlines_missed+=deadlines_missed

        return reward/len(self.ready_tasks) #normalize by number of ready tasks

    def reset(self):
        self.ready_tasks = []
        self.previous_tasks = []
        self.time = 0
        self.tp_mapping = []
        self.context_switches = 0

        for p in self.pset:
            p.context_switched = False

        for t in self.taskset:
            self.ready_tasks.append(Task.create_runnable(t, self.task_counter))
            self.task_counter+=1

        return (self.previous_tasks, self.ready_tasks, self.pset, self.tp_mapping)

    def step(self, action):
        self.time+=1

        #Decode input action into a mapping between processors and ready tasks
        self.tp_mapping = []
        if action is not None:
            used_tasks = []
            for pset_i, ready_task_i in enumerate(action):
                if ready_task_i not in used_tasks: #Check if another processor is mapped to the ready task, if it is then skip.
                    self.tp_mapping.append({"processor":self.pset[pset_i], "task":self.ready_tasks[ready_task_i]})
                    used_tasks.append(ready_task_i)

        self.previous_tasks = deepcopy(self.ready_tasks)

        #First schedule tasks that are ready to run
        for t in self.taskset:
            # at each period for the task schedule it
            if self.time % t.period == 0:
                self.ready_tasks.append(Task.create_runnable(t, self.task_counter))
                self.task_counter+=1

        #Now execute mapped tasks
        for tp_map in self.tp_mapping:
            if tp_map["task"] is not None:
                task_done = tp_map["processor"].process(tp_map["task"]) #execute tasks ie decreases its execution time
                if task_done is True: #if the task is fully processed then remove it from the ready_tasks list
                    self.ready_tasks.remove(tp_map["task"])
                    #may need to remove from mapping too

        #Decrement deadline for all tasks
        for task in self.ready_tasks:
            task.deadline-=1

        reward = self.calc_reward() #calculate reward

        # print(f"Time Step: {self.time}")
        # print(self.ready_tasks)
        # print(self.tp_mapping)
        # print(f"OS reward {reward}")
        # print("")

        if len(self.ready_tasks) == 0:
            return None, 0 #state, reward

        return (self.previous_tasks, self.ready_tasks, self.pset, self.tp_mapping), reward #state, reward





