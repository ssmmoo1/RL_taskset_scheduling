from Task import Task
from Scheduler import Scheduler


def calc_reward(ready_tasks):
    reward = 0
    for task in ready_tasks:
        if task.period <=0:
            reward-=1
        else:
            reward+=1
    return reward

class OS:
    #Prememption will occur at every time step so time slice will always be 1
    def __init__(self, taskset, pset, scheduler):
        self.scheduler = scheduler
        self.taskset = taskset #list of readonly tasks that define the task set
        self.pset = pset    #list of processor objects
        self.tp_mapping = [] #array of diciontaries {"processor":p, "task":t}
        self.ready_tasks = [] #hold tasks that are ready to run but not mapped
        self.time = 0

    #calling run simulates 1 time step
    #each time step, do the following
    #map tasks to processors
    #add tasks to ready list
    #map tasks to processors
    #run the tasks on the processors

    def step(self):

        #First schedule tasks that are ready to run
        for t in self.taskset:
            # at each period for the task schedule it
            if self.time % t.period == 0:
                self.ready_tasks.append(Task.create_runnable(t))



        #Now call scheduler to map tasks to processors
        self.tp_mapping = self.scheduler(self.ready_tasks, self.pset, self.tp_mapping)


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

        reward = calc_reward(self.ready_tasks) #calculate reward based on deadlines being misssed

        print(f"Time Step: {self.time}")
        print(self.ready_tasks)
        print(self.tp_mapping)
        print(f"OS reward {reward}")
        print("")

        self.time+=1






