from Processor import Processor
from Scheduler import Scheduler, EDF_Scheduler
from Task import Task
from OS import OS
from time import time

#Create set of processors
pset = Processor.create_homogeneous_pset(1,1)

#Create set of tasks
task1 = Task.create_non_runnable(10, 5)
task2 = Task.create_non_runnable(20, 5)
task3 = Task.create_non_runnable(30, 5)
tset = [task1,task2,task3]

#Create OS
os = OS(tset, pset, EDF_Scheduler)


st = time()
for i in range(61):

    os.step()
print(f"Simulation Time: {time() - st}")