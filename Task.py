from dataclasses import dataclass
import math
from random import randint

@dataclass
class Task:
    period: int
    exec_time: int
    runnable: bool

    # Returns Task object that is runnable (ie exec_time is modifiable to simulate execution)
    # Takes in a non-runnable task an input
    @classmethod
    def create_runnable(cls, task):
        return cls(task.period, task.exec_time, True)

    # Returns a Task object that is not runnable, exec_time is fixed and is used to generate runnable tasks by the OS
    @classmethod
    def create_non_runnable(cls, period, exec_time):
        return cls(period, exec_time, False)

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
    def generate_taskset(num_tasks, max_lcm=1000):
        tasks = []
        done = False
        while not done:
            tasks = [Task.generate_task(randint(1, max_lcm)) for x in range(num_tasks)]
            if Task.task_lcm(tasks) <= max_lcm:
                done = True

        return tasks

    @staticmethod
    # max_lcm is inclusive
    # feasible means sum of cpu utilization for each task must be less than number of processors
    # TODO implement this
    def generate_feasible_taskset(num_tasks, max_lcm=1000, processors=1):
        tasks = []
        done = False
        while not done:
            tasks = [Task.generate_task(randint(1, max_lcm)) for x in range(num_tasks)]
            if Task.task_lcm(tasks) <= max_lcm:
                done = True

        return TaskSet(tasks,)












if __name__ == "__main__":
    t1 = Task(10,10)
    t2 = Task(20, 10)
    t3 = Task(30, 10)

    tasks = [t1,t2,t3]
    lcm = Task.task_lcm(tasks)
    print(lcm)


    tasks = TaskSet.generate_taskset(5, 1000)
    print(tasks)