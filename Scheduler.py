def Scheduler(ready_tasks, pset, previous_mapping):

    tasks = ready_tasks.copy()
    new_mapping = []
    for p in pset:
        if len(tasks) != 0:
            t = tasks.pop()
            new_mapping.append({"processor":p, "task":t})

    return new_mapping

def EDF_Scheduler(ready_tasks, pset, previous_mapping):
    tasks = ready_tasks.copy()
    tasks.sort(key=lambda x: x.period)
    print(tasks)
    new_mapping = []
    for p in pset:
        if len(tasks) != 0:
            t = tasks.pop(0)
            new_mapping.append({"processor": p, "task": t})

    return new_mapping