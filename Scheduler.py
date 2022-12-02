from Graph import create_task_graph
import torch
def Scheduler(ready_tasks, pset, previous_mapping):

    tasks = ready_tasks.copy()
    new_mapping = []
    for p in pset:
        if len(tasks) != 0:
            t = tasks.pop()
            new_mapping.append({"processor":p, "task":t})

    return new_mapping

def EDF_Scheduler(previous_tasks, ready_tasks, pset, tp_mapping):
    tasks = ready_tasks.copy()
    tasks.sort(key=lambda x: x.period)
    #print(tasks)
    new_mapping = []
    for p in pset:
        if len(tasks) != 0:
            new_mapping.append(ready_tasks.index(tasks.pop(0)))

    return new_mapping

def Model_Scheduler(previous_tasks, ready_tasks, pset, tp_mapping):
    state = create_task_graph(previous_tasks, ready_tasks, pset, tp_mapping)
    pt_feats = state.nodes["previous_task"].data["features"]
    pc_feats = state.nodes["processor"].data["features"]
    rt_feats = state.nodes["ready_task"].data["features"]
    node_features = {"previous_task": pt_feats, "processor": pc_feats, "ready_task": rt_feats}
    edge_probs = Model_Scheduler.model(state, node_features, "will_process")
    action_list = []
    for eprob in edge_probs:
        action_list.append(torch.argmax(eprob).item())
    return action_list