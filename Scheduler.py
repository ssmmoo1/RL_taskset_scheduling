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

    min_period = ready_tasks[0].period
    min_index = 0
    for i in range(len(ready_tasks)):
        if ready_tasks[i].period < min_period:
            min_period = ready_tasks[i].period
            min_index = i

    return min_index

def Model_Scheduler(previous_tasks, ready_tasks, pset, tp_mapping):
    state = create_task_graph(previous_tasks, ready_tasks, pset, tp_mapping)
    pt_feats = state.nodes["previous_task"].data["features"]
    pc_feats = state.nodes["processor"].data["features"]
    rt_feats = state.nodes["ready_task"].data["features"]
    node_features = {"previous_task": pt_feats, "processor": pc_feats, "ready_task": rt_feats}
    edge_probs = Model_Scheduler.model(state, node_features, "will_process")
    return torch.argmax(edge_probs).item()