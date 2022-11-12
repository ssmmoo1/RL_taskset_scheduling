import dgl
import torch
from Processor import Processor
from Task import Task
import numpy as np


#Converts the state of the OS to a heterogenous graph
#Two types of nodes: Task and Processor
#Two types of connections: Processing and Not Processing
#Outputs in DGL heterogenous format
def create_task_graph(ready_tasks, pset, tp_mapping):

    #node_ID is equal to the task/processor index in ready_tasks and pset

    #Create node start and end lists that fully connect tasks to all processors

    # start nodes (tasks)
    np_u = np.arange(len(ready_tasks))
    np_u = np.repeat(np_u, len(pset))

    # end nodes (processors)
    np_v = np.arange(len(pset))
    np_v = np.tile(np_v, len(ready_tasks))


    print("Fully Connected Start Nodes")
    print(np_u)
    print("Fully Connected End Nodes")
    print(np_v)
    print()

    #Create processing only graph
    p_u = [] #tasks
    p_v = [] #processors

    np_remove_indices = [] #track which tasks/processors and mapped to remove them from np_u, np_v graph

    #Only create links that indicate a processor is mapped to a task
    for mapping in tp_mapping:
        task_index = ready_tasks.index(mapping["task"])
        processor_index = pset.index(mapping["processor"])

        p_u.append(task_index)
        p_v.append(processor_index)

        np_remove_indices.append(task_index * len(pset) + processor_index)

    #Delete mapped tasks/processors to get a graph for all unmapped tasks
    np_u = np.delete(np_u, np_remove_indices)
    np_v = np.delete(np_v, np_remove_indices)


    #Convert start and end node lists to tensors
    #Link defined graph connecting tasks that are not mapped to a given processor
    np_u = torch.tensor(np_u)
    np_v = torch.tensor(np_v)

    print("Not Processing Start Nodes")
    print(np_u)
    print("Not Processing End Nodes")
    print(np_v)
    print()

    #Link defined graph connected tasks that are mapped to a processor
    p_u = torch.tensor(p_u)
    p_v = torch.tensor(p_v)

    print("Processing Start Nodes")
    print(p_u)
    print("Processing Start Nodes")
    print(p_v)
    print()

    #Create heterograph that links tasks to processors which two types of edges, processing and not processing
    graph_data = {("task", "not_processing", "processor"): (np_u,np_v), ("task","processing", "processor"):(p_u, p_v)}
    graph = dgl.heterograph(graph_data)

    #Add node and edge features
    #Tasks - execution time, deadline
    #Processors - power


    execution_times = []
    deadlines = []
    for task in ready_tasks:
        execution_times.append(task.exec_time)
        deadlines.append(task.period)

    powers = []
    for processor in pset:
        powers.append(processor.power)


    graph.nodes["task"].data["execution_time"] = torch.tensor(execution_times)
    graph.nodes["task"].data["deadline"] = torch.tensor(deadlines)
    graph.nodes["processor"].data["power"] = torch.tensor(powers)

    return graph


if __name__ == "__main__":
    # Create set of processors
    pset = Processor.create_homogeneous_pset(3, 1)

    # Create set of tasks
    task1 = Task.create_non_runnable(10, 5)
    task2 = Task.create_non_runnable(20, 6)
    task3 = Task.create_non_runnable(30, 7)
    ready_tasks = [Task.create_runnable(task1), Task.create_runnable(task2), Task.create_runnable(task3)]

    tp_mapping = [{"processor":pset[0], "task":ready_tasks[0]}, {"processor":pset[1], "task":ready_tasks[1]}, {"processor":pset[2], "task":ready_tasks[2]}]

    graph = create_task_graph(ready_tasks, pset, tp_mapping)
    print(graph)
    print(f"Types of nodes: {graph.ntypes}")
    print(f"Types of edges: {graph.etypes}")
    print(f"Edge Resitrictions: {graph.canonical_etypes}")

    print(graph.nodes["task"])

    print(graph.nodes["processor"])
