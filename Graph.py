import dgl
import torch
from Processor import Processor
from Task import Task
import numpy as np

def index_of_task_id(task_list, task):
    for i in range(len(task_list)):
        if task.instance_id == task_list[i].instance_id:
            return i

    raise Exception("Mapped task was not found in previous task list")



#Converts the state of the OS to a heterogenous graph
#Three types of nodes: Previous Task, Processor, Future Task
#Three types of connections: Processing, Not Processing, Can Process
#Outputs in DGL heterogenous format
def create_task_graph(previous_tasks, ready_tasks, pset, tp_mapping):

    #Create node start and end lists that fully connect previous tasks to all processors

    # start nodes (tasks)
    np_u = np.arange(len(previous_tasks))
    np_u = np.repeat(np_u, len(pset))

    # end nodes (processors)
    np_v = np.arange(len(pset))
    np_v = np.tile(np_v, len(previous_tasks))


    # print("Fully Connected Start Nodes")
    # print(np_u)
    # print("Fully Connected End Nodes")
    # print(np_v)
    # print()

    #Create processing only graph
    p_u = [] #tasks
    p_v = [] #processors

    np_remove_indices = [] #track which tasks/processors and mapped to remove them from np_u, np_v graph

    #Only create links that indicate a processor is mapped to a task
    for mapping in tp_mapping:
        task_index = index_of_task_id(previous_tasks, mapping["task"])
        processor_index = pset.index(mapping["processor"])

        p_u.append(task_index)
        p_v.append(processor_index)

        np_remove_indices.append(task_index * len(pset) + processor_index)

    #Delete mapped tasks/processors to get a graph for all unmapped tasks
    np_u = np.delete(np_u, np_remove_indices)
    np_v = np.delete(np_v, np_remove_indices)

    #Link defined graph connecting tasks that are not mapped to a given processor
    # print("Not Processing Start Nodes")
    # print(np_u)
    # print("Not Processing End Nodes")
    # print(np_v)
    # print()
    #
    # #Link defined graph connected tasks that are mapped to a processor
    # print("Processing Start Nodes")
    # print(p_u)
    # print("Processing End Nodes")
    # print(p_v)
    # print()

    #Create graph to connect all processors to ready tasks that it could process

    # start nodes (processors)
    wp_u = np.arange(len(pset))
    wp_u = np.repeat(wp_u, len(ready_tasks))

    # end nodes (ready_tasks)
    wp_v = np.arange(len(ready_tasks))
    wp_v = np.tile(wp_v, len(pset))

    # print("Will Process Start Nodes")
    # print(wp_u)
    # print("Will Process End Nodes")
    # print(wp_v)
    # print()

    #Create heterograph that links tasks to processors which two types of edges, processing and not processing
    graph_data_1 = {("previous_task", "not_processing", "processor"): (np_u,np_v), ("previous_task","processing", "processor"):(p_u, p_v), ("processor", "will_process", "ready_task"):(wp_u, wp_v)}

    #add reverse edges to make it undirected
    graph_data_2 = {("processor", "not_processing_r", "previous_task"): (np_v,np_u), ("processor","processing_r", "previous_task"):(p_v, p_u), ("ready_task", "will_process_r", "processor"):(wp_v, wp_u)}

    graph_data = graph_data_1 | graph_data_2


    graph = dgl.heterograph(graph_data)

    #Add node and edge features
    #Tasks - execution time, deadline
    #Processors - power

    #Features for previous tasks

    features = torch.zeros(len(previous_tasks), 3)
    for i, task in enumerate(previous_tasks):
        features[i][0] = task.exec_time
        features[i][1] = task.period
        features[i][2] = task.deadline

    graph.nodes["previous_task"].data["features"] = features

    #Features for ready tasks
    features = torch.zeros(len(ready_tasks), 3)
    for i, task in enumerate(ready_tasks):
        features[i][0] = task.exec_time
        features[i][1] = task.period
        features[i][2] = task.deadline

    graph.nodes["ready_task"].data["features"] = features

    #Features for processors
    features = torch.zeros(len(pset), 3)
    for i, processor in enumerate(pset):
        features[i][0] = processor.power

    graph.nodes["processor"].data["features"] = features

    return graph


if __name__ == "__main__":
    # Create set of processors
    pset = Processor.create_homogeneous_pset(3, 1)

    # Create set of tasks
    task1 = Task.create_non_runnable(10, 5)
    task2 = Task.create_non_runnable(20, 6)
    task3 = Task.create_non_runnable(30, 7)
    previous_tasks = [Task.create_runnable(task1), Task.create_runnable(task2), Task.create_runnable(task3)]

    for task in previous_tasks:
        task.period-=1

    ready_tasks = previous_tasks


    tp_mapping = [{"processor":pset[0], "task":previous_tasks[0]}, {"processor":pset[1], "task":previous_tasks[1]}, {"processor":pset[2], "task":previous_tasks[2]}]

    graph = create_task_graph(previous_tasks, ready_tasks, pset, tp_mapping)
    print(graph)
    print(f"Types of nodes: {graph.ntypes}")
    print(f"Types of edges: {graph.etypes}")
    print(f"Edge Resitrictions: {graph.canonical_etypes}")

    print(graph.nodes["previous_task"])
    print(graph.nodes["ready_task"])
    print(graph.nodes["processor"])


