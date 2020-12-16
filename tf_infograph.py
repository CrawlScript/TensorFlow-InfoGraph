# coding=utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tf_geometric as tfg
import tensorflow as tf
import numpy as np

graph_dicts = tfg.datasets.TUDataset("MUTAG").load_data()

graphs = [tfg.Graph(
    x=np.ones([len(graph_dict["node_labels"]), 1]),
    edge_index=graph_dict["edge_index"],
    y=graph_dict["graph_label"]
) for graph_dict in graph_dicts]

num_graphs = len(graphs)
graph_labels = np.concatenate([graph.y for graph in graphs], axis=0)

batch_size = 50

for batch_index in tf.data.Dataset.range(num_graphs).batch(batch_size):
    batch_graphs = [graphs[i] for i in batch_index]
    batch_graph = tfg.BatchGraph.from_graphs(batch_graphs)
    print(batch_graph)




