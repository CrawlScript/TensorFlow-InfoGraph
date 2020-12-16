# coding=utf-8
import os

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

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
num_classes = np.max(graph_labels) + 1

train_graphs, test_graphs = train_test_split(graphs, test_size=0.1)

batch_size = 50


class GNN(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gcn0 = tfg.layers.GCN(512, activation=tf.nn.relu)
        self.gcn1 = tfg.layers.GCN(512, activation=tf.nn.relu)
        self.fc = tf.keras.Sequential([
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(num_classes)
        ])

    def call(self, inputs, training=None, mask=None):
        x, edge_index, node_graph_index = inputs
        h = self.gcn0([x, edge_index], training=training)
        h = self.gcn1([h, edge_index], training=training)
        graph_h = tfg.nn.sum_pool(h, node_graph_index)
        logits = self.fc(graph_h, training=training)
        return logits


model = GNN()


def create_data_generator(graphs, shuffle=False, batch_size=50):
    dataset = tf.data.Dataset.range(len(graphs))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)

    for batch_index in dataset:
        batch_graphs = [graphs[i] for i in batch_index]
        yield batch_graphs


def evaluate():
    y_true = np.concatenate([graph.y for graph in test_graphs], axis=0)
    batch_preds_list = []
    for step, batch_graphs in enumerate(create_data_generator(test_graphs, shuffle=False)):
        batch_graph = tfg.BatchGraph.from_graphs(batch_graphs)
        logits = model([batch_graph.x, batch_graph.edge_index, batch_graph.node_graph_index])
        batch_preds = tf.argmax(logits, axis=-1)
        batch_preds_list.append(batch_preds)

    y_pred = tf.concat(batch_preds_list, axis=0).numpy()

    accuracy = accuracy_score(y_true, y_pred)
    return accuracy

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

for epoch in range(100000):
    for step, batch_graphs in enumerate(create_data_generator(train_graphs, shuffle=True)):
        with tf.GradientTape() as tape:
            batch_graph = tfg.BatchGraph.from_graphs(batch_graphs)
            logits = model([batch_graph.x, batch_graph.edge_index, batch_graph.node_graph_index], training=True)
            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits,
                labels=tf.one_hot(batch_graph.y, depth=num_classes)
            )
            loss = tf.reduce_mean(losses)
        vars = tape.watched_variables()
        grads = tape.gradient(loss, vars)
        optimizer.apply_gradients(zip(grads, vars))

        if step == 0 and epoch % 10 == 0:
            test_accuracy = evaluate()
            print("epoch = {}\tstep = {}\tloss = {}\ttest_accuracy = {}"
                  .format(epoch, step, loss, test_accuracy))
