import pickle
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import DataLoader
from torch_geometric.nn import TransformerConv
from matplotlib import pyplot as plt

from graph_embedding import Py150KDataset

DEVICE = torch.device("cpu")


class GNNConvolution(nn.Module):
    def __init__(
        self,
        input_dim: int,
        edge_dim: int,
        hidden_dims: Union[int, List[int], None],
        output_dim: int,
        dropout: float = 0.25,
        layer_norm: bool = True,
    ):
        super(GNNConvolution, self).__init__()
        if hidden_dims is None:
            hidden_dims = []
        elif not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims]
        self.num_layers = len(hidden_dims) + 1
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.layer_norm = layer_norm
        self.lns = nn.ModuleList()

        i_dim = input_dim
        for i in range(len(hidden_dims)):
            self.convs.append(self.build_conv_layer(i_dim, edge_dim, hidden_dims[i]))
            if self.layer_norm:
                self.lns.append(nn.LayerNorm(hidden_dims[i]))
            i_dim = hidden_dims[i]
        self.convs.append(self.build_conv_layer(i_dim, edge_dim, output_dim))

        # this loss is useful for multi-label classifier
        # self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.ones([output_dim]))

    def build_conv_layer(self, input_dim, edge_dim, hidden_dim):
        return TransformerConv(input_dim, hidden_dim, edge_dim=edge_dim)

    def forward(self, node_attr, edge_index, edge_attr):
        for i in range(self.num_layers):
            node_attr = self.convs[i](node_attr, edge_index, edge_attr)
            node_attr = F.relu(node_attr)
            node_attr = F.dropout(node_attr, p=self.dropout, training=self.training)
            if self.layer_norm and i < self.num_layers - 1:
                node_attr = self.lns[i](node_attr)
        return F.log_softmax(node_attr, dim=1)

    def loss(self, pred, labels, variable_mask):
        loss_val = F.nll_loss(pred[variable_mask], labels[variable_mask])
        return loss_val


def batch_predict(model, batch):
    with torch.no_grad():
        node_attr, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        pred = model(node_attr, edge_index, edge_attr)
        pred_class = pred[batch.variables_mask].argmax(dim=1)
    return pred_class


def predict_for_loader(model, test_loader):
    preds = [batch_predict(model, batch).detach().numpy() for batch in test_loader]
    return np.hstack(preds)


if __name__ == "__main__":

    data = Py150KDataset()

    data_size = len(data)
    train_loader = DataLoader(data[: int(data_size * 0.8)], batch_size=16, shuffle=True)
    test_loader = DataLoader(data[int(data_size * 0.8) :], batch_size=16, shuffle=True)

    input_dim = data.num_node_features
    edge_dim = data.num_edge_features
    output_dim = data.num_classes
    hidden_dims = [16] * 5

    with open("label_encoder.pckl", "rb") as file:
        label_encoder: LabelEncoder = pickle.load(file)

    model = GNNConvolution(input_dim, edge_dim, hidden_dims, output_dim)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    # train
    for epoch in range(200):
        total_loss = 0
        model.train()
        for batch in train_loader:
            opt.zero_grad()
            node_attr, edge_index, edge_attr = (
                batch.x,
                batch.edge_index,
                batch.edge_attr,
            )
            pred = model(node_attr, edge_index, edge_attr)
            label = batch.y
            loss = model.loss(pred, label, batch.variables_mask)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(train_loader.dataset)

        if epoch % 10 == 0:
            model.eval()
            y_pred = predict_for_loader(model, test_loader)
            y_true = np.hstack([batch.y.detach().numpy()[batch.variables_mask] for batch in test_loader])
            unique_labels = np.unique(np.array([y_true, y_pred]).flatten()).sort()
            cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=unique_labels)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
            disp.plot()
            plt.show(block=False)
            plt.pause(3)
            plt.close()
            print(f"Epoch {epoch}. " f"Loss: {total_loss:.4f}. ")
