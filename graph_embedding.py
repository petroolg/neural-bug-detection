import gzip
import json
import os
import pickle
from glob import iglob

import numpy as np
import torch
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_sparse import SparseTensor
from tqdm import tqdm

ALL_GRAPHS_DIR = "./graph-dataset"

EDGE_TYPES = [
    "CHILD",
    "NEXT",
    "NEXT_USE",
    "LAST_LEXICAL_USE",
    "OCCURRENCE_OF",
    "SUBTOKEN_OF",
    "COMPUTED_FROM",
    "RETURNS_TO",
]

# We've used only 2 error codes for Flake8: `F821`, `F841`
# In total we have 3 classes, None = no error
ERROR_TYPES = ["F821", "F841", None]


# Embedding for nodes
def ngram_embedding(word, n=3, size=2000):
    ngram_indices = {}
    for i in range(len(word) - n + 1):
        ngram = "".join(word[i + j] for j in range(n))
        idx = sum([(ord(c) << 16 * j) % size for j, c in enumerate(reversed(ngram))]) % size
        ngram_indices[idx] = ngram_indices.get(idx, 0) + 1
    return ngram_indices


def ngram_embedding_batch(word_list, n=3, size=2000):
    batch_indices, batch_values = [], []
    for i, word in enumerate(word_list):
        ngram_indices = ngram_embedding(word, n, size)
        batch_indices += [[i, j] for j in ngram_indices.keys()]
        batch_values += list(ngram_indices.values())
    emb = torch.sparse_coo_tensor(
        np.array(batch_indices).T,
        batch_values,
        (len(word_list), size),
        dtype=torch.float,
    )
    return emb


# Label encoder, only useful for multi-label classification
class OneHotLabelEncoder(OneHotEncoder):
    def transform(self, x):
        try:
            return super().transform(x)
        except ValueError:
            emb_len = len(self.get_feature_names())
            out = sparse.csr_matrix((1, emb_len), dtype=np.long)
            if not self.sparse:
                return out.toarray()
            else:
                return out


def scipy_csr_to_torch(csr, dtype):
    coo = csr.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse_coo_tensor(i, v, torch.Size(shape), dtype=dtype)


class Py150KDataset(InMemoryDataset):
    def __init__(self):
        super().__init__(".", transform=None, pre_transform=None)
        self.data, self.slices = torch.load("processed/data.pt")

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def process(self):

        edge_embedder = self.train_edge_embedder()
        label_encoder = self.train_label_encoder()
        all_graphs = self.load_all_graphs_file()

        data_list = []
        for graph in tqdm(all_graphs):
            node_emb = ngram_embedding_batch(graph["nodes"], n=3, size=500)
            edge_attr, edge_index = self.extract_connectivity(graph, edge_embedder)
            y = label_encoder.transform(graph["labels"])

            data_list.append(
                Data(
                    x=SparseTensor.from_torch_sparse_coo_tensor(node_emb).to_dense(dtype=torch.float32),
                    edge_index=torch.tensor(edge_index, dtype=torch.long),
                    edge_attr=SparseTensor.from_scipy(edge_attr).to_dense(dtype=torch.float32),
                    y=torch.tensor(y, dtype=torch.long),
                    variables_mask=torch.tensor(graph["variable_mask"], dtype=torch.bool),
                    num_nodes=len(graph["nodes"]),
                )
            )

        data, slices = self.collate(data_list)
        torch.save((data, slices), "processed/data.pt")

    def extract_connectivity(self, graph, edge_embedder):
        edges, edges_types = [], []
        for relationship, connections in graph["edges"].items():
            for parent, children in connections.items():
                edges += [(int(parent), child) for child in children]
                edges_types += [relationship for _ in children]
        edge_index = np.array(edges).T
        edge_attr = edge_embedder.transform([[edge_type] for edge_type in edges_types])
        return edge_attr, edge_index

    def load_all_graphs_file(self):
        all_graphs = []
        for all_graph_chunk_path in iglob(os.path.join(ALL_GRAPHS_DIR, "all-graphs*.jsonl.gz")):
            print(f"Loading graph file {all_graph_chunk_path}.")
            with gzip.open(all_graph_chunk_path, "rb") as file:
                for line in file.readlines():
                    all_graphs.append(json.loads(line))
        return all_graphs

    def train_label_encoder(self):
        # Label encoding: mapping from ERROR_TYPES to int
        label_encoder = LabelEncoder()
        label_encoder.fit(ERROR_TYPES)
        with open("label_encoder.pckl", "wb") as file:
            pickle.dump(label_encoder, file)
        return label_encoder

    def train_edge_embedder(self):
        # Embedding for edges: mapping from EDGE_TYPES to onehot vectors
        edge_embedder = OneHotEncoder(dtype=np.float32)
        edge_embedder.fit([[t] for t in EDGE_TYPES])
        return edge_embedder


if __name__ == "__main__":
    # pytorch-geometric Dataset loading
    Py150KDataset()
