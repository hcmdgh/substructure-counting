import torch 
from torch import Tensor 
import torch_geometric.utils as pyg_util 


def count_substructure(
    edge_index: Tensor,
    num_nodes: int,
    substructure_set: set[str],
) -> dict[str, Tensor]:
    assert edge_index.ndim == 2 and edge_index.shape[0] == 2

    edge_index = pyg_util.to_undirected(edge_index=edge_index, num_nodes=num_nodes)
    edge_index = pyg_util.remove_self_loops(edge_index=edge_index)
