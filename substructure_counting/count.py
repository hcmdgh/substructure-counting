import torch 
from torch import Tensor 
import torch_geometric.utils as pyg_util 

from . import codex_impl 


def count_substructure(
    edge_index: Tensor,
    num_nodes: int,
    substructure_set: set[str],
    induced: bool,
) -> dict[str, Tensor]:
    assert edge_index.ndim == 2 and edge_index.shape[0] == 2

    edge_index = pyg_util.to_undirected(edge_index=edge_index, num_nodes=num_nodes)
    edge_index, _ = pyg_util.remove_self_loops(edge_index=edge_index)

    return codex_impl.count_substructure(
        edge_index = edge_index,
        num_nodes = num_nodes,
        substructure_set = substructure_set,
        induced = induced,
    )
