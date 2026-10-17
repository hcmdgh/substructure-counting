import torch 
from torch import Tensor 
import torch_geometric.utils as pyg_util 


def count_substructure(
    edge_index: Tensor,
    num_nodes: int,
    substructure_set: set[str],
) -> dict[str, Tensor]:
    pass 
