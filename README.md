# Substructure Counting

统计简单无向图中**每个节点参与的子结构数量**，支持 12 种三至五节点子结构，以及 induced（诱导）和非 induced 两种计数模式。核心计算使用 Numba 和 CSR 稀疏邻接表，输出 PyTorch Tensor。

## 安装与运行

依赖：

- PyTorch
- PyTorch Geometric（`torch-geometric`）
- NumPy
- Numba

在已有 PyTorch 的 Python 环境中补齐其他依赖：

```bash
python -m pip install torch-geometric numpy numba
```

如果尚未安装 PyTorch，先安装适合当前平台的 PyTorch。本项目没有配置安装包，以下示例请在仓库根目录运行，或将仓库根目录加入 `PYTHONPATH`。

## 快速开始

图中，节点 `0、1、2` 构成三角形，节点 `3` 连接节点 `0`，节点 `4` 为孤立节点。每条无向边只提供一个方向即可。

```python
import torch
from substructure_counting import count_substructure

edge_index = torch.tensor([
    [0, 1, 2, 0],
    [1, 2, 0, 3],
], dtype=torch.int64)

counts = count_substructure(
    edge_index=edge_index,
    num_nodes=5,
    substructure_set={"triangle", "3-star"},
    induced=False,
)

print(counts["triangle"])
# tensor([1, 1, 1, 0, 0])

print(counts["3-star"])
# tensor([1, 1, 1, 1, 0])
```

这里的 `3-star` 以节点 `0` 为中心、节点 `1、2、3` 为叶子。中心和三个叶子都参与这个子结构，因此各加一次。

改为诱导计数后，叶子 `1、2` 之间的额外边会使这个星形不符合要求：

```python
induced_counts = count_substructure(
    edge_index=edge_index,
    num_nodes=5,
    substructure_set={"triangle", "3-star"},
    induced=True,
)

print(induced_counts["triangle"])
# tensor([1, 1, 1, 0, 0])

print(induced_counts["3-star"])
# tensor([0, 0, 0, 0, 0])
```

## 接口

```python
def count_substructure(
    edge_index: torch.Tensor,
    num_nodes: int,
    substructure_set: set[str],
    induced: bool,
) -> dict[str, torch.Tensor]:
    ...
```

| 参数 | 说明 |
| --- | --- |
| `edge_index` | 整数 Tensor，形状为 `[2, num_edges]`，推荐 `torch.int64`。节点编号范围为 `[0, num_nodes)`。 |
| `num_nodes` | 节点总数，包含孤立节点，必须为非负整数。 |
| `substructure_set` | 要计算的子结构名称集合，见下表。空集合返回 `{}`。 |
| `induced` | `True` 要求所选节点之间的边恰好组成目标结构；`False` 允许额外边。 |

返回字典的键为请求的子结构名称。每个值：

- 形状为 `[num_nodes]`，dtype 为 `torch.int64`。
- 第 `i` 个元素表示节点 `i` 参与的该类子结构数量，不区分中心、叶子等角色。
- 位于与输入 `edge_index` 相同的设备上。
- 没有参与该类结构的节点取值为 `0`。

公开接口会将输入边转为无向边、合并重复边并移除自环，因此可传入单向列出的无向边，也可传入双向边。底层 `codex_impl.count_substructure` 本身不接受自环；通常应使用上述公开接口。

## 支持的子结构

下表的节点编号只用于描述结构，与输入图的实际编号无关。`12` 表示节点 `1` 与节点 `2` 之间有边。

| 名称 | 节点数 | 边数 | 定义 |
| --- | ---: | ---: | --- |
| `triangle` | 3 | 3 | 三角形：`12, 13, 23`。 |
| `3-star` | 4 | 3 | 三叶星：`12, 13, 14`。 |
| `4-cycle` | 4 | 4 | 四环：`12, 23, 34, 41`。 |
| `diamond` | 4 | 5 | 四环加一条对角线：`12, 23, 34, 41, 13`。 |
| `4-clique` | 4 | 6 | 四个节点两两相连，即 K4。 |
| `4-star` | 5 | 4 | 四叶星：`12, 13, 14, 15`。 |
| `5-cycle` | 5 | 5 | 五环：`12, 23, 34, 45, 51`。 |
| `banner` | 5 | 5 | 四环接一个悬挂节点：`12, 23, 34, 41, 15`。 |
| `bow-tie` | 5 | 6 | 两个三角形恰好共享一个节点：`12, 13, 23, 14, 45, 51`。 |
| `biclique` | 5 | 6 | 完全二部图 K2,3，两侧为 `{1, 2}` 和 `{3, 4, 5}`。 |
| `wheel` | 5 | 8 | 四环加一个连接所有环节点的中心：`12, 23, 34, 41, 51, 52, 53, 54`。 |
| `5-clique` | 5 | 10 | 五个节点两两相连，即 K5。 |

计算全部类型：

```python
all_substructures = {
    "triangle", "3-star", "4-cycle", "diamond", "4-clique",
    "4-star", "5-cycle", "banner", "bow-tie", "biclique",
    "wheel", "5-clique",
}

counts = count_substructure(
    edge_index=edge_index,
    num_nodes=5,
    substructure_set=all_substructures,
    induced=False,
)
```

## 计数口径

### 非 induced：按不同边集合计数

允许所选节点之间存在额外边。使用不同边集合的子结构分别计数；仅改变同一个子结构的节点命名、环的起点或行进方向，不会重复计数。

例如，K4 的四个节点记为 A、B、C、D，其中有三个四环：

1. A → B → C → D → A
2. A → B → D → C → A
3. A → C → B → D → A

因此，K4 中每个节点的非 induced `4-cycle` 计数都是 `3`，即使它们使用同一组四个节点。

### Induced：所选节点之间不允许额外边

只检查所选节点之间的边，连接到所选集合之外的边不影响匹配。同一组节点若诱导出目标结构，只计一次。

K4 中的四个节点有六条边，因此无法诱导出只有四条边的四环。对应示例：

```python
from itertools import combinations

k4_edges = torch.tensor(list(combinations(range(4), 2)), dtype=torch.int64).T

for induced in (False, True):
    result = count_substructure(
        edge_index=k4_edges,
        num_nodes=4,
        substructure_set={"4-cycle"},
        induced=induced,
    )
    print(induced, result["4-cycle"].tolist())

# False [3, 3, 3, 3]
# True [0, 0, 0, 0]
```

### 从逐节点计数得到全图数量

每个子结构会给它包含的每个节点各加一次。因此，对逐节点计数求和后，除以该结构的节点数，即为全图子结构数量：

```python
num_triangles = counts["triangle"].sum().item() // 3
num_four_cycles = counts["4-cycle"].sum().item() // 4
```
