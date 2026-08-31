"""Level 1 outcome model 数据集定义。

按 dataset_role 过滤 split，10 维输入 + 3 维输出。
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch




# LEVEL1_OUTCOME_INPUT_FIELDS: tuple[str, ...] = ()
# Python 的类型注解（Type Hint） 语法，具体拆解如下：
#     1. LEVEL1_OUTCOME_INPUT_FIELDS — 变量名
#     2. : tuple[str, ...] — 类型注解.
#         2.1 [str, ...] 是 泛型参数，说明：元组中的元素类型是 str（字符串）
#         2.2 ...（省略号）表示元组长度可变，不固定
#         2.3  = () — 赋值
# 含义：创建一个名为 LEVEL1_OUTCOME_INPUT_FIELDS 的变量，它是一个空元组，但类型注解告诉开发者/工具：这个元组预期应该存放字符串元素。


BASE_LEVEL1_OUTCOME_INPUT_FIELDS: tuple[str, ...] = (
    "contact_point_local_x",
    "contact_point_local_y",
    "contact_normal_local_x",
    "contact_normal_local_y",
    "force_direction_local_x",
    "force_direction_local_y",
    "commanded_force_N",
    "ramp_up_s",
    "hold_s",
    "ramp_down_s",
)
"""当前 baseline forward model 使用的 10 维 action feature。"""

LEVEL1_CONDITION_FEATURE_FIELDS: tuple[str, ...] = (
    "hidden_object_table_sliding_friction",
    "hidden_object_table_torsional_friction",
    "hidden_object_table_rolling_friction",
)
"""后续 condition-conditioned forward model 可显式输入的物理条件 feature。"""

LEVEL1_OUTCOME_INPUT_FIELDS: tuple[str, ...] = BASE_LEVEL1_OUTCOME_INPUT_FIELDS
"""默认输入字段保持 10 维，避免改变现有 baseline 训练语义。"""

LEVEL1_CONDITIONED_OUTCOME_INPUT_FIELDS: tuple[str, ...] = (
    *BASE_LEVEL1_OUTCOME_INPUT_FIELDS,
    *LEVEL1_CONDITION_FEATURE_FIELDS,
)
"""预留 13 维输入：action feature + object-table friction 条件。"""

LEVEL1_OUTCOME_TARGET_FIELDS: tuple[str, ...] = (
    "delta_x",
    "delta_y",
    "delta_yaw",
)


# 1. csv_path: str | Path — 联合类型（Union Type）：
#       csv_path 参数可以是 str（字符串） 或 Path（路径对象）
#       | 是 Python 3.10+ 引入的联合类型操作符，表示"或"
# 2. -> list[dict[str, str]] — 返回值类型注解
#       函数返回一个列表（list）
#       列表中的每个元素是一个字典（dict）
#       字典的键（key）是字符串，值（value）也是字符串

def load_level1_rows(csv_path: str | Path) -> list[dict[str, str]]:
    """读取 CSV 文件，返回所有行。"""
    import csv

    with open(str(csv_path), "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


# 参数中的逗号可有可无
def rows_to_arrays(
    rows: list[dict[str, str]],
    input_fields: tuple[str, ...] = LEVEL1_OUTCOME_INPUT_FIELDS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    将行列表转换为 X (N,D) 和 y (N,3)。
    :param rows: csv文件读出来的数据
    :param input_fields: forward model 输入字段
    :return: x是输入特征，y是 delta_x/delta_y/delta_yaw 三个标签
    """
    n = len(rows)
    x = np.zeros((n, len(input_fields)), dtype=np.float32)
    y = np.zeros((n, len(LEVEL1_OUTCOME_TARGET_FIELDS)), dtype=np.float32)
    for i, row in enumerate(rows):
        for j, field in enumerate(input_fields):
            x[i, j] = float(row[field])
        y[i, 0] = float(row["delta_x"])
        y[i, 1] = float(row["delta_y"])
        y[i, 2] = float(row["delta_yaw"])
    return x, y


class Level1OutcomeDataset(torch.utils.data.Dataset):
    """PyTorch Dataset，按 dataset_role 过滤，支持标准化。"""

    def __init__(
        self,
        csv_path: str | Path,
        dataset_role: str,
        normaliser: Any | None = None,
        input_fields: tuple[str, ...] = LEVEL1_OUTCOME_INPUT_FIELDS,
    ):
        all_rows = load_level1_rows(csv_path)
        rows = [r for r in all_rows if r.get("dataset_role", "") == dataset_role]
        if not rows:
            # !r 在 f-string 中表示 repr()，输出带引号：dataset_role='train'
            raise ValueError(f"No rows found for dataset_role={dataset_role!r}")
        self.input_fields = tuple(input_fields)
        self.x_raw, self.y = rows_to_arrays(rows, self.input_fields)
        if normaliser is not None:
            # 如果提供了标准化器，对输入特征进行标准化（如归一化到均值为0，标准差为1）
            self.x = normaliser.transform(self.x_raw)
        else:
            self.x = self.x_raw.copy()

    def __len__(self) -> int:
        # 返回数据集样本数量
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # 根据索引 idx 返回一个样本
        # self.x[idx]是 NumPy数组（形状：(n_features,)）
        # torch.from_numpy()转换为PyTorchTensor
        # 返回元组：(输入Tensor, 目标Tensor)
        return torch.from_numpy(self.x[idx]), torch.from_numpy(self.y[idx])
