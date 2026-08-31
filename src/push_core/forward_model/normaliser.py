"""标准化工具：基于 train split 的 X 计算 mean/std，支持 JSON 序列化。"""

from dataclasses import dataclass

import numpy as np

# 这是数据标准化器的完整实现，用于机器学习中的特征标准化。
# 1. @dataclass — 数据类装饰器;自动生成 __init__、__repr__、__eq__ 等方法
# Z-Score 标准化: 标准化后的值 = (原始值 - 平均值) / 标准差

@dataclass
class StandardNormaliser:
    """Standard normaliser: (x - mean) / std，std < 1e-8 的列设为 1.0。"""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "StandardNormaliser":
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        std[std < 1e-8] = 1.0
        return cls(mean=mean, std=std)  # cls(...) 就是在调用这个类本身，用来创建一个该类的实例对象。

    def transform(self, x: np.ndarray) -> np.ndarray:
        '''
        Z-Score 标准化:标准化后的值 = (原始值 - 平均值) / 标准差
        :param x: 每一个特征值，NumPy 的数组， N 维数组（N-dimensional array）
        :return: 标准化之后的值
        '''
        return (x - self.mean) / self.std

    def to_json_dict(self) -> dict:
        '''
        序列化为字典;将 NumPy 数组转换为 Python 列表，方便 JSON 序列化
        .tolist() 将 np.ndarray 转为嵌套列表
        :return: 字典
        '''
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_json_dict(cls, data: dict) -> "StandardNormaliser":
        '''
        从字典反序列化;从 JSON 字典恢复 StandardNormaliser 对象;
        指定 dtype=np.float32 节省内存（PyTorch 默认 float32）
        :param data: 字典
        :return: StandardNormaliser 对象
        '''
        return cls(
            mean=np.array(data["mean"], dtype=np.float32),
            std=np.array(data["std"], dtype=np.float32),
        )
