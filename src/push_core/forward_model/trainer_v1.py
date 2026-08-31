"""训练循环与统一评估。"""

import math  # 导入数学库，用于角度转换（弧度转度）
from dataclasses import dataclass  # 导入数据类装饰器，用于创建 Metrics 类
from pathlib import Path  # 导入路径处理库，用于创建保存模型的目录

import torch  # PyTorch 主库
import torch.nn as nn  # 导入神经网络模块


@dataclass  # 自动生成 __init__、__repr__ 等方法
class Metrics:
    """多指标评估结果。"""

    mse_total: float  # 总均方误差（所有4个输出维度）
    mse_planar: float  # 平面位移的均方误差（仅 dx, dy）
    mae_dx: float  # x方向位移的平均绝对误差
    mae_dy: float  # y方向位移的平均绝对误差
    mae_planar: float  # 平面位移向量的平均绝对误差（欧氏距离）
    mae_yaw_deg: float  # 偏航角的平均绝对误差（单位：度）


def evaluate_metrics(
        model: nn.Module,  # 要评估的 PyTorch 模型
        loader: torch.utils.data.DataLoader,  # 数据加载器（验证集或测试集）
        device: torch.device,  # 计算设备（CPU 或 CUDA）
) -> Metrics:  # 返回 Metrics 对象
    """在 DataLoader 上计算全部指标。yaw 误差直接比较 unwrapped delta_yaw。"""

    model.eval()  # 切换到评估模式（关闭 Dropout、BatchNorm 使用全局统计量）
    n = 0  # 初始化样本计数器
    total_loss = 0.0  # 初始化总 MSE 累加器
    planar_loss = 0.0  # 初始化平面位移 MSE 累加器
    sum_ae_dx = 0.0  # 初始化 dx 绝对误差累加器
    sum_ae_dy = 0.0  # 初始化 dy 绝对误差累加器
    sum_ae_planar = 0.0  # 初始化平面位移绝对误差累加器
    sum_ae_yaw = 0.0  # 初始化偏航角绝对误差累加器

    with torch.no_grad():  # 禁用梯度计算（节省内存，加速推理）
        for xb, yb in loader:  # 遍历数据加载器中的每个 batch
            xb, yb = xb.to(device), yb.to(device)  # 将数据移到指定设备
            pred = model(xb)  # 前向传播：得到模型预测值
            batch_n = xb.shape[0]  # 获取当前 batch 的样本数
            n += batch_n  # 累加总样本数

            # MSE total (3 维)
            # 计算 delta_x、delta_y、delta_yaw 三个输出维度的均方误差
            total_loss += nn.MSELoss()(pred, yb).item() * batch_n
            # .item() 将 Tensor 转为 Python 标量
            # * batch_n 是为了后续计算加权平均

            # MSE planar (dx, dy)
            # 只计算前 2 个维度（dx, dy）的均方误差
            planar_loss += nn.MSELoss()(pred[:, :2], yb[:, :2]).item() * batch_n

            # MAE dx, dy
            # 计算 x 方向位移的绝对误差并求和
            sum_ae_dx += torch.abs(pred[:, 0] - yb[:, 0]).sum().item()
            # 计算 y 方向位移的绝对误差并求和
            sum_ae_dy += torch.abs(pred[:, 1] - yb[:, 1]).sum().item()

            # MAE planar displacement
            # 计算平面位移向量的欧氏距离误差
            diff_planar = torch.norm(pred[:, :2] - yb[:, :2], dim=1)
            # torch.norm(..., dim=1) 计算每个样本的 2D 向量长度
            sum_ae_planar += diff_planar.sum().item()

            # MAE yaw
            # 第 3 维直接是 unwrapped delta_yaw，不做 sin/cos 反解或 wrap。
            err = pred[:, 2] - yb[:, 2]
            sum_ae_yaw += torch.abs(err).sum().item()

    # 返回包含所有指标的对象
    return Metrics(
        mse_total=total_loss / n,  # 计算平均总 MSE
        mse_planar=planar_loss / n,  # 计算平均平面位移 MSE
        mae_dx=sum_ae_dx / n,  # 计算平均 dx 绝对误差
        mae_dy=sum_ae_dy / n,  # 计算平均 dy 绝对误差
        mae_planar=sum_ae_planar / n,  # 计算平均平面位移绝对误差
        mae_yaw_deg=sum_ae_yaw / n * 180.0 / math.pi,  # 弧度转度
    )


def train_model(
        model: nn.Module,  # 要训练的模型
        train_loader: torch.utils.data.DataLoader,  # 训练数据加载器
        val_loader: torch.utils.data.DataLoader,  # 验证数据加载器
        epochs: int,  # 训练轮数
        lr: float,  # 学习率
        weight_decay: float,  # 权重衰减（L2 正则化系数）
        device: torch.device,  # 计算设备
        ckpt_path: str | None = None,  # 模型保存路径（可选）
) -> None:  # 无返回值
    """训练模型，跟踪最佳 val 指标并保存 checkpoint。"""
    # 函数文档字符串

    model.to(device)  # 将模型移到指定设备（CPU/GPU）

    # 创建优化器（AdamW：带权重衰减的 Adam）
    optimizer = torch.optim.AdamW(
        model.parameters(),  # 要优化的模型参数
        lr=lr,  # 学习率
        weight_decay=weight_decay  # L2 正则化系数
    )

    loss_fn = nn.MSELoss()  # 定义损失函数：均方误差
    best_val = float("inf")  # 初始化最佳验证损失为正无穷大

    for epoch in range(epochs):  # 遍历每个训练轮次
        model.train()  # 切换到训练模式（启用 Dropout、BatchNorm 使用 batch 统计量）

        for xb, yb in train_loader:  # 遍历训练数据加载器
            xb, yb = xb.to(device), yb.to(device)  # 将数据移到指定设备
            optimizer.zero_grad()  # 清空上一步的梯度（防止累积）
            loss = loss_fn(model(xb), yb)  # 前向传播 + 计算损失
            loss.backward()  # 反向传播：计算梯度
            optimizer.step()  # 更新模型参数

        # 每个 epoch 结束后在验证集上评估模型
        val_metrics = evaluate_metrics(model, val_loader, device)

        # 如果当前验证 MSE 比历史最佳更好，则保存模型
        if val_metrics.mse_total < best_val:
            best_val = val_metrics.mse_total  # 更新最佳值

            if ckpt_path:  # 如果指定了保存路径
                # 创建父目录（如果不存在）
                Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
                # 保存模型状态字典（只保存权重，不保存结构）
                torch.save(model.state_dict(), ckpt_path)

        # 每 50 个 epoch 或第 1 个 epoch 时打印进度
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(
                f"  epoch {epoch + 1}/{epochs}  "  # 当前 epoch / 总 epoch
                f"val_mse={val_metrics.mse_total:.6f}  "  # 验证集 MSE（6位小数）
                f"val_yaw_deg={val_metrics.mae_yaw_deg:.3f}"  # 验证集偏航误差（度，3位小数）
            )

    # 训练结束后打印最终最佳结果
    print(f"  best_val_mse={best_val:.6f}  ckpt={ckpt_path}")
