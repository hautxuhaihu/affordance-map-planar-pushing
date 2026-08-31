"""Level 1 outcome model 的第二版训练工具。

本文件不替换旧的 trainer_v1.py，而是新增一套更清晰的训练架构：
- 每个 epoch 同时记录训练集和验证集的 mse_total_scaled。
- 使用验证集 val_mse_total_scaled 选择最优模型参数。
- 当验证集指标连续若干 epoch 不再变好时提前停止。
- 返回可序列化的 history，供训练报告和可视化脚本使用。
"""

# 导入 Path，用于统一处理 checkpoint 路径。
import math
from pathlib import Path
# 导入 Any，用于标注 history 这种混合结构字典。
from typing import Any

# 导入 PyTorch 主包，用于张量、优化器和模型保存。
import torch
# 导入神经网络模块，用于 MSELoss 和 Module 类型标注。
import torch.nn as nn

# 复用旧 trainer_v1.py 中已经验证过的 Metrics 数据结构。
from push_core.forward_model.trainer_v1 import Metrics


def metrics_to_dict(metrics: Metrics) -> dict[str, float]:
    """将 Metrics 数据类转换为普通字典，方便写入 JSON。"""

    # JSON 不能直接稳定序列化 dataclass，这里显式展开字段。
    return {
        "mse_total": metrics.mse_total,
        "mse_planar": metrics.mse_planar,
        "mae_dx": metrics.mae_dx,
        "mae_dy": metrics.mae_dy,
        "mae_planar": metrics.mae_planar,
        "mae_yaw_deg": metrics.mae_yaw_deg,
    }


def make_target_scale(target_values: Any) -> torch.Tensor:
    """根据训练集 target 计算三维输出的尺度。

    target_values 应该是训练集 y，形状为 (N, 3)，三列分别对应：
    delta_x、delta_y、delta_yaw。训练时三个标签都使用训练集标准差归一化。
    """

    # 转成 float32 Tensor，后续可以直接移动到训练设备。
    target_tensor = torch.as_tensor(target_values, dtype=torch.float32)
    # 三个标签都按训练集标准差缩放，避免不同物理量量纲主导损失。
    scale = target_tensor.std(dim=0)
    # 极小标准差会导致除法不稳定；常量维度直接设为 1。
    scale[scale < 1e-8] = 1.0
    return scale


def evaluate_mse_total_scaled(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    target_scale: torch.Tensor,
) -> float:
    """计算经过 target 尺度归一化后的三维 MSE。

    这是 V2 训练和早停使用的监控指标；旧的 mse_total 仍由
    evaluate_metrics() 计算，并保留为原始尺度报告指标。
    """

    # 切换到评估模式，关闭训练态行为。
    model.eval()
    # 将三维尺度移动到同一设备，并整理成可广播形状。
    scale = target_scale.to(device).view(1, -1)
    # 在计算设备上累计，避免每个 batch 都同步回 CPU。
    sum_squared_error = torch.zeros((), dtype=torch.float64, device=device)
    # 记录参与平均的标量数量，即样本数乘以输出维度数。
    value_count = 0

    # 评估阶段不需要梯度。
    with torch.inference_mode():
        # 遍历 DataLoader 中的 batch。
        for xb, yb in loader:
            # 将输入和标签移动到同一设备。
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            # 得到模型预测。
            pred = model(xb)
            # 对每个输出维度按训练集尺度归一化误差。
            diff_scaled = (pred - yb) / scale
            # 累计平方误差。
            sum_squared_error += torch.sum(
                diff_scaled * diff_scaled,
                dtype=torch.float64,
            )
            # 累计标量数量。
            value_count += yb.numel()

    # 返回平均后的归一化三维 MSE。
    return float(sum_squared_error.item() / value_count)


def evaluate_metrics_with_scaled_mse(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    target_scale: torch.Tensor,
) -> tuple[Metrics, float]:
    """单遍计算原始物理指标和归一化 MSE。"""

    model.eval()
    scale = target_scale.to(device).view(1, -1)
    totals = torch.zeros(7, dtype=torch.float64, device=device)
    sample_count = 0

    with torch.inference_mode():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            diff = model(xb) - yb
            diff_planar = diff[:, :2]
            diff_scaled = diff / scale
            totals += torch.stack(
                (
                    torch.sum(diff * diff),
                    torch.sum(diff_planar * diff_planar),
                    torch.sum(torch.abs(diff[:, 0])),
                    torch.sum(torch.abs(diff[:, 1])),
                    torch.sum(torch.linalg.vector_norm(diff_planar, dim=1)),
                    torch.sum(torch.abs(diff[:, 2])),
                    torch.sum(diff_scaled * diff_scaled),
                )
            ).to(dtype=torch.float64)
            sample_count += int(yb.shape[0])

    values = totals.cpu().tolist()
    metrics = Metrics(
        mse_total=values[0] / (sample_count * 3),
        mse_planar=values[1] / (sample_count * 2),
        mae_dx=values[2] / sample_count,
        mae_dy=values[3] / sample_count,
        mae_planar=values[4] / sample_count,
        mae_yaw_deg=values[5] / sample_count * 180.0 / math.pi,
    )
    mse_total_scaled = values[6] / (sample_count * 3)
    return metrics, mse_total_scaled


def _train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    target_scale: torch.Tensor,
) -> None:
    """训练一个 epoch。

    这里不单独返回 batch 平均指标，因为 epoch 结束后会用完整
    train_loader 计算 train_mse_total_scaled。
    """

    # 将三维 target 尺度移动到同一设备，并整理成可广播形状。
    scale = target_scale.to(device).view(1, -1)
    # 切换到训练模式，确保可训练层按训练逻辑工作。
    model.train()

    # 遍历训练集 batch，并对每个 batch 做一次参数更新。
    for xb, yb in train_loader:
        # 将输入和标签移动到 CPU 或 GPU。
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        # 清空上一轮 batch 留下的梯度。
        optimizer.zero_grad(set_to_none=True)
        # 前向传播，得到三维 outcome 预测。
        pred = model(xb)
        # 先按训练集 target 尺度归一化误差。
        diff_scaled = (pred - yb) / scale
        # 计算当前 batch 的 mse_total_scaled。
        mse_total_scaled = torch.mean(diff_scaled * diff_scaled)
        # 反向传播，计算参数梯度。
        mse_total_scaled.backward()
        # 根据梯度更新模型参数。
        optimizer.step()


def train_model_with_early_stopping(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    max_epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    ckpt_path: str | Path,
    target_scale: torch.Tensor,
    patience: int = 10,
    min_delta: float = 1e-6,
    model_name: str = "model",
) -> dict[str, Any]:
    """训练可学习的 outcome model，并按 val_mse_total_scaled 做早停。

    参数说明：
    - max_epochs 是训练轮数上限，不表示一定跑满。
    - patience 表示最多允许连续多少个 epoch 没有验证集提升。
    - min_delta 表示 val_mse_total_scaled 至少下降多少才算一次有效提升。
    - ckpt_path 只保存验证集 mse_total_scaled 最优的模型参数。
    """

    # 确保模型和数据使用同一个计算设备。
    model.to(device)
    # 将 checkpoint 路径规范化为 Path，方便后续创建目录和保存文件。
    ckpt_path = Path(ckpt_path)
    # 创建 checkpoint 父目录，避免保存模型时目录不存在。
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    # 使用 AdamW 作为优化器，保持和旧训练脚本的优化风格一致。
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # 当前观察到的最优验证集 mse_total_scaled，初始设为无穷大。
    best_val_mse_total_scaled = float("inf")
    # 记录最优 checkpoint 对应的 epoch。
    best_epoch = 0
    # 记录连续多少个 epoch 没有有效改善。
    stale_epochs = 0
    # 如果没有提前停止，则 stopped_epoch 等于 max_epochs。
    stopped_epoch = max_epochs

    # history 会完整保存训练配置和每个 epoch 的指标，供报告和绘图使用。
    history: dict[str, Any] = {
        "model": model_name,
        "monitor": "val_mse_total_scaled",
        "max_epochs": max_epochs,
        "patience": patience,
        "min_delta": min_delta,
        "best_epoch": None,
        "best_val_mse_total_scaled": None,
        "stopped_epoch": None,
        "checkpoint": str(ckpt_path),
        "target_scale": target_scale.detach().cpu().tolist(),
        "epochs": [],
    }

    # epoch 从 1 开始，方便日志和图像横轴阅读。
    for epoch in range(1, max_epochs + 1):
        # 先在训练集上做一轮参数更新。
        _train_one_epoch(model, train_loader, optimizer, device, target_scale)

        # 每个数据集只遍历一次，同时得到物理指标和早停指标。
        train_metrics, train_mse_total_scaled = evaluate_metrics_with_scaled_mse(
            model, train_loader, device, target_scale
        )
        val_metrics, val_mse_total_scaled = evaluate_metrics_with_scaled_mse(
            model, val_loader, device, target_scale
        )

        # 只有超过 min_delta 的下降才算真正改善，避免浮点噪声频繁刷新 checkpoint。
        improved = val_mse_total_scaled < best_val_mse_total_scaled - min_delta
        if improved:
            # 更新最优验证集指标。
            best_val_mse_total_scaled = val_mse_total_scaled
            # 记录最优 epoch。
            best_epoch = epoch
            # 验证集变好后，连续未改善计数归零。
            stale_epochs = 0
            # 保存当前最优模型参数。
            torch.save(model.state_dict(), ckpt_path)
        else:
            # 本轮没有有效改善，连续未改善计数加一。
            stale_epochs += 1

        # 将本轮所有关键指标记录到 history，字段名保持 train/val 对称。
        epoch_record = {
            "epoch": epoch,
            "train_mse_total_scaled": train_mse_total_scaled,
            "val_mse_total_scaled": val_mse_total_scaled,
            "train_mse_total": train_metrics.mse_total,
            "val_mse_total": val_metrics.mse_total,
            "train_mse_planar": train_metrics.mse_planar,
            "val_mse_planar": val_metrics.mse_planar,
            "train_mae_dx": train_metrics.mae_dx,
            "val_mae_dx": val_metrics.mae_dx,
            "train_mae_dy": train_metrics.mae_dy,
            "val_mae_dy": val_metrics.mae_dy,
            "train_mae_planar": train_metrics.mae_planar,
            "val_mae_planar": val_metrics.mae_planar,
            "train_mae_yaw_deg": train_metrics.mae_yaw_deg,
            "val_mae_yaw_deg": val_metrics.mae_yaw_deg,
            "best_val_mse_total_scaled": best_val_mse_total_scaled,
            "stale_epochs": stale_epochs,
            "checkpoint_saved": improved,
        }
        # 追加本轮记录。
        history["epochs"].append(epoch_record)

        # 每个 epoch 打印一行，便于手动观察收敛、泛化和早停状态。
        print(
            f"  epoch {epoch:03d}/{max_epochs}  "
            f"train_mse_total_scaled={train_mse_total_scaled:.6f}  "
            f"val_mse_total_scaled={val_mse_total_scaled:.6f}  "
            f"best_val_mse_total_scaled={best_val_mse_total_scaled:.6f}  "
            f"train_yaw_deg={train_metrics.mae_yaw_deg:.3f}  "
            f"val_yaw_deg={val_metrics.mae_yaw_deg:.3f}  "
            f"stale={stale_epochs}/{patience}"
        )

        # 连续 patience 个 epoch 没有改善时，提前停止训练。
        if stale_epochs >= patience:
            stopped_epoch = epoch
            print(
                f"  early stop: val_mse_total_scaled 连续 {patience} 个 epoch 没有明显改善，"
                f"best_epoch={best_epoch}"
            )
            break

    # 训练结束后，将最终早停摘要写回 history 顶层。
    history["best_epoch"] = best_epoch
    history["best_val_mse_total_scaled"] = best_val_mse_total_scaled
    history["stopped_epoch"] = stopped_epoch
    # 返回完整训练记录，由外层脚本决定是否保存为 JSON。
    return history
