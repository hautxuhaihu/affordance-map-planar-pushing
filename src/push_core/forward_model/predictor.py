"""Level 1 MLP outcome model 推理接口。"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from push_core.forward_model.dataset import LEVEL1_OUTCOME_INPUT_FIELDS
from push_core.forward_model.models import OutcomeMLP
from push_core.forward_model.normaliser import StandardNormaliser
from push_core.project_paths import FORWARD_MODEL_RESULTS_DIR, PROJECT_ROOT


def default_project_root() -> Path:
    """返回当前项目根目录。"""
    return PROJECT_ROOT


def default_checkpoint_path() -> Path:
    """返回当前正式 Level 1 MLP v2 checkpoint 路径。"""
    return FORWARD_MODEL_RESULTS_DIR / "checkpoints" / "level1_mlp_model_v2.pt"


def default_normaliser_path() -> Path:
    """返回当前正式 Level 1 normaliser v2 路径。"""
    return FORWARD_MODEL_RESULTS_DIR / "level1_outcome_normaliser_v2.json"


class Level1OutcomePredictor:
    """加载已训练 MLP，对 Level 1 candidate row 预测 outcome。"""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        normaliser_path: str | Path | None = None,
        device: str | torch.device = "cpu",
    ):
        self.checkpoint_path = Path(checkpoint_path or default_checkpoint_path())
        self.normaliser_path = Path(normaliser_path or default_normaliser_path())
        self.device = torch.device(device)

        with open(self.normaliser_path, "r", encoding="utf-8") as f:
            normaliser_data = json.load(f)

        input_fields = tuple(normaliser_data.get("input_fields", ()))
        if input_fields != LEVEL1_OUTCOME_INPUT_FIELDS:
            raise ValueError(
                "normaliser input_fields 与 LEVEL1_OUTCOME_INPUT_FIELDS 不一致"
            )

        self.normaliser = StandardNormaliser.from_json_dict(normaliser_data)
        self.model = OutcomeMLP(input_dim=len(LEVEL1_OUTCOME_INPUT_FIELDS))
        state_dict = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def row_to_input(self, row: dict[str, Any]) -> np.ndarray:
        """将单行 CSV dict 转换为模型输入数组。"""
        values = [float(row[field]) for field in LEVEL1_OUTCOME_INPUT_FIELDS]
        return np.asarray(values, dtype=np.float32)

    def rows_to_input(self, rows: list[dict[str, Any]]) -> np.ndarray:
        """将多行 CSV dict 转换为模型输入矩阵。"""
        return np.stack([self.row_to_input(row) for row in rows], axis=0)

    def predict_array(self, x_raw: np.ndarray) -> list[dict[str, float]]:
        """对原始 10 维输入数组做批量预测。"""
        if x_raw.ndim == 1:
            x_raw = x_raw.reshape(1, -1)
        x = self.normaliser.transform(x_raw.astype(np.float32))
        xb = torch.from_numpy(x).to(self.device)

        with torch.no_grad():
            pred = self.model(xb).detach().cpu().numpy()

        return [self._prediction_to_dict(row) for row in pred]

    def predict_row(self, row: dict[str, Any]) -> dict[str, float]:
        """预测单行 candidate 的 outcome。"""
        return self.predict_array(self.row_to_input(row))[0]

    def predict_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, float]]:
        """批量预测多行 candidate 的 outcome。"""
        if not rows:
            return []
        return self.predict_array(self.rows_to_input(rows))

    @staticmethod
    def _prediction_to_dict(pred: np.ndarray) -> dict[str, float]:
        """将模型三维输出转换为预测结果。"""
        pred_delta_x = float(pred[0])
        pred_delta_y = float(pred[1])
        pred_delta_yaw = float(pred[2])

        return {
            "pred_delta_x": pred_delta_x,
            "pred_delta_y": pred_delta_y,
            "pred_delta_yaw": pred_delta_yaw,
        }
