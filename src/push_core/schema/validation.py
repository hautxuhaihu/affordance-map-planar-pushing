"""Level 1 CSV validator for direct-region-force dataset.

对采集输出的 CSV 执行文件级 + 行级校验，不修改输入文件。
"""

import csv
from pathlib import Path
from typing import Any

from push_core.schema import level1_schema as schema

# ──────────────────────────────────────────────
# 文件读取
# ──────────────────────────────────────────────


def read_csv_rows(path: str | Path) -> list[dict[str, Any]]:
    """读取 CSV 文件，返回 row dict 列表。

    文件级校验：
    - 文件必须存在
    - 前 3 字节必须是 UTF-8 BOM (\\xef\\xbb\\xbf)
    - 表头字段顺序必须等于 schema.FIELD_NAMES

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: BOM 缺失、header 不匹配。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # BOM 检查
    with open(path, "rb") as f:
        raw = f.read(3)
    if raw != b"\xef\xbb\xbf":
        raise ValueError(
            f"Missing UTF-8 BOM at start of {path}. "
            f"Expected \\xef\\xbb\\xbf, got {raw.hex()!r}"
        )

    # 读取 header
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)

    header_errors = schema.validate_header(header)
    if header_errors:
        raise ValueError(f"Header mismatch in {path}: {'; '.join(header_errors)}")

    # 读取数据行
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, fieldnames=schema.FIELD_NAMES)
        next(reader)  # 跳过已校验的 header
        for row in reader:
            rows.append(row)

    return rows


# ──────────────────────────────────────────────
# 行级校验
# ──────────────────────────────────────────────


def validate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """对每行调用 schema.validate_row()，收集错误。

    Returns:
        错误列表，每项包含 row_index、candidate_id、errors。
        空列表表示全部通过。
    """
    errors: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        row_errors = schema.validate_row(row)
        if row_errors:
            errors.append(
                {
                    "row_index": i + 2,  # 第 1 行是 header
                    "candidate_id": row.get("candidate_id", "?"),
                    "errors": row_errors,
                }
            )
    return errors


# ──────────────────────────────────────────────
# 报告打印
# ──────────────────────────────────────────────


def print_validation_report(
    total_rows: int,
    errors: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> None:
    """打印校验报告到 stdout。"""
    print(f"Rows: {total_rows}")

    if errors:
        print(f"Validation: {len(errors)} rows with errors")
        for item in errors[:5]:
            print(f"  row {item['row_index']} (candidate {item['candidate_id']}):")
            for e in item["errors"][:3]:
                print(f"    - {e}")
            if len(item["errors"]) > 3:
                print(f"    ... and {len(item['errors']) - 3} more")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more error rows")
    else:
        print("Validation: 0 errors")

    if not rows:
        print("delta_z: n/a (empty)")
        return

    # 质量字段统计
    def count_flag(field: str, value: int) -> int:
        return sum(1 for r in rows if int(float(str(r.get(field, 0)))) == value)

    total = len(rows)
    qpass = count_flag("quality_pass", 1)
    stable = count_flag("simulation_unstable", 0)
    contact = count_flag("contact_success", 1)
    stopped = count_flag("stopped_by_threshold", 1)

    print("\nQuality:")
    print(f"  quality_pass=1:          {qpass}/{total} ({100.0 * qpass / total:.1f}%)")
    print(
        f"  simulation_unstable=0:   {stable}/{total} ({100.0 * stable / total:.1f}%)"
    )
    print(
        f"  contact_success=1:       {contact}/{total} ({100.0 * contact / total:.1f}%)"
    )
    print(
        f"  stopped_by_threshold=1:  {stopped}/{total} ({100.0 * stopped / total:.1f}%)"
    )

    # delta_z 统计
    dz_values = [float(r["delta_z"]) for r in rows]
    dz_abs = [abs(v) for v in dz_values]
    print("\ndelta_z:")
    print(
        f"  min: {min(dz_values):.6f}  max: {max(dz_values):.6f}  mean(|dz|): {sum(dz_abs) / len(dz_abs):.6f}"
    )


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────


def validate_csv(path: str | Path) -> int:
    """完整的 CSV 校验流程。

    Returns:
        0 = 全部通过
        1 = 存在错误
    """
    try:
        rows = read_csv_rows(path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Validation failed: {e}")
        return 1

    print(f"File: {Path(path).resolve()}")
    errors = validate_rows(rows)
    print_validation_report(len(rows), errors, rows)
    return 1 if errors else 0
