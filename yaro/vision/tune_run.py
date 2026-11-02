"""真调优执行器：真的把多组超参跑完，而不是"规划 trials"。

默认策略（为了在单机上划算）：
- 先用便宜网格（epochs 小、imgsz 小）筛掉明显差的组合；
- 每轮都受 ``max_minutes`` 预算约束，超预算直接停并保留已有结果；
- 结果写 ``runs/tune/trials.jsonl`` + ``best.json``，并用 ``plan.summarize_trials``
  按噪声底线判定"差异是否可信"（相邻 < 0.002 一律当噪声，不选"更好的"）。

CUDA 门槛与 ``train_run`` 一致：没有 GPU 时默认拒绝（显式 device="cpu" 才冒烟）。
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any, Callable

from . import plan
from .train_run import DEFAULT_MODEL, TRAIN_MODELS, TrainConfig, train_dataset, train_support

# 默认搜索空间：只放"便宜且真影响结果"的维度。维度越大越烧时间，故意保守。
DEFAULT_SPACE: dict[str, Any] = {
    "epochs": [8, 16],
    "lr": [0.002, 0.005],
    "imgsz": [416, 512],
    "batch": [2],
}

_MAX_TRIALS_HARD = 32


def _config_from_trial(base: TrainConfig, trial: dict[str, Any]) -> TrainConfig:
    allowed = {field.name for field in dataclasses.fields(TrainConfig)}
    payload = {key: value for key, value in trial.items() if key in allowed}
    return dataclasses.replace(base, **payload)


def tune_support(*, device: str = "auto") -> dict[str, Any]:
    return train_support(device=device)


def run_trials(
    dataset: Path | str,
    *,
    space: dict[str, Any] | None = None,
    trials: int = 6,
    mode: str = "grid",
    seed: int = 7,
    base: TrainConfig | None = None,
    metric: str = "map50",
    direction: str = "higher",
    root: Path | str | None = None,
    out_dir: Path | str | None = None,
    max_minutes_total: float | None = None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """依次跑多组配置。返回 {ok, rows, best, pareto, notes, path, stopped_reason}。"""
    base = base or TrainConfig()
    dataset_path = Path(dataset).resolve()
    support = tune_support(device=base.device)
    if not support["ok"]:
        return {
            "ok": False,
            "error": support["reason"],
            "hint": support.get("hint", ""),
            "rows": [],
            "written": [],
            "stopped_reason": "env_missing",
        }
    base.device = support.get("device", base.device)

    combos = plan.plan_trials(space or DEFAULT_SPACE, trials=trials, seed=seed, mode=mode)
    combos = combos[:_MAX_TRIALS_HARD]
    out = Path(out_dir).resolve() if out_dir else (dataset_path / "runs" / "tune")
    out.mkdir(parents=True, exist_ok=True)
    trials_path = out / "trials.jsonl"
    if trials_path.exists():
        trials_path.unlink()

    started = time.time()
    rows: list[dict[str, Any]] = []
    stopped_reason = "done"
    for index, combo in enumerate(combos, start=1):
        if max_minutes_total is not None:
            elapsed_min = (time.time() - started) / 60.0
            if elapsed_min >= float(max_minutes_total):
                stopped_reason = "budget"
                break
        config = _config_from_trial(base, combo)
        if max_minutes_total is not None:
            remaining = float(max_minutes_total) - (time.time() - started) / 60.0
            config.max_minutes = max(1.0, min(config.max_minutes, remaining))
        result = train_dataset(dataset_path, config=config, root=root, out_dir=out / f"trial-{index:02d}")
        row = {
            "trial": index,
            "config": config.to_dict(),
            "ok": result.ok,
            metric: result.best_map50,
            "epochs_done": result.epochs_done,
            "duration_s": result.duration_s,
            "stopped_reason": result.stopped_reason,
            "checkpoint": result.best_checkpoint,
            "error": result.error[:400] if result.error else "",
        }
        rows.append(row)
        with trials_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if on_progress is not None:
            try:
                on_progress(row)
            except Exception:  # 回调不应该影响训练
                pass

    summary = plan.summarize_trials(rows, metric=metric, direction=direction)
    best = summary.get("best") or {}
    written = [str(trials_path)]
    if best:
        best_path = out / "best.json"
        best_path.write_text(
            json.dumps(
                {
                    "metric": metric,
                    "direction": direction,
                    "best": best,
                    "notes": summary.get("notes", []),
                    "model": base.model,
                    "dataset": str(dataset_path),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        written.append(str(best_path))
    return {
        "ok": any(row["ok"] for row in rows),
        "dataset": str(dataset_path),
        "metric": metric,
        "direction": direction,
        "rows": rows,
        "best": best,
        "ordered": summary.get("ordered", []),
        "pareto": summary.get("pareto", []),
        "notes": summary.get("notes", []),
        "written": written,
        "stopped_reason": stopped_reason,
        "duration_s": round(time.time() - started, 2),
        "models": TRAIN_MODELS,
        "default_model": DEFAULT_MODEL,
    }


def describe_space(space: dict[str, Any] | None = None) -> str:
    space = space or DEFAULT_SPACE
    return " · ".join(f"{key}={value}" for key, value in space.items())
