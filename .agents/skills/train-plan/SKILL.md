---
name: train-plan
description: 生成训练计划并在训练后诊断日志（平台期/过拟合/无验证指标）
when_to_use:
  - 用户要训练一个小而快的检测模型时
  - 训练日志需要诊断下一步该怎么调时
allowed-tools: [train_plan, train_run, train_analyze]
license: MIT
version: 1
---
# 训练计划与诊断

目标：用伪标签蒸馏出可落地的 student 小模型，训练默认只在 CUDA 机器启用。

## 步骤
1. `train_plan` 生成 recipe.json + train.py + acceptance.md（先读 acceptance 的验收清单）。
2. 确认 split 已就位（缺则 `dataset_split --apply`）。
3. `train_run` 真训练（默认 fasterrcnn-mnv3，CUDA；device=cpu 仅冒烟）。
4. `train_analyze` 读 metrics.jsonl 诊断：loss 平台期 → 降 lr / 加数据；val_loss 上升 → 过拟合，早停或加增强；无验证指标 → 先补 val 集。

## 验收
- best_map50 记录在案，且两次同种子复跑差 < 噪声底线
- 有模型卡（model_card），记录数据/类别/阈值/延迟
