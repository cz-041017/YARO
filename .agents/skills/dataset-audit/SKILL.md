---
name: dataset-audit
description: 体检数据集，找脏数据（截断图/缺标注/越界框/重复图/类别失衡）
when_to_use:
  - 收到一批新图片或数据集需要体检时
  - 训练前想确认数据是否干净时
allowed-tools: [dataset_scan, dataset_audit, dataset_render]
license: MIT
version: 1
---
# 数据集体检

目标：在训练前发现并隔离脏数据，避免"垃圾进垃圾出"。

## 步骤
1. `dataset_scan` 看概览（图片数/有标注数/框数/类别分布）。
2. `dataset_audit` 列出问题，按 severity 排序；error 级别（corrupt_image / box_out_of_range / orphan_label）必须先处理。
3. `dataset_render` 抽查标注框是否贴合目标，尤其小目标与密集场景。
4. 重复图（duplicate_image）要 `dataset_dedupe --apply` 隔离，否则会虚高指标。

## 验收
- audit 的 errors == 0
- 类别失衡（class_imbalance）在合理范围内，或已在报告里说明
- 抽查渲染图无系统性错框
