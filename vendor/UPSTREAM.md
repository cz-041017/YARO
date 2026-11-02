# 上游 OpenVLHarness 的 pin 与 patch 规范

## 当前 pin

```
repo:   https://github.com/uclanlp/OpenVLHarness
license: MIT
pinned: TODO —— 首次 vendor 时填写 40 位 commit sha
date:   2026-11-02 记录
```

> 首次 vendor 执行（在 YARO 根目录）：
> ```bash
> git remote add upstream https://github.com/uclanlp/OpenVLHarness.git
> git fetch upstream
> git checkout -b vendor/openvlharness upstream/main
> # 把 openvlharness/ 与 evaluation_kit/ 拷进 yaro/vendor/openvlharness/
> # 写 patches/0000-initial-forks.patch 后合回 main
> ```

## 规则

1. **每改一处上游文件，必须落一个 patch**：`patches/NNNN-<一句话>.patch`。
   patch 头部注释写清：改了什么、为什么、上游若合并需不需要跟进。
2. **新能力一律放 `yaro/capabilities/`，不放进 vendor 目录。** vendor 目录只做"最小必要修改"。
3. **patch 数量超过 30 个即触发重新评估**（见 ADR-0001 后果段）：要么回灌上游，要么改为插件化。
4. **每季度对齐一次上游 main**，只做评估不自动合并。
5. vendor 目录保留上游 `LICENSE`、`SECURITY.md`、`THIRD_PARTY_LICENSES.md` 原文，不得删除。

## 当前已知需要打 patch 的点

| patch | 位置 | 原因 |
|---|---|---|
| 0001 | `openvlharness/agent/` | 接入 YARO 事件溯源会话与预算控制 |
| 0002 | coding tool | 宿主机执行 → Docker 沙箱（上游 [!WARNING] 明确无沙箱） |
| 0003 | 模型端点 | 内部端点 → DeepSeek 双 adapter |
| 0004 | 搜索后端 | Serper → DeepSeek Search & Fetch / 内部检索 |
