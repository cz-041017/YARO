# YARO

**You Only Run Once** —— 面向机器视觉开发的开源 agent harness（智能体编排框架）。

> 给它一个图片集，它把 **体检 → 预标注 → 人工审难例 → 清洗 → 切分 → 训练 → 调优 → 导出测速**
> 串成一条可回滚、可复现、可审计的链路。模型只负责"下一步做什么"这个决策，动作一律经过工具层执行，
> 状态只写进 append-only 事件链，分数只由 harness 自己算，**默认一个字节都不出本机**。

<p align="center">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
  <a href="#"><img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-green.svg"></a>
  <a href="#"><img alt="tests 323 passed" src="https://img.shields.io/badge/tests-323%20passed-brightgreen.svg"></a>
</p>

---

## 目录

- [项目简介](#项目简介)
- [核心特性](#核心特性)
- [快速开始](#快速开始)
- [一次全链路长什么样](#一次全链路长什么样)
- [架构](#架构)
- [目录结构](#目录结构)
- [能力与边界](#能力与边界)
- [项目状态](#项目状态诚实版)
- [隐私与密钥](#隐私与密钥)
- [二次开发指南](#二次开发指南)
- [测试与自检](#测试与自检)
- [贡献](#贡献)
- [许可证与引用](#许可证与引用)

---

## 项目简介

YARO 由「**薄 agent + 厚评估器 + 事件溯源会话**」构成，回答一个问题：*在图像 / 客户数据 / 密钥都
不能随便上云的约束下，怎么让一个大模型安全、可复现、可审计地完成一条机器视觉开发链路？*

- **薄 agent**：模型只做决策（下一步调哪个工具、训练是否达标、要不要继续），不直接碰文件系统。
- **厚评估器**：体检、切分、训练指标、p50/p95 测速、故障注入评分，全部由 harness 自己的代码算，不信任模型口述。
- **事件溯源会话**：每一步都 append 一条带 `prev_digest` 的事件，回滚 = 移指针，篡改可校验，产出走内容寻址库（CAS）。
- **约束先行**：出网四档闸门（默认 `off`）、预算上限、工具审批（PLAN/DEFAULT/YOLO）、训练仅 CUDA。

v0.7 起，编排层从手写事件循环迁到 **LangGraph**（见 [架构](#架构)），同时把 DeepSeek 的
`reasoning_content` 逐轮回传、视觉脱敏、出网闸、成本拆分这些"框架不替你做"的部分自持在 provider 层。

---

## 核心特性

| 特性 | 说明 |
|---|---|
| 全链路真执行 | 不是"生成一段训练脚本给你看"，而是真跑：预标注落 `proposals/`、训练落 `best.pt`、导出落 `model.onnx`、测速落 `bench.json` |
| 蒸馏闭环 | teacher（大模型定"什么 + 类" + GroundingDINO/SAM2 定"框"）→ student 训小模型 → `select_hard` 主动学习选难例 → 回灌 |
| 企业级可信 | 事件溯源 digest 链、`AuditLedger` 审计账本、工具审批、`goal` 持久目标状态机、故障注入 + 行为评分（`agent_eval`） |
| 隐私默认关 | egress 四档（`off`/`text`/`derived`/`full`），策略 + 端点判定双闸；原始图与渲染图分档 |
| 密钥安全 | 四级来源（env → keyring → `secrets.json`(0600) → `.env`），日志/事件/错误栈只出现掩码 |
| 可编排可扩展 | LangGraph 图 + `ToolRegistry` 工具注册 + `.agents/skills/` 技能 + MCP 消费，扩展点清晰 |
| 零依赖前端 | 内置三栏 Web IDE（stdlib `http.server` + 原生 HTML/CSS/JS，无构建），另有 pywebview 桌面壳 |

---

## 快速开始

### 用打包好的 exe（推荐给同事）

直接下载 [YARO-win64.zip](https://github.com/cz-041017/YARO/raw/releases/YARO-win64.zip)，解压后双击 `YARO.exe`。目标机**不需要** Python、pip 或联网。

构建见 [packaging/README.md](packaging/README.md)（`packaging\build_windows.ps1`）。

### 从源码运行

```sh
git clone git@github.com:cz-041017/YARO.git
cd YARO

python -m venv .venv
.venv\Scripts\activate              # macOS/Linux: source .venv/bin/activate

python -m pip install -e ".[full,dev]"
python -m yaro init                 # 建工作区、铺演示数据、写 .env 模板
python -m yaro app                  # 独立窗口；没有窗口后端时用 yaro gui
```

> `python -m pip` 而不是裸 `pip`：否则依赖可能装到别的解释器里。
>
> 依赖分组（`pyproject.toml` 的 optional-dependencies）：`api`（OpenAI SDK）、`graph`（LangGraph 运行时）、
> `cv`（numpy/opencv）、`ml`（torch/onnx，**真训练才需要，体积大**）、`keyring`（系统钥匙串）、
> `desktop`（pywebview）、`dev`（pytest）、`full`（api+cv+desktop+graph 的推荐组合）。

### 不花 token 先看一眼

```sh
python -m yaro chain --dataset demo/dataset                    # 体检 + 抽查标注 + 切分 + 训练计划
python -m yaro pipeline --dataset demo/dataset --skip-train    # 预标注 + 体检 + 切分
```

GUI 里勾选「演示模式」同样零 token，会跑完 8 轮确定性剧本（走 `DemoChatModel`，验证整条新引擎接线）。

---

## 一次全链路长什么样

| 阶段 | 工具 | 产出 | 需要什么 |
|---|---|---|---|
| 预标注 | `prelabel_run` | `proposals/*.txt` + `report.json` | CPU 即可（轮廓法）；torch / ONNX 模型可选 |
| 体检 | `dataset_audit` | 问题清单 + 错误/警告计数 | 无 |
| 切分 | `dataset_split` | `splits/*.txt` + `yaro-data.yaml` | 无 |
| 训练 | `train_run` | `best.pt` + `metrics.jsonl` + `train.log` | **CUDA** + torch / torchvision |
| 调优 | `tune_run` | `trials.jsonl` + `best.json` | 同上 |
| 导出测速 | `deploy_run` | `model.onnx` + `bench.json`（p50 / p95） | onnx + onnxruntime |

一条命令串起来：

```sh
python -m yaro pipeline --dataset <数据集目录> --epochs 12 --trials 4 --latency-target 33
```

训练执行器默认**只在 CUDA 机器上启用**；显式 `--cpu` 才允许冒烟（几轮即停）。

---

## 架构

```
接口层     yaro app（系统 WebView 桌面壳） · yaro gui（127.0.0.1:8765） · CLI · Python API
   │
编排层    LangGraph 状态图（yaro/graph/）
   │        build_agent：agent → tools → finalize → compact（工具延迟加载 + 上下文压缩 + stop_check）
   │        build_distill_graph：label → review(HITL 中断) → train → analyze → select_hard 蒸馏闭环
   │
provider   ChatDeepSeek（reasoning_content 逐轮回传） · GuardedModel（出网闸 + 脱敏） · usage 拆分记账
   │
能力层    底座工具 + 14 视觉工具 + MCP 工具（每个返回：摘要 + 证据图 + 表格 + 指标）
   │
记忆/审计  事件溯源会话树（SessionStore，GUI 事件日志） · 产物 CAS（artifact:// URI） · AuditLedger（digest 链）
   │
执行层    runner（子进程 + 超时杀进程树） · Docker 沙箱（--network none / 只读根）
```

模型是可替换的，harness 是不变的那一半。

### 设计取舍

| 决策 | 理由 | 代价 |
|---|---|---|
| LangGraph 编排 + 自持 `reasoning_content` 序列化 | 框架的消息抽象会吃掉 DeepSeek 的非标准字段（带 tools 必须逐轮回传，否则 400），所以在 provider 层覆盖解析/注入 | 需自己维护 `ChatDeepSeek` 子类 |
| 事件溯源会话树做 GUI 事件日志 | 回滚/fork/审计只需 append + 移指针，`prev_digest` 链可校验篡改 | 需自己实现压缩切点校验 |
| 工具返回证据，不返回裸日志 | 上下文稀缺，证据图 + 紧凑数值既省 token 又让模型"看得见" | 每个工具都要写渲染 |
| 执行一律走子进程 / 沙箱 | 训练/导出可能拖死会话或占满显存，超时要能杀整棵进程树 | 启动开销、跨进程传产物 |
| 默认不出网 | 图像/客户数据/密钥默认不出本机 | 第一次用必须走演示模式或显式授权 |
| 主动学习 + 故障注入评估先行 | 蒸馏用 `select_hard` 选难例回灌；`agent_eval` 用故障注入 + 行为评分测 harness 是否诚实 | 额外评估成本 |

更长的决策记录（ADR）在 [docs/](docs/) 保留。

---

## 目录结构

```text
yaro/
├── graph/                  # LangGraph 编排（v0.7 核心运行时）
│   ├── agent.py            #   build_agent：agent/tools/finalize/compact 四节点
│   ├── distill.py          #   build_distill_graph：蒸馏闭环（含 select_hard 主动学习）
│   ├── runtime.py          #   run_distillation：串 checkpointer + 审批 + 审计
│   ├── runner.py           #   run_agent：把图输出映射回 SessionStore 事件（GUI 引擎桥）
│   ├── context.py          #   estimate_tokens / summarize_prefix / select_schemas 上下文经济
│   └── state.py            #   GraphState
├── provider/               # 模型与协议适配（自持的部分都在这里）
│   ├── deepseek.py         #   ChatDeepSeek(ChatOpenAI)：reasoning_content 逐轮回传
│   ├── guard.py            #   GuardedModel：出网闸 + 内容脱敏
│   ├── models.py           #   build_model
│   ├── usage.py            #   usage_from_messages：hit/miss/out 拆分（缓存价差 50×）
│   └── demo.py             #   DemoChatModel：离线剧本
├── capabilities/           # 底座工具 + 视觉工具 + ToolRegistry + Docker 沙箱
├── vision/                 # 视觉能力：dataset/prelabel/active/train_run/tune_run/deploy_run/...
├── exec/                   # 受控子进程 runner（超时杀进程树）
├── memory/                 # SessionStore（事件溯源）+ ArtifactStore（CAS）
├── loop/                   # 旧手写循环（已退为回退路径，server.py 不再 import）
├── llm/                    # 旧 DeepSeek/Demo 接口（回退路径）
├── web/                    # 零依赖三栏 Web IDE
├── agent_eval.py           # 故障注入 + 行为评分
├── approval.py             # 工具审批（PLAN/DEFAULT/YOLO）
├── audit.py                # AuditLedger：digest 链审计账本
├── goal.py                 # 持久目标状态机（CAS + 锁）
├── skills.py               # SKILL.md 渐进披露
├── mcp.py                  # MCP 对内消费（包 langchain_mcp_adapters）
├── egress.py               # 出网四档闸门
├── secrets.py              # 密钥四级来源
├── server.py               # HTTP 服务 + GUI 后端（12 路由，仅 127.0.0.1）
├── config.py / contracts.py / gates.py / evals.py / cli.py
.agents/skills/             # 技能（SKILL.md）
examples/                   # 最小示例
docs/                       # ADR + 安装文档
tests/                      # 323 个用例
```

---

## 能力与边界

- **现在就能做**（无需 GPU / 联网 / token）：数据集体检、切分、去重、标注渲染、COCO/VOC 互转、轮廓法预标注、标定评估、演示模式、回滚与分叉。
- **需要条件**：真训练要 CUDA + torch；导出测速要 onnx + onnxruntime；JPEG 要 cv2 或 Pillow；云端模型要显式授权。
- **明确不做**：交互式标注 UI、分割/关键点训练、替代训练框架生态、多用户 RBAC、人脸与合规链路、通用代码 agent。

## 项目状态（诚实版）

| 项目 | 状态 |
|---|---|
| 数据链路（体检/切分/渲染/转换） | 实机跑通，pytest 覆盖 |
| LangGraph 编排 + server.py 换新引擎 | 已接通，`323 passed` + `verify.py 19/19` |
| 预标注 / 训练 / 调优 / 部署 | **代码完成，未做 GPU 实机验证**（需要 CUDA 机器） |
| 打包 exe | 已构建 `dist/YARO-win64.zip`（v0.7.0，doctor 冒烟通过），**未做换机验收** |
| 阶段 | v0.7.0（进行中）——**开发者预览**，CLI 与接口可能变 |

**任何"跑通"结论都以你机器上的输出为准**；首次使用请按 [docs/INSTALL.md](docs/INSTALL.md) 的验收清单走一遍。

---

## 隐私与密钥

- **egress 四档**：`off`（默认，一个字节不出本机）/ `text` / `derived` / `full`。策略之外还有第二道闸：目标端点判定（公网必须显式授权）。
- **原始图与 harness 渲染图分档**：`image_probe` 的图标记为 `raw`，`off` / `text` / `derived` 都不会外发；被剥离时模型会收到 `[隐私策略]` 提示。
- **密钥四级来源**：环境变量 → 系统 keyring → `~/.yaro/secrets.json`（0600）→ `.env`。界面、日志、事件、错误栈只出现掩码（前 4 后 4）。

### 填写 API Key（入口在哪）

| 入口 | 命令 / 位置 | 说明 |
|---|---|---|
| ① 程序窗口 | 左栏底部点 `api key` → 弹窗输入两次 | 只发到本机 `127.0.0.1`；存钥匙串或 `secrets.json`(0600)；`YARO_GUI_PUBLIC=1` 时该接口直接 403 |
| ② 命令行 | `yaro key set` | 交互式输入，**不回显、不进 shell 历史** |
| ③ 脚本 / CI | `yaro key set --stdin` 或环境变量 `DEEPSEEK_API_KEY` | 管道输入 / CI 注入 |
| ④ 本地 `.env` | `DEEPSEEK_API_KEY=sk-...` | `yaro init` 会生成 `.env` 模板，已在 `.gitignore` |

无论从哪个入口进来，Key 都**不会**出现在事件账本、日志、错误栈或页面 HTML 里。

```sh
yaro key status   # configured / 来源 / 掩码 sk-1234…abcd
yaro doctor       # 密钥来源 + egress 生效情况 + 模型端点是否公网
```

- **模型名写错会直接报错**：价格表同时也是"认识的型号清单"，未知型号启动即失败，不按 0 或猜价记账。
- **用云端 DeepSeek 需显式授权出网**：`YARO_EGRESS=text` + `YARO_EGRESS_ALLOW_PUBLIC=1`，或 `yaro init --egress text --allow-public`。
- **换成内网 / 本机模型**：`DEEPSEEK_BASE_URL=http://127.0.0.1:8000/v1`（内网端点不触发公网授权闸）。

---

## 二次开发指南

YARO 的扩展点按"改动成本从小到大"排列，绝大多数二次开发只需要动一处。

### 1. 加一个工具（最小改动）

工具是一个「`ToolSpec` + `async run(**kwargs) -> ToolResult`」，注册进 `ToolRegistry` 即可被 agent 图调用：

```python
from yaro.capabilities.registry import ToolRegistry
from yaro.contracts import Evidence, ToolResult, ToolSpec


class InspectTool:
    spec = ToolSpec(
        name="inspect",
        description="检查单张图片的尺寸与格式",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "图片相对路径"}},
            "required": ["path"],
        },
    )

    async def run(self, path: str = "") -> ToolResult:
        return ToolResult(
            ok=True,
            summary="640x480 JPEG",
            evidence=[Evidence(kind="scalar", payload=640, caption="宽度")],
            metrics={"width": 640.0, "height": 480.0},
        )


registry = ToolRegistry([InspectTool()])
```

要点：**工具绝不抛裸异常**（由 `ToolRegistry.call` 兜底成 `ok=False`）；证据只给紧凑数值/表格/图，不给裸日志。

### 2. 加一个技能（SKILL.md）

在 `.agents/skills/<name>/SKILL.md` 写一个技能，agent 分析阶段会自动注入技能目录（渐进披露：先只给
name+description，正文按需再读）。参考现有 `.agents/skills/dataset-audit/SKILL.md`。

### 3. 加一个视觉能力

在 `yaro/vision/` 下加模块（如 `my_calib.py`），返回同构的 `dict`（`ok`/`summary`/`metrics`/`errors`），
再由 `yaro/capabilities/` 包一层 `ToolSpec` 暴露给 agent。预标注三后端（`prelabel.py`）、主动学习
（`active.py`）都是这个模式。

### 4. 改编排（graph 层）

- 单任务 agent 图：`yaro/graph/agent.py` 的 `build_agent(...)`（节点 agent/tools/finalize/compact）。
- 蒸馏闭环：`yaro/graph/distill.py` 的 `build_distill_graph(...)`（label→review→train→analyze→select_hard）。
- 会话桥：`yaro/graph/runner.py` 的 `run_agent(...)`（把图输出映射回 SessionStore 事件，GUI 不感知引擎）。

### 5. 接一个 MCP 工具源

`yaro/mcp.py` 的 `mcp_tools([McpServer(...)])` 把社区 MCP 视觉工具转成 `ToolSpec`，混进同一个 registry：

```python
from yaro.mcp import McpServer, mcp_tools

async with mcp_tools([McpServer(name="my-tool", transport="stdio", command="python", args=["server.py"])]) as tools:
    registry = ToolRegistry([*your_tools, *tools])
```

### 6. 改 provider（新模型 / 新协议）

`yaro/provider/` 是唯一放协议适配的地方：新供应商继承 `ChatOpenAI`（或直接实现
`bind_tools` + `ainvoke` 的最小接口），再把 `reasoning_content` 逐轮回传、出网闸、成本拆分这些约束照抄进去。

### 提交要求（写代码前请先看）

每个模块都要求：

1. **离线测试**：`python -m pytest -q`（当前 323 passed），不要引入需要网络/API Key/CUDA 的用例。
2. **端到端自检**：`python verify.py`（19 步，不花 token）。
3. **诚实降级**：异常路径要 fail-closed（返回 `ok=False` 而不是抛穿），不确定的结论要显式标注，不要猜。

---

## 测试与自检

```sh
python -m pytest -q      # 离线测试（323 用例，不花 token / 不需要网络）
python verify.py         # 19 步端到端自检（不花 token）
python -m yaro doctor    # 环境与能力探测（CUDA / 依赖 / egress / 密钥来源）
```

打包成 exe 或离线 wheelhouse 见 [packaging/README.md](packaging/README.md)。

---

## 贡献

欢迎任何形式的贡献：Bug 报告、功能建议、文档纠错、PR。提交前请：

1. 先开 issue 描述问题或想法（避免重复劳动）；
2. 跑通 `python -m pytest -q` 与 `python verify.py`；
3. 新功能配上测试，说明"为什么这样做"（ADR 风格）；
4. 遵循仓库的 fail-closed / 隐私默认关 / 工具不抛裸异常三条硬约束。

**提交修改 / 报告问题**：欢迎直接发邮件至 **q83760113@163.com**，或在本仓库提 issue / PR。

> 本项目只做**通用视觉工具开发**，明确不涉足质检、医疗、安防等高风险合规领域；请勿提交此类用例或数据。

---

## 许可证与引用

[MIT](LICENSE)。第三方依赖及其许可见各依赖仓库；主链路不使用 AGPL/GPL 组件。

设计参考 [_Harness Engineering: Anatomy, Architecture, and Evolution of Coding Agents_](https://arxiv.org/abs/2609.00006)（2026）。

```bibtex
@misc{yaro2026,
  title  = {YARO: A Harness for Machine-Vision Development},
  author = {YARO contributors},
  year   = {2026},
  url    = {https://github.com/cz-041017/YARO}
}
```
