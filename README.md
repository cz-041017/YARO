# YARO

**You Only Run Once** —— 面向机器视觉的开源 agent harness（智能体框架）。

给它一个图片集，它把 **体检 → 标注 → 清洗 → 切分 → 训练 → 调优 → 导出测速** 串成一条可回滚、可复现、可审计的链路。

YARO 由「薄 agent + 厚评估器 + 事件溯源会话」构成：模型只决定下一步做什么，动作一律经过工具层，状态只写进 append-only 事件链，分数只由 harness 自己算。**默认不出网**。

文档：[docs/INSTALL.md](docs/INSTALL.md) · 分发与打包：[packaging/README.md](packaging/README.md)

## 开发者预览

YARO 处于**开发者预览**阶段（v0.6.0），正在快速迭代，**CLI 与接口会变**。

本轮能力尚未完成生产级验证，请先读 [状态](#状态诚实版) 再决定怎么用。

## 运行

### 用打包好的 exe（推荐给同事）

解压 `YARO-win64.zip`，双击 `YARO.exe`。目标机**不需要** Python、pip 或联网。

构建见 [packaging/README.md](packaging/README.md)（`packaging\build_windows.ps1`）。

### 从源码运行

```sh
git clone git@github.com:cz-041017/YARO.git
cd YARO

python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate

python -m pip install -e ".[api,dev]"
python -m yaro init               # 建工作区、铺演示数据、写 .env 模板
python -m yaro app                # 独立窗口；没有窗口后端时用 yaro gui
```

`python -m pip` 而不是裸 `pip`：否则依赖会装到别的解释器里。

### 不花 token 先看一眼

```sh
python -m yaro chain --dataset demo/dataset                    # 体检 + 抽查标注 + 切分 + 训练计划
python -m yaro pipeline --dataset demo/dataset --skip-train    # 预标注 + 体检 + 切分
```

GUI 里勾选「演示模式」同样零 token，会跑完 8 轮完整剧本。

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

## 能力

| 层 | 内容 |
|---|---|
| 底座工具（4） | `list_files` · `image_probe` · `calib_assess` · `run_python`（Docker 沙箱） |
| 数据集 | `dataset_scan` · `dataset_audit` · `dataset_split` · `dataset_dedupe` · `dataset_render` · `label_convert` |
| 标注 | `prelabel_run`（torchvision / ONNX / 轮廓法三后端）→ `proposals/` → 人工只审难例 → `apply` 合并 |
| 训练与调优 | `train_plan` · `train_run` · `train_analyze` · `tune_plan` · `tune_run` |
| 部署 | `deploy_plan` · `deploy_run`（真导出 ONNX + 真测 p50/p95） |
| 标定 | 棋盘格内参与畸变评估（RMS、覆盖率、逐视角误差） |
| 会话 | 事件溯源 + digest 链、`rewind` / `fork`、内容寻址产物库、五道闸门、18 步自检 |

## 架构

```
接口    yaro app（系统 WebView） · yaro gui（127.0.0.1:8765） · CLI · Python API
  │
策略    CONTRACT.md（稳定前缀） · egress 四档 · 预算上限 · 人工审批
  │
循环    Agent.step：预算 → 压缩 → 请求 → 工具派发 → 卡死检测 → 停机
  │
能力    4 底座 + 14 视觉工具（每个返回：摘要 + 证据图 + 表格 + 指标）
记忆    事件溯源会话树（append-only + digest 链） · 产物 CAS · artifact:// URI
执行    runner（子进程 + 超时杀进程树） · Docker 沙箱（--network none / 只读根）
资源    模型（DeepSeek API 或本机 vLLM） · GPU · 数据集 · 工作区 · 密钥
```

模型是可替换的，harness 是不变的那一半。

## 能力边界

- **现在就能做**（无需 GPU / 联网 / token）：数据集体检、切分、去重、标注渲染、COCO/VOC 互转、轮廓法预标注、标定评估、演示模式、回滚与分叉。
- **需要条件**：真训练要 CUDA + torch；导出测速要 onnx + onnxruntime；JPEG 要 cv2 或 Pillow；云端模型要显式授权。
- **明确不做**：交互式标注 UI、分割/关键点训练、替代训练框架生态、多用户 RBAC、人脸与合规链路、通用代码 agent。

## 隐私与密钥

- **egress 四档**：`off`（默认，一个字节不出本机）/ `text` / `derived` / `full`。策略之外还有第二道闸：目标端点判定（公网必须显式授权）。
- **原始图与 harness 渲染图分档**：`image_probe` 的图标记为 `raw`，`off` / `text` / `derived` 都不会外发；被剥离时模型会收到 `[隐私策略]` 提示。
- **密钥四级来源**：环境变量 → 系统 keyring → `~/.yaro/secrets.json`（0600）→ `.env`。界面、日志、事件、错误栈只出现掩码（前 4 后 4）。

### 填写 API Key（入口在哪）

Key 只从下面四个入口进，按推荐顺序：

| 入口 | 命令 / 位置 | 说明 |
|---|---|---|
| ① 程序窗口 | 左栏底部点 `api key` 一行 → 弹窗输入两次 | 只发到本机 `127.0.0.1`；存进钥匙串或 `~/.yaro/secrets.json`（0600）；关窗即清空输入框；`YARO_GUI_PUBLIC=1` 时该接口直接 403 |
| ② 命令行（终端党） | `yaro key set` | 交互式输入，**不回显、不进 shell 历史**；能装 keyring 就交给系统钥匙串，否则写 `~/.yaro/secrets.json`（权限 0600） |
| ③ 脚本 / CI | `yaro key set --stdin` 或环境变量 `DEEPSEEK_API_KEY` | 管道输入：`echo $KEY \| yaro key set --stdin`；CI 里直接注入环境变量 |
| ④ 本地 `.env`（项目根或当前目录） | `DEEPSEEK_API_KEY=sk-...` | `yaro init` 会生成 `.env` 模板，**`.env` 已在 `.gitignore` 里** |

无论从哪个入口进来，Key 都**不会**出现在事件账本、日志、错误栈或页面 HTML 里；界面只显示掩码（前 4 后 4）。

查看与排错（都不会打印明文）：

```sh
yaro key status   # 显示 configured / 来源(env|keyring|file|dotenv) / 掩码 sk-1234…abcd
yaro doctor       # 密钥来源 + egress 生效情况 + 模型端点是否公网
```

- **窗口里怎么确认读到了**：左栏底部 `api key` 一行显示 `configured`（绿）/`missing`（红）。
- **模型名写错会直接报错**：价格表同时也是"认识的型号清单"，`YARO_MODEL` / `YARO_MODEL_PLAN` 不在表里时启动即失败，不会按 0 或猜一个价记账。
- **用云端 DeepSeek 还需要显式授权出网**（默认 `off` 会拒绝请求）：`YARO_EGRESS=text` + `YARO_EGRESS_ALLOW_PUBLIC=1`，或 `yaro init --egress text --allow-public`。
- **换成内网 / 本机模型**：`DEEPSEEK_BASE_URL=http://127.0.0.1:8000/v1`（内网端点不触发公网授权闸）。

## 状态（诚实版）

| 项目 | 状态 |
|---|---|
| 数据链路（体检/切分/渲染/转换） | 实机跑通，pytest 覆盖 |
| 预标注 / 训练 / 调优 / 部署 | **代码完成，未运行验证**（需要 CUDA 机器） |
| 打包 exe | 成功构建过一次，未做换机验收 |
| GUI / 独立窗口 | 浏览器模式与 `yaro app` 均实机开过窗口 |
| 测试 | 曾 111 passed / 7 failed，修复后未重跑 |

**任何"跑通"结论都以你机器上的输出为准**；首次使用请按 [docs/INSTALL.md](docs/INSTALL.md) 的验收清单走一遍。

## 设计取舍

| 决策 | 理由 | 代价 |
|---|---|---|
| 手写薄循环，不用 LangGraph / AutoGen | 带 tools 时 DeepSeek 要求逐轮回传 `reasoning_content`，框架的消息抽象会吃掉非标准字段（会 400） | 没有可视化编排 |
| 事件溯源会话树 | 回滚/fork/审计只需 append + 移指针，`prev_digest` 链可校验篡改 | 需自己实现压缩切点校验 |
| 工具返回证据，不返回裸日志 | 上下文是稀缺资源，证据图 + 紧凑数值既省 token 又让模型"看得见" | 每个工具都要写渲染 |
| 执行一律走子进程 / 沙箱 | 训练与导出的代码可能拖死会话或占满显存，超时要能杀整棵进程树 | 启动开销、跨进程传产物 |
| 默认不出网 | 图像/客户数据/密钥默认不出本机 | 第一次用必须走演示模式或显式授权 |
| 核心零依赖 | 装完即跑、打包体积可控、许可面干净（主链路避开 AGPL/GPL） | 前端手写；PNG 仅支持零依赖子集 |

更长的决策记录（ADR）在本地 `docs/` 保留，不随仓库发布。

## 开发

```sh
python -m pytest -q      # 离线测试
python verify.py         # 18 步端到端自检（不花 token）
python -m yaro doctor    # 环境与能力探测（含 CUDA / 依赖 / egress）
```

打包成 exe 或离线 wheelhouse 见 [packaging/README.md](packaging/README.md)。

## 引用

设计参考 [_Harness Engineering: Anatomy, Architecture, and Evolution of Coding Agents_](https://arxiv.org/abs/2609.00006)（2026）。

```bibtex
@misc{yaro2026,
  title  = {YARO: A Harness for Machine-Vision Development},
  author = {YARO contributors},
  year   = {2026},
  url    = {https://github.com/cz-041017/YARO}
}
```

## 许可证

[MIT](LICENSE)。第三方依赖及其许可见各依赖仓库；主链路不使用 AGPL/GPL 组件。
