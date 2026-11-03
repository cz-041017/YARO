# CHANGELOG

## v0.7.0（进行中）— LangGraph 重构

按用户 m03658 的 9 条需求把 YARO 从"手写事件溯源循环"迁到 LangGraph 架构（B 方案：
全用预置抽象，但 DeepSeek `reasoning_content` 序列化与视觉/出网闸自持）。旧 `loop/agent.py`
已不被 server.py 引用，`memory/session.py` 保留作 GUI 事件日志，新 `yaro/graph/` 是主引擎。

- **provider 层**（`yaro/provider/`）：`ChatDeepSeek(ChatOpenAI)` 子类补齐 DeepSeek
  `reasoning_content` 逐轮回传——覆盖响应解析层 `_create_chat_result` /
  `_convert_chunk_to_generation_chunk` 把 reasoning_content 抄进 `additional_kwargs`，
  `_get_request_payload` 注入、`_thinking_enabled` 双源、图仅 user 校验；`build_model`、
  `GuardedModel`（invoke 前过 egress + 脱敏）。
- **graph 层**（`yaro/graph/`）：`GraphState` + `build_agent`（agent/tools/finalize 节点，
  预算前置 + 悬空 tool_call 兜底 + `tool_selector` 工具延迟加载）；`context.py` 上下文经济
  原语（`estimate_tokens`/`find_safe_cut`/`summarize_prefix`/`select_schemas` 等）。
- **蒸馏闭环垂直切片**：`distill.py`（label→review(HITL)→train→analyze 循环子图）+
  `runtime.py`（`run_distillation` 串 SqliteSaver + 审批 + 审计）+ `audit.py`（digest 链账本，
  从 SessionStore 抽出）。
- **配套模块**：`skills.py`（SKILL.md 渐进披露）、`goal.py`（持久目标状态机）、
  `approval.py`（PLAN/DEFAULT/YOLO 工具审批）、`agent_eval.py`（故障注入 + 行为评分）、
  `vision/active.py`（`select_hard_examples` 主动学习选难例，子进程跑学生权重）。
- **蒸馏闭环打通**：`distill.py` 加 select_hard 节点（analyze→select_hard→label，`label_node`
  传 `subset=state["hard"]` 只预标注难例）；Skills/Goal 接进 `run_distillation`（analyze 提示
  system/用户分离注入目标 + 技能目录，goal 写回按 `stop_reason=="done" and review.approve and
  metrics.ok is True` 判达成）。
- **server.py 换新 graph 引擎**：`graph/runner.py`（会话桥：`session_to_langchain` +
  `run_agent` 用 `astream(stream_mode="values")` 增量落账、按消息 id 去重（compact 删旧前缀
  不丢已付费事件），映射回 SessionStore 的 user/assistant/tool/run_end 事件，GUI/rewind/fork/
  verify 不感知引擎已换）、`provider/demo.py`（DemoChatModel 适配旧 DemoLLM 剧本）、
  `provider/usage.py`（`usage_from_messages` 从 `response_metadata["deepseek_usage"]` 还原
  hit/miss/out 拆分，缓存价差 50× 记账）、`build_agent` 加 `stop_check` 协作式取消 + `stuck`
  护栏（`max_identical_calls=2` **当前 user 轮内**连续相同 tool_calls 即停，遇 HumanMessage 轮边界即止、不跨轮泄漏）。旧 `loop/agent.py` 不再被
  server.py import（`memory/session.py` 仍作 GUI 事件日志）。
- **MCP 对内消费**：`yaro/mcp.py`（`McpServer`/`McpTool`/`mcp_tools` 包 langchain_mcp_adapters，
  零第三方 import、pydantic schema 转 OpenAI、异常兜 ok=False，`tests/test_mcp.py` 含真实
  FastMCP stdio 端到端）；`config.py` 加 `YARO_MCP_SERVERS`（JSON 数组）→ `Settings.mcp_servers`
  → `server.py` 在 `_run_async` 现连现发现、工具进 ToolRegistry；`/api/config` 的 `mcp_servers`
  字段让 GUI 工具发现路径看到已配置的服务器（连接/配置失败 fail-closed 但降级为本地工具集）。
  `servers_from_config` 对 `args`/`env` 做类型校验（必须 list/dict，否则 ValueError 降级而非
  崩溃穿透）；MCP 工具与内置工具同名时跳过并 warning（不静默覆盖、不炸运行）。
- **vision parts 注入（补旧 `_request_messages` 能力）**：`graph/state.py` 加 `pending_images`、
  `graph/agent.py` 加 `_images_from_records`/`_vision_message`（tools 节点抽图写 pending_images，
  agent 节点按 `egress.allows_images(sensitivity)` 过滤后 `render_vision_parts` 现读现编码注入下一轮，
  剥图加文本 part 告知，注入一次即清空、字节不进状态）；`egress=None` 时 fail-closed（不注入图，
  与压缩节点守卫一致）；`runner.run_agent` / `server._run_async` 透传 `artifacts`，模型又能看诊断图
  而非只看文本摘要。
- **桌面壳窗口诊断**：`desktop.py` 加 `_winforms_available()`——pywebview 6.x 的 EdgeChromium 后端
  需要 .NET Windows Desktop Runtime（`System.Windows.Forms`），只装 .NET 基础运行时会导致
  `webview.start()` 开不出窗口（控制台还在、无 GUI）；`diagnose()` 现报 `winforms` 字段 + 准确 hint，
  `launch()` 检测到缺 WinForms 即抛 `DesktopUnavailable` 回退浏览器模式，而非开一个空白/报错控制台。
- **GUI 一键改出网策略**：`config.py` 加 `update_egress_env()`（只改 `.env` 的 `YARO_EGRESS` /
  `YARO_EGRESS_ALLOW_PUBLIC` 两行、其它键保留）、`server.py` 加 `Harness.set_egress()`（`normalize_mode`
  校验 → 重建不可变 `egress_policy` 立即生效 → 落盘）+ `POST /api/egress` 路由（回环 + 公开模式拒绝，
  镜像 `/api/key`）；GUI 的 Key 对话框加「出网策略」区（4 档下拉 + 公网复选框 + 保存按钮 + full+公网
  二次确认）。默认仍 `egress=off` fail-closed，用户可一键放宽到 text/derived/full。
- 每个模块均 spawn 子智能体对抗性审查并修完致命/高危项；全量 `pytest -q` 318 passed、
  `verify.py` 19/19（截至本条目）。
- 尚未接线：max_cost/max_wall 预算；打包 app 已出 `dist/YARO-win64.zip`（v0.7.0，含
  langgraph/langchain 隐藏导入 + doctor 冻结安装检查不再误报 egg-info 旧版本），换机验收属文档化后续项。

## v0.6.2 — 未知型号直接报错 + GUI 里的 API Key 入口

用户要求："未知型号直接报错，并且在 gui 中加入 api_key 添加入口。"

- **未知型号 fail-fast（不再猜价）**：`contracts.cost_usd()` 遇到价格表里没有的型号直接抛
  `ContractError`（v0.6.1 是"取最贵档计价"，v0.6.0 是"返回 0"）；新增
  `contracts.ensure_known_model()`，`Settings.load()` 在启动时校验 `YARO_MODEL` /
  `YARO_MODEL_PLAN`，配错模型名不会跑到一半才炸。价格表同时是"认识的型号清单"。
- **GUI 的 API Key 入口**：窗口左栏 `api key` 一行可点，弹出对话框（两次输入防手滑，
  保存到系统钥匙串或 `~/.yaro/secrets.json` 0600，可一键清除）。边界：
  - 新增 `POST /api/key`，**只接受回环地址**，且 `YARO_GUI_PUBLIC=1` 时直接 403；
  - 响应只回掩码与来源，不写事件、不写日志、不落 traceback；
  - 关闭对话框立刻清空输入框，避免从开发者工具里读到刚输入的 Key；
  - Key 读取顺序（env → keyring → 本机文件 → .env）与 `yaro key set` 完全一致。
- 文档：README「填写 API Key（入口在哪）」改为四个入口并写清 GUI 的边界；
  `docs/INSTALL.md` 4.1/4.2 同步；`yaro init` 的提示不再说"界面里没有密码框"。
- 测试：`tests/test_contracts.py` 改为断言未知型号抛 `ContractError`（并测
  `ensure_known_model`）；`tests/test_server.py` 新增 `POST /api/key` 的写入/清除/拒绝
  三条用例（monkeypatch 掉 keyring，统一落到临时 home）；`verify.py` 的配置步与 HTTP 步
  各加一条断言。

## v0.6.1 — 商用标准代码审查后的第一轮修复

四路子 agent 按"商用 agent"标准做了全量静态审查（核心 / 执行与视觉 / 服务前端打包 / 商用尽调）。
结论见 README「状态（诚实版）」：**Alpha**，单人本机可用、模型链未验证、不可直接给团队用。
这一版先修 P0/P1 里能静态确定的部分（本机 pwsh 不可用，全部未运行验证）：

- 隐私 P0：`Agent._maybe_compact()` 摘要请求之前也走 `egress.filter_messages()`；
  以前压缩摘要直接出网，`egress=text/derived` 档下绝对路径、工具原文、疑似密钥会外发。
- 成本 P0：`Usage.cost_usd()` 未知型号不再返回 0（v0.6.1 先改为 fail-closed 取最贵档，
  v0.6.2 又改为**直接报错**），否则改一下 `YARO_MODEL` 就能让 `max_cost` 永不触发；GUI/会话成本改为**跨 run 累加**。
- 压缩 P0：`append_compaction()` 失败时先把付过费的摘要落成 `compaction_error` 事件再抛，
  不再"钱花了、上下文没降、下次再付一次"。
- 执行层 P0：`yaro.exec.__init__` 从错误模块导 `which` 导致 `import yaro.exec` 直接失败；
  `plan.py` 生成脚本的标签路径改为支持 `images/<split>/`（以前 split 布局下标签全读不到，
  整批当背景训练、loss 照降还被当成功）；`deploy_run` 的 `torch.load` 改 `weights_only=True`。
- 执行层 P1：`runner` 子进程**不再继承宿主环境变量**（只放白名单 + 显式 overrides，
  生成脚本 `print(os.environ)` 拿不到 API Key）；输出改保留**尾部**且 stdout/stderr 独立计数；
  POSIX 加入 `killpg`；超时返回 -9 不再伪装成功；截断时提示"只保留末尾 N 行"。
- 视觉 P1：`tiny_box` 的像素宽高交叉项修正（`w*width` / `h*height`）；
  `Dockerfile` 引用但从不生成的 `serve.py` 现在会生成；`deploy_run` 在缺延迟证据时
  不再报 `ok=True`（"只导出不算落地"）。
- 服务端安全：`--host` 非本机地址必须显式 `YARO_GUI_PUBLIC=1` 才允许（并打印风险警告）；
  500 响应不再把 traceback 回给浏览器（改 request id，排障用 `YARO_DEBUG=1`）；
  `/api/sessions?since=abc` 这类参数错误回 400 而不是 500。
- 前端安全：`esc()` 补单引号转义；会话 id / event id / sha 一律走 `attr()`；
  `.lightbox[hidden]` 不再被 `display:flex` 覆盖（以前打开页面就全屏黑罩）；
  轮询里 `verified` 为 `null` 时显示 `unverified` 而不是假装已验证。
- 桌面壳：删掉"WebView2 失败就静默退回 mshtml(IE11)"的降级路径 —— 宁可报错。
- 闸门 P2：`scan_files()` 支持目录（`/api/gates?path=.` 以前拿到目录后直接 `ok=True`，
  闸门等于没有）；新增 `expand_paths()`。
- 文档：新增「填写 API Key（入口在哪）」完整章节（README / INSTALL 4.1、4.2 / doctor / init /
  GUI 左栏点击提示）；`pyproject` 的 urls 从 `your-org` 改成真实仓库。

## v0.6.0 — 从"生成计划"到"真的执行"（蒸馏链路的第一段）

用户 m02550/m02564 的结论：助手要"给一个图片集，就真的把标注、清洗、训练、调优跑完"，
而且这不该是某个项目的专用脚本，应该是**多数视觉项目通用的 harness**；训练只在 CUDA 机器启用，
用大模型的能力把数据变成"快速可用的小模型"（蒸馏）。这一版把执行层补上。

- 新增 `yaro/exec/`（受控执行层，零依赖）：
  - `env.py`：`capabilities()` 探测 python/packages(torch/torchvision/onnx/onnxruntime/cv2/numpy/PIL)/
    `cuda{available,devices,note}`/`can_train`/`can_prelabel_*`/`can_export_onnx`；
    `suggest_install()` 用 `sys.executable -m pip`（不裸 `pip`）；`describe()` 供 doctor/GUI 显示。
  - `runner.py`：`RunSpec`/`RunOutcome`/`run()`/`run_python()`；Popen + 双泵线程、
    `PYTHONUNBUFFERED=1`、超时**杀整棵进程树**（Windows `taskkill /T`）、`tail()` 截断。
- 新增 `yaro/vision/prelabel.py`（真预标注）：`torchvision`（FCOS COCO）/`onnx`/`contour`
  （Otsu + 形态学 + 外轮廓）三后端；写 `proposals/` + `report.json`；`review_queue()` /
  `apply_proposals()`（`overwrite=False` 时不覆盖人工标注）。
- 新增 `yaro/vision/train_run.py`（真训练）：生成 `train_generated.py` 后子进程执行；
  `TrainConfig`（model/epochs/imgsz/batch/lr/device/max_minutes…）、`TrainResult`；
  **默认要求 CUDA**（`device="cpu"` 才允许冒烟）；每轮写 `metrics.jsonl`（loss/val_loss/map50/
  classes_evaluated/lr/seconds）、`best.pt`/`last.pt`；mAP 由脚本内 greedy IoU 匹配自算；
  `max_minutes` 到点 `budget_stop`；训练后用 `plan.analyze_training` 出诊断。
- 新增 `yaro/vision/tune_run.py`（真调优）：`run_trials()` 按网格/随机真的跑多组，
  每组独立目录，总体时间预算，`trials.jsonl` + `best.json`，噪声底线由 `plan.summarize_trials` 判。
- 新增 `yaro/vision/deploy_run.py`（真部署）：`deploy_run()` 真的 `torch.onnx.export` 导出，
  再用 onnxruntime 真的测 p50/p95/mean（warmup+多次），写 `bench.json`/`deploy.json`；
  给了 `latency_target_ms` 才判 `pass/fail`。
- 工具层从 10 个扩到 14 个：新增 `prelabel_run` / `train_run` / `tune_run` / `deploy_run`
  （惰性 import，`build_default_tools` 与 `build_tool_catalog` 同步）。
- 新增 `yaro pipeline` 命令：预标注 → 体检 → 切分 → 训练 → 调优 → 导出测速；
  支持 `--skip-*` 分阶段、`--cpu` 冒烟、`--latency-target` 判定、`--max-minutes`/`--tune-minutes` 预算。
- `pyproject.toml` 新增 `ml` extra（torch/torchvision/onnx/onnxruntime），不放进 `full`（体积太大）。
- `verify.py` 视觉步骤追加：真执行模块可导入、`train_support()` 返回布尔、`available_backends()` 非空、
  `export_support()` 返回布尔、工具清单含 12 个视觉工具名。
- 测试 `test_vision.py` 更新为 14 个工具 + 新增 4 个真执行工具名断言。
- 版本 0.6.0。

已知边界（诚实写在这里）：本轮全部代码**没有运行验证**（开发环境 `pwsh` 返回
`0xC0000142`，无法执行任何命令）；真训练需要用户机器上有 CUDA + torch，否则 `train_support()`
会返回 `ok=False` 并给出安装命令——这是设计行为，不是崩溃。
蒸馏闭环的编排（teacher→student→主动学习再标注）本轮只具备零件，编排留到下一版。

## v0.5.2 — 构建脚本抗住"PyPI 被拦"

- 现象：`build_windows.ps1` 在 `[2/6]` 失败，`pip` 对 `pypi.org` 报
  `SSLError(SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING] ...'))`，
  最后卡在 `Could not find a version that satisfies the requirement setuptools>=68`。
- 真因：网络层拦/改写 TLS（代理或 TLS 检查），不是 pip 配置错。
  且 editable 安装默认走 build isolation，需要联网下 `setuptools`，所以只换 `-i` 仍会卡。
- 修（不改变构建逻辑，只加"能不能过网"的出口）：
  - `build_windows.ps1` 新增 `-IndexUrl` / `-TrustedHost` / `-FindLinks` / `-NoIsolation`；
    index 参数以 `PIP_*` 环境变量下发（这样 build isolation 的子进程也能继承）；
  - 依赖安装失败后**自动用 `--no-build-isolation` 重试一次**，仍失败才报错；
  - 失败时打印五条可执行的下一步（探测源 / 换源 / 代理 env / 离线 wheelhouse / 换构建机）。
  - 新增 `packaging/probe_index.ps1`：只下载一个很小的 wheel，逐个探测官方源与常用镜像，
    打印哪个可用，并直接给出该用的构建命令。
- `packaging/README.md` 新增「网络被拦（SSL EOF）怎么过」；`verify.py` 的分发包清单加上
  `probe_index.ps1`；版本 0.5.2。
- 实机续修（同版追加）：
  - 构建不再安装 `-e .`：PyInstaller 按 `yaro.spec` 的 `pathex` 直接从源码树打包 yaro，
    editable 安装在 pip 26 + 受管机器上会**静默卡在构建隔离**（连 `Obtaining` 都不打印）；
    `[2/6]` 改为按名字装 `pywebview` + `openai` + `pyinstaller`（`-WithCv` 才加 numpy/opencv）；
  - 所有 pip 调用加 `--no-cache-dir`（被 `Ctrl+C` 中断的 pip 会留下缓存锁，下一次运行静默等待）；
  - PyInstaller 前显式设 `PYTHONPATH=<repo>`，并把即将执行的 pip 命令打印出来（`> pip install ...`）；
  - 新增 `-BuildVenv`（复用已有 venv）与 `-WithCv`（把 OpenCV 打进 exe）；
  - `packaging/README.md` 增加「依赖安装一行输出都没有 / 看起来卡死」一节。

## v0.5.1 — 修 Windows 脚本的中文编码崩解析

- 现象：`powershell -File packaging\build_windows.ps1` 直接报
  `表达式或语句中包含意外的标记"瀹夎渚濊禆锛坹aro[full]"` / `字符串缺少终止符`。
- 真因：Windows PowerShell 5.1 在**没有 UTF-8 BOM** 时按系统 ANSI 代码页读 `.ps1`，
  脚本里的中文被解码成乱码后把后面的引号吃掉，于是整段解析失败。
  这不是环境编码设置问题，是脚本字符集问题。
- 修：`packaging/build_windows.ps1`、`build_wheelhouse.ps1`、`install_offline.ps1`、
  `scripts/install.ps1`、`packaging/start-yaro.cmd` **全部改成纯 ASCII 输出**（英文提示），
  中文解释留在 `.md` 文档里；`start-yaro.cmd` 同时去掉了 `echo` 里的中文（cmd 按控制台
  代码页解析，同样会乱码）。
- 加回归封条：`verify.py` 的 `step_packaging` 新增断言——上述 Windows 脚本必须是纯 ASCII，
  写了中文就验证失败；`packaging/README.md` 里写清了这条约定和那个报错长什么样。
- 版本 0.5.1；静态资源 `?v=0.5.1`。

## v0.5.0 — 换方案：删掉浏览器 app 模式，改用“打包好的 exe”

- **删除**：`yaro app` 不再尝试调用系统 Edge/Chrome 的 `--app=URL` 开窗口。
  v0.4.2→v0.4.3 的 6 种组合穷举在受管机器上只得到 `exit=0` + 空输出：窗口开不出来，
  也拿不到任何可诊断的信息。把“能不能开窗口”押在同事的浏览器上，本身就是错误的依赖方向。
  `yaro/desktop.py` 现在只有一个后端：系统 WebView（pywebview）。
- **新增分发形态（本次的核心）**：`packaging/` 目录 —— 用 PyInstaller 把 Python 运行时、
  依赖、窗口后端和 `yaro/web`、`yaro/data` 一起打进 `YARO.exe`。同事拿到
  `YARO-win64.zip` 解压双击即可，**目标机不需要 Python、不需要 pip、不需要联网**。
  - `packaging/yaro.spec`（onedir；`console=True` 故意保留控制台，出问题能看到一行中文原因）
  - `packaging/build_windows.ps1`（建 `.venv-build` → 打包 → **校验产物里带 `web/` 与 `data/`** → `YARO.exe doctor` 冒烟 → 出 zip）
  - `packaging/build_wheelhouse.ps1` + `packaging/install_offline.ps1`（有 Python 但没网的机器：离线 wheel + 建独立 venv + 桌面快捷方式，全程 `--no-index`）
  - `packaging/start-yaro.cmd`、`packaging/entry.py`、`packaging/README.md`（含事故记录、两条路线、验收清单、FAQ）
- `app --check` 输出改为 `ok/backend/pywebview/webview2/frozen/python/platform/executable/code_version/dist_version/stale_install/package/hint`，并给出结论行；`--prefer` 移除；`DesktopUnavailable` 时明确建议用打包 exe，然后回退 `yaro gui`。
- 新增回归测试 `test_no_browser_app_mode_left`，并让 `verify.py` 的 `step_packaging` 断言
  `desktop.py` 里不再出现 `--app=`（防止再退回去），同时校验 `packaging/` 七个文件与
  `yaro.spec` 的资源清单。
- 文档：`README.md` 快速开始/对照表/分发章节改为“exe 是多人分发唯一形态”；
  `docs/INSTALL.md` 第 0 节区分“使用者”和“维护者”；`packaging/README.md` 写成给构建机看的手册。
- `MANIFEST.in` 纳入 `packaging/` 与 `verify.py`；版本 0.5.0，静态资源 `?v=0.5.0`。

## v0.4.3 — "第二次启动没有窗口"：浏览器实例复用导致秒退

- 现象：第一次 `yaro app` 窗口正常；关掉后（或还留着一个窗口时）再启动，**没有窗口弹出来**。
- 真因：Edge/Chrome 用 `--user-data-dir=<home>/window-profile` 时，如果那个配置目录
  已被一个还在运行的实例占用，新进程会**把启动请求转交给旧实例然后立刻退出**。
  YARO 看到浏览器进程退出，就以为"用户关窗了"，立刻停掉 HTTP 服务 → 新窗口要么没出现，
  要么出来就是连接失败。
- 修：真实原因是**单靠一种启动参数猜不出来**。现在按"最可能成功"的顺序依次试
  6 种组合：`app 模式(私有配置)` → `app 模式(全新配置)` → `普通窗口(全新配置)`，
  然后换另一个浏览器（Edge / Chrome）再走一遍；每种组合把浏览器自己的 stdout/stderr
  抓到 `<home>/chrome-launch-N.log`，秒退时打印它的最后几行。
- 第一个"窗口还活着"的组合被采用；6 种全秒退才判定失败，抛 `DesktopUnavailable`，
  CLI **自动回退浏览器模式**（功能一样，只是用标签页），并把日志路径打出来。

## v0.4.2 — 窗口不再依赖 pip：Edge/Chrome app 模式兜底

- 用户实测：`python -m pip install -e ".[full]"` 直接**卡住**（网络/代理导致 pip 挂死），
  于是 `yaro app` 永远起不来。教训：把"能不能用窗口"绑定在"能不能装上一个 Python 包"上，
  对多人分发是不可接受的设计。
- `yaro app` 现在有两级后端：
  1. **pywebview**（可选，装了最干净）；
  2. **Edge / Chrome 的 app 模式**（**零依赖**）：用系统已有浏览器以
     `--app=URL --disable-extensions --user-data-dir=<私有配置目录>` 起一个
     无地址栏、无第三方扩展、配置独立的窗口。同事机器上只要装了 Edge 就有窗口。
  两者都不可用才回退 `yaro gui`。
- `app --check` 新增 `chromium` 一行（找到的 Edge/Chrome 路径），`ok` 判定改为
  "pywebview **或** chromium 任一可用"；不再提示"必须装 pywebview"。
- `app --prefer webview|chromium` 可强制后端，便于排查是哪一层的问题。
- 私有配置目录 `<home>/window-profile`：不继承任何人浏览器里的扩展与强制深色设置。
- 诊断字段调整：`code_version` / `dist_version` / `stale_install` / `chromium`。
- `tests/test_desktop.py` 更新：模拟"无任何后端"时 `available()` 为 False、
  `launch()` 在碰 `Settings` 之前抛 `DesktopUnavailable`。

## v0.4.1 — `yaro app` 不再"静默无反应"

- 用户反馈：按文档跑 `pip install "pywebview>=5.0"` + `python -m yaro app` **没有任何反应**。
  这属于最糟的失败模式：既没窗口、也没报错，无法判断卡在哪。
- 新增 `python -m yaro app --check`：不启窗口，直接打印 `ok / pywebview 版本 / 后端 /
  WebView2 运行时 / python / platform / executable / yaro_version / package`——
  一次回答"依赖装没装、运行时缺不缺、跑的是哪份代码"。
- `launch()` 现在分四步打日志（`[1/4] 检查后端 → [2/4] 准备服务 → [3/4] 服务就绪(打印 URL)
  → [4/4] 打开窗口`），全部 `flush=True`；首次导入 pythonnet/WebView 慢时也能看出卡在哪一步。
- Windows 上默认后端（edgechromium）启动失败时，自动用 `mshtml` 旧内核重试一次；
  仍失败才抛 `DesktopUnavailable`（含原因 + WebView2 安装提示 + `yaro gui` 回退）。
- `cmd_app` 兜住所有异常并打印完整 traceback（以前只捕 `DesktopUnavailable`，
  其它异常在部分终端里表现为"没反应"）；`--gui <backend>` 可手动指定后端。
- `yaro doctor` 的 `desktop` 检查改为显示后端与 WebView2 状态，便于远程排障。
- **新增"陈旧安装"检测**：`app --check` 现在分别打印 `code_version`（代码版本）与
  `dist_version`（`.venv` 里 pip 记录的版本）。用户实测就是这种情况——代码已到 0.4.1，
  但 `.venv` 里装的是 0.1.0，于是 `yaro` 控制台脚本可能仍在跑旧代码（旧版本没有 `app`
  子命令，敲下去就是"没反应"）。同时给出与当前解释器绑定的修复命令（`"<python>" -m pip install -e ".[full]"`）。
- `yaro doctor` 增加 `install` 检查：安装版本与代码版本不一致时给出明确提示。
- `docs/INSTALL.md` FAQ 增加"`yaro app` 没反应"的四个原因与排查顺序；正文强调
  "pip 必须与 yaro 用同一个解释器"（用 `python -m pip`，不要裸 `pip`）。
- 新增 `test_desktop.py::test_diagnose_shape`。

## v0.4.0 — 独立窗口（`yaro app`）：界面不再取决于"谁的浏览器"

- **背景**：多人项目里同一个页面在不同机器上长得不一样——Dark Reader 之类的扩展、
  浏览器 `#enable-force-dark`、系统对比度主题都会改写页面。这不是 CSS 能修的，
  因为改写发生在浏览器里。
- **新增 `yaro app`**：用系统 WebView 开一个独立窗口（Windows=Edge WebView2、
  macOS=WKWebView、Linux=WebKitGTK），内部仍是同一个本地 HTTP 服务与同一套静态页面，
  只是没有第三方扩展介入，配色只由 YARO 决定。`yaro gui` 保留为浏览器模式回退。
- 新增可选依赖 `yaro[desktop]`（`pywebview>=5.0`）与推荐组合 `yaro[full]`；
  **没装也能用**：`yaro app` 检测不到后端会打印原因与修复命令，然后自动回退到 `yaro gui`。
- `install.ps1` / `install.sh` 改为默认装 `full` 并探测 WebView 可用性（`-NoDesktop` /
  `--no-desktop` 可跳过）；探测失败只提示、不中断安装。
- `docs/INSTALL.md` 增加"独立窗口需要系统 WebView"的前置与 `yaro app` 说明；
  README 增加 `app` vs `gui` 对照表。
- `verify.py` 的打包步骤新增断言：桌面壳存在、降级异常存在、pyproject 有 `desktop` extra、
  CLI 有 `app` 子命令、两个安装脚本都探测桌面后端、INSTALL.md 写了 `yaro app`。
- 新增 `tests/test_desktop.py`（5 个离线用例，覆盖"缺后端时不崩、能回退"这条分发关键路径）。

## v0.3.4 — 页面自报"实际渲染背景色"

- 用户反馈：左栏 `build` 已显示 `v0.3.3 · light`，但页面**看起来仍是黑的**。这说明
  版本与主题选择都生效了，问题在"样式是否真的落到浏览器"或"浏览器是否在反转页面"。
- 左栏 `build` 行现在显示三段：`v0.3.4 · light · rgb(242, 244, 247)`——第三段是
  `getComputedStyle(document.body).backgroundColor` 的**真实值**。
  - 显示期望值（`rgb(242, 244, 247)`）且绿字：CSS 生效，页面若仍黑 = 浏览器/扩展在做深色反转；
  - 显示 `rgb(25, 27, 31)`：拿到的是旧 CSS（缓存）；
  - 显示 `rgba(0, 0, 0, 0)` 或 `?`：CSS 变量没解析（文件损坏）。
  悬停该行有对应解释。

## v0.3.3 — 让"改了却没生效"不再靠猜

- **资源带版本号**：`index.html` 引用 `/assets/app.css?v=0.3.3`、`/assets/app.js?v=0.3.3`、
  `logo.svg?v=0.3.3`。之前浏览器可能复用旧 CSS（旧服务没发 `no-store` 时会被启发式缓存），
  于是"后端换了主题、页面还是黑的"。带 `?v=` 的 URL 一定重新拉取。
- **左栏新增 `build` 行**：显示 `v0.3.3 · light/dark`，一眼看出后端版本与当前主题，
  免去"到底是没生效还是看不清"的猜谜。
- `verify.py` 增加断言：静态资源引用必须带 `?v=`，`index.html` 必须有 `foot-build`，
  `app.js` 必须同时更新 `foot-build` 与主题。

## v0.3.2 — GUI 默认浅色高对比 + 双主题

- 用户反馈"背景和 UI、字体的对比度太低，根本看不清"。深色底反复提亮都不如白底黑字稳，
  于是把**默认主题改成浅色高对比**（`--fg #0d1014` on `--bg #f2f4f7`，正文 13.5px），
  顶栏新增「主题」按钮切到深色（深灰底、不是纯黑），选择存 `localStorage`，下次打开保持。
- 把散落的硬编码颜色全部收进 CSS 变量（徽章/提示描边、正文强调、代码块、图片衬底、
  滚动条、主按钮 hover、输入框与下拉框），避免"换主题只换一半"。
- `verify.py` 静态接线检查新增：双主题变量存在、`btn-theme` 按钮接线、`applyTheme` 持久化逻辑。

## v0.3.1 — 首跑自检修复（4 个失败项）+ GUI 对比度

首跑实机结果：`verify.py` 14/18 通过（Python 3.11.17）。四个失败逐项修掉：

- **`verify.py` 回滚断言写错**：rewind 只能指向当前 head 的祖先，原来试图"穿越"回被丢弃的
  未来事件（`事件 … 不在分支 'main' 的历史上`）。改为回到 `events[1]` 后继续追加一条主线事件。
- **截断图没被检出**：截断的 PNG 仍有合法 IHDR，只读头会当成好图。`probe_image` 现在判
  完整性（PNG `IEND` / JPEG `EOI` / GIF trailer / BMP 声明长度），`scan_dataset` 透传
  `truncated`，`audit_dataset` 据此报 `corrupt_image`。
- **坏图让整批渲染失败**：`render_dataset` 原来只 catch `ValueError`，截断 PNG 抛的是
  `zlib.error`，导致 `dataset_render` 整体 ok=False、GUI 一张缩略图都没有。改为逐张隔离
  （坏图进 `skipped`，其余照常渲染）。
- **`demo/dataset` 缺失**：没跑 `yaro init` 就直接 `yaro chain` 会报"目录不存在"。
  `Harness` 启动与 `yaro chain` / `yaro dataset` 首次使用时自动铺演示资源（幂等、零 token）。
- 演示数据集的"越界框"同时埋 `cx>1` 与 `w>1` 两种坏法，`box_out_of_range` 与
  `box_center_out_of_range` 都能被体检演示出来。
- **GUI 主题提亮**：背景 `#0a0a0b` → `#191b1f`，面板/分隔线/次要文字整体提高对比度，
  字号 12.5 → 13px，修掉若干过暗的硬编码颜色（滚动条、位置信息、徽章描边）。

### 第二轮（`pytest` 首跑：7 failed / 111 passed）

- **`compose_contact_sheet` 行切片少了 `*3`**：联系人表每行只取 `width` 字节而不是 `width*3`，
  于是 `encode_png` 报"每行应为 2256 字节，实际 752"；连带 `dataset_render` 整体失败、
  GUI 一张缩略图都没有、`/api/artifacts` 为空。这是 4 个测试失败的同一个根因，已修
  （`rows_bytes = canvas[y*width*3 : (y+1)*width*3]`）。
- **`write_split` 契约与测试不一致**：实现同时返回 `yaml` 键，测试断言精确等于三个 split。
  以实现为准修正断言：`{"train","val","test"} <= set(written)` 且 `"yaml" in written`；
  verify 同步。
- **`export_voc()["files"]` 是计数不是列表**：测试与 `LabelConvertTool` 都按 list 处理。
  工具层改为兼容 int/list 两种形态，测试改为 `voc["files"] > 0`。
- **缺少 Key 的报错文案**补上环境变量名（`DEEPSEEK_API_KEY`），既满足测试也更好排障。

## v0.3.0 — 全链路视觉开发（数据 · 标注 · 训练 · 调优 · 部署）

### 新增

- **`yaro/vision/`（零依赖视觉链路包）**：
  - `png.py`：纯标准库 PNG 编解码（filter 0–4 + Paeth）、最近邻缩放、画框、联系人表；
  - `synth.py`：确定性合成数据集（`demo/dataset`），故意埋 6 类脏数据用于演示体检能力；
  - `dataset.py`：YOLO/COCO 扫描、体检（15 类问题，error/warn/info 分级）、分层切分（固定种子）、
    重复图隔离（move 不删）、`manifest.jsonl`；
  - `annotate.py`：标注渲染 + 联系人表、COCO/VOC 导出、预标注提案与人工审核队列；
  - `plan.py`：训练计划（recipe + 可执行脚本 + 验收清单）、日志解析与诊断（平台期/过拟合/指标缺失）、
    调优试验计划（grid/random + 噪声底线）、部署计划（ONNX 导出 + p95 基准 + Dockerfile + 模型卡）；
  - `tools.py`：10 个 agent 工具（`dataset_scan` / `dataset_audit` / `dataset_split` / `dataset_dedupe` /
    `dataset_render` / `label_convert` / `train_plan` / `train_analyze` / `deploy_plan` / `tune_plan`）。
- **CLI**：`yaro chain`（scan→audit→render→split→train_plan，确定性、不花 token）、
  `yaro dataset scan|audit|render|split|dedupe|convert`（单步）。
- **DemoLLM 剧本改为全链路**：`list_files → dataset_scan → dataset_audit → dataset_render →
  dataset_split + dataset_dedupe → train_plan → train_analyze → 三段式结论`，8 轮 8 次工具调用。
- **`ensure_demo_assets`** 同时生成 `demo/dataset`（12 张图 + 故意脏数据）。
- **测试**：`tests/test_vision.py`；**verify.py** 新增第 18 步「视觉全链路」。

### 变更

- `build_default_tools` / `build_tool_catalog` 懒加载注册 10 个视觉工具（不破坏最小可用集）。
- GUI 欢迎页与任务占位符改为全链路表述；`config.demo` 暴露数据集路径与埋雷清单。
- 版本 0.2.0 → 0.3.0。

### 未做（有意）

- 训练与推理**不在 agent 进程内执行**：仍走沙箱 + 计划脚本，避免把 torch/CUDA 拖进核心依赖；
- 不做近重复（embedding）去重、不做自动标注模型推理（只生成提案脚本与审核队列）。

## v0.2.0 — 隐私边界与可分发安装

### 新增

- **出网策略（`yaro/egress.py`）**：`off`（默认不出网）/ `text` / `derived` / `full` 四档；
  公网端点需 `YARO_EGRESS_ALLOW_PUBLIC=1` 显式授权；消息深拷贝脱敏（绝对路径、密钥形态字符串）。
- **图像敏感度标签**：`ToolResult.sensitive`；`image_probe` 的原始输入图默认永不出网。
- **密钥保护（`yaro/secrets.py`）**：环境变量 → 系统钥匙串 → `<home>/secrets.json`(0600) → `.env`；
  `yaro key set|set-stdin|status|clear`；GUI/doctor 只显示来源与掩码；500 traceback 与线程异常均擦除密钥。
- **`yaro init`**：建工作区、铺演示图、写 `contract/CONTRACT.md` 与 `.env`；幂等，`--force` 覆盖。
- **打包**：`yaro/data/CONTRACT.md` 随包发布（修复 wheel 安装后契约丢失）；`MANIFEST.in`；
  `scripts/install.ps1` / `install.sh` 一键建 venv + 安装 + init；`docs/INSTALL.md` 给同事的一页纸。
- **测试**：`tests/test_egress.py`、`tests/test_secrets.py`。
- **doctor** 新增 `egress` 检查（公网端点未授权会报配置矛盾），密钥项显示来源与掩码。

### 变更

- `Settings` 新增 `egress` / `egress_allow_public` / `api_key_source`（均有默认值，向后兼容）；
  `contract_path` 支持仓库态与安装态两种位置。
- `DeepSeek` 与 `Agent` 均接受 `egress=`；`DeepSeek.complete()` 发出请求前做端点检查。
- GUI：顶栏/左栏显示当前出网策略；`off` 模式下非演示运行给出明确提示。
- CONTRACT.md 增加「隐私与出网（不可协商）」一节。
- 版本 0.1.0 → 0.2.0。

## v0.1.0 — P0 骨架

- 事件溯源会话树（append / rewind / fork / verify）、内容寻址产物库、Docker 沙箱、
  五道闸门、评估器噪声阈值、标定判据、DeepSeek 传输层（reasoning_content 不变式）、
  零依赖 Web GUI（三栏 / 演示模式 / Codex 风格）。
