# 上手（给新同事的一页纸）

目标：**下载 → 三条命令 → 能跑通一个视觉任务**，且默认状态下你的图像与密钥一个字节都不会离开本机。

## 0. 前置

**如果你只是使用者（最常见）**：拿到 `YARO-win64.zip` → 解压 → 双击 `YARO.exe`。
不需要 Python、不需要 pip、不需要联网、不需要改浏览器设置。**下面的内容你不用看。**

**如果你是维护者/开发**，需要：

- Python **3.10+**（`python --version`）
- 可选：Docker Desktop（真实执行 Python 沙箱需要；不装也能用演示模式与判据）
- 可选：`pip install yaro[keyring]`（把 API Key 交给操作系统钥匙串保管）
- 独立窗口后端是 **pywebview**（`[desktop]`/`[full]` extras 自带）；没有它时自动回退浏览器模式
  `yaro gui`。多人分发时**不要**指望同事自己装它——用 `packaging/` 打出来的 `YARO.exe`。

> 为什么不再"借系统 Edge/Chrome 开窗口"：实测在受管机器上浏览器会静默退出（exit 0、无输出），
> 既开不出窗口也没法诊断。把"能不能开窗口"押在用户浏览器上，是多人分发最不该做的事。
> 现在只有一条路：系统 WebView，且随 exe 一起打包。详见 `packaging/README.md`。

## 1. 安装（维护者）

**Windows（PowerShell）**

```powershell
git clone <repo> yaro; cd yaro
.\scripts\install.ps1
```

**macOS / Linux**

```bash
git clone <repo> yaro && cd yaro
./scripts/install.sh
```

**没有网络的机器**（离线 wheel）：

```powershell
# 构建机（有网）先跑：packaging\build_wheelhouse.ps1
# 然后把仓库 + dist\wheelhouse 拷到目标机器：
powershell -ExecutionPolicy Bypass -File packaging\install_offline.ps1 -Wheelhouse D:\YARO\dist\wheelhouse
```

> ⚠ **一定要让 pip 和 yaro 用同一个解释器**。装到别的 Python 里，表现就是"命令敲了没反应"。
> 稳妥写法是显式绑定 venv 解释器：`python -m pip install ...`（而不是裸 `pip`），
> 装完用 `python -m yaro app --check` 看 `executable` 与 `pywebview` 两行确认。

安装脚本会：建 `.venv` → 装 `yaro[full]`（含独立窗口后端）→ 探测 WebView 是否可用 →
在当前目录生成 `.env`、`demo/`、`contract/CONTRACT.md`。

不想用脚本，等价于：

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .\.venv\Scripts\Activate.ps1
pip install ".[full]"
python -m yaro init
```

## 2. 第一次运行（零 token，不出网）

```bash
yaro doctor        # 自检：python / contract / docker / 隐私边界 / 密钥来源
yaro app           # 独立窗口（推荐：不受浏览器扩展/强制深色影响）
yaro gui           # 或者：浏览器打开 http://127.0.0.1:8765
```

> **为什么推荐 `yaro app`**：浏览器里的页面会被扩展（Dark Reader 之类）、浏览器强制深色、
> 系统对比度主题改写，同一份界面在同事机器上可能完全变样。独立窗口用系统 WebView，
> 没有第三方扩展，配色只由 YARO 决定——这是"多人协作时界面人人一致"最省事的办法。
> 窗口内仍是同一套本地 HTTP 服务，功能、证据、回滚没有任何区别。

在界面里勾选 **演示模式**，按 `Ctrl+Enter`：会跑一遍
`list_files → dataset_scan → dataset_audit → dataset_render → dataset_split → train_plan → train_analyze → 结论`
（合成数据集里故意埋了截断图/缺标注/越界框/重复图，正好演示体检能抓到什么）。
全程**不调用任何模型服务**，可在断网机器上验证安装是否完整。

## 3. 配模型（两种，选一种）

### A. 完全不出网（推荐：隐私优先）

把模型跑在本机，YARO 通过 OpenAI 兼容协议访问它：

```bash
# 例：vLLM 起一个本地 VLM
vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000
# 然后在 .env 里：
#   DEEPSEEK_BASE_URL=http://127.0.0.1:8000/v1
#   YARO_EGRESS=full      # 或 derived：只允许发送渲染后的诊断图，原图仍剥离
```

`YARO_EGRESS=off`（默认）时，任何真实模型调用都会被**明确拒绝**并告诉你如何改。

### B. 云端 DeepSeek（必须显式授权）

```bash
yaro key set                 # 交互式输入，不回显、不进 shell 历史
# .env 里：
#   YARO_EGRESS=text          # 更保守：只发文本，图全部剥离
#   YARO_EGRESS_ALLOW_PUBLIC=1
yaro doctor                  # 确认 egress 一行显示"公网端点 + 已授权"
```

四档策略：`off`（不出网）/ `text`（只发文本）/ `derived`（发诊断图，原图不出）/ `full`（含原图）。

## 4. 密钥怎么存、怎么不泄露

### 4.1 填写入口（四个，任选其一）

Key 只从这四个入口进来，按推荐顺序：

| 入口 | 怎么填 | 适用 |
|---|---|---|
| ① 程序窗口里点左栏 `api key` | 弹窗输入两次 → 保存到本机 | 本机日常使用（最直观） |
| ② `yaro key set` | 交互式输入，不回显、不进 shell 历史 | 终端用户（推荐） |
| ③ `yaro key set --stdin` / 环境变量 `DEEPSEEK_API_KEY` | `echo $KEY \| yaro key set --stdin` 或 CI 注入环境变量 | 脚本、CI、容器 |
| ④ `.env`（项目根或当前目录） | `DEEPSEEK_API_KEY=sk-...` | 团队共享一份配置（`.env` 已被 gitignore） |

窗口里的入口是**唯一**会走 HTTP 的路径，它的边界是：只接受本机回环地址、
`YARO_GUI_PUBLIC=1` 时直接拒绝、响应里只回掩码、关闭对话框立即清空输入框；
Key 仍然存进系统钥匙串或 `<home>/secrets.json`（0600），行为与 `yaro key set` 完全一致。

装 `yaro keyring` 可选依赖时优先存系统钥匙串；否则写 `<home>/secrets.json` 并设 0600。

确认入口是否生效：

```bash
yaro key status   # configured / 来源(env|keyring|file|dotenv) / 掩码 sk-1234…abcd
yaro doctor       # 密钥来源 + egress 生效情况 + 模型端点是否公网
```

窗口/GUI 里看左栏底部 `api key` 一行：`configured`（绿）/ `missing`（红）；点它就能填写或更换。

### 4.2 存储与泄露面

- 读取顺序：环境变量 → 系统钥匙串 → `<home>/secrets.json`（0600）→ `.env`
- `yaro key set` 走 `getpass`，**不进 shell 历史**；`yaro key status` 只显示 `sk-1234…abcd`
- 密钥不会进事件账本、不会进日志；GUI 只显示掩码，输入框关掉即清空；
  异常/500 的 traceback 都会被擦一遍（且默认不回给浏览器）
- `.gitignore` 已含 `.env`、`.yaro/`；发给别人前建议 `git status` 自查

## 5. 验证安装（30 秒）

```bash
python -m pytest -q      # 单元测试
python verify.py         # 端到端自检：事件链/回滚/篡改检测/CAS/闸门/HTTP 全链路
```

## 6. 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `拒绝把任何请求发到 …` | 默认 `egress=off` | 按提示改 `YARO_EGRESS` 或指向本地模型 |
| `目标 … 是公网端点，但未显式授权` | 忘了 `YARO_EGRESS_ALLOW_PUBLIC=1` | 确认隐私风险后再开 |
| GUI 打开是白页 | 包发布时漏了 `package-data` | `pip install -e .` 重装；`yaro doctor` 看 contract 路径 |
| `docker: 不可用` | 没装 Docker | 演示/判据仍可用；真实执行会降级为拒绝而非宿主运行 |
| 端口 8765 被占用 | 已有服务 | YARO 会自动顺延到 8766…；或 `yaro gui --port 9000` |
| `pip install` 卡住/超时 | 网络或公司代理拦 PyPI | 用 `packaging/install_offline.ps1` + wheelhouse 离线装；开发态可以直接 `python -m yaro gui`（pywebview 没装也能用浏览器模式）；要在线装可换镜像 `-i https://pypi.tuna.tsinghua.edu.cn/simple` |
| `yaro app` 没反应 / 没窗口 | ①解释器里没有 pywebview（装到别的 Python 了）；②跑的是旧版本（没有 `app` 子命令）；③缺 WebView2 Runtime | 先 `python -m yaro app --check` 看 `pywebview`/`executable`/`code_version` 三行；确认 `python -c "import yaro;print(yaro.__file__)"` 指向仓库；多人分发请直接用 `packaging/` 打的 `YARO.exe`，不要指望同事装 pywebview |
| `yaro app` 一闪而过 | WebView 后端启动失败 | 终端里会打印完整栈；试 `yaro app --gui mshtml`（旧内核，样式会退化）；再不行 `yaro gui` 一定可用 |
| `pip uninstall yaro` 报 `WinError 5 拒绝访问`（`.pth` 文件） | 有 python 进程正在用这个 venv（上一次 `yaro app` 没关干净），或文件被杀软/只读属性锁住 | 先 `Get-Process python \| Stop-Process`（确认没有正在跑的任务），再手动删 `site-packages\__editable__.yaro-*.pth` 与 `yaro-*.dist-info`、`Scripts\yaro*`，最后 `Remove-Item -Recurse $env:TEMP\pip-uninstall-*`；**完全不做这一步也没影响**（走 `python -m yaro` 时用的是源码） |
