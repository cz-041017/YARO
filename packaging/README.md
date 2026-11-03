# YARO 打包与分发（给同事用的正确方式）

> 一句话：**不要让同事自己装 Python、装依赖、调浏览器。**
> 构建机打一个包，同事解压双击即可 —— 这是唯一能保证"多人普适"的形态。

## 为什么脚本里只有英文输出

`packaging/*.ps1`、`scripts/install.ps1`、`packaging/start-yaro.cmd` **只写 ASCII**。

原因：Windows PowerShell 5.1 在没有 UTF-8 BOM 时按系统 ANSI 代码页读取 `.ps1`。
文件里写了中文，就会先被解码成乱码，再让解析器把引号"吃掉"，报出这种错：

```text
表达式或语句中包含意外的标记"瀹夎渚濊禆锛坹aro[full]"
字符串缺少终止符: "。
```

这不是编码设置问题，是脚本本身的字符集问题。所以约定：**Windows 脚本里不要写中文**，
中文解释放在 `.md` 文档里；`verify.py` 的 `step_packaging` 会检查这几个文件是否纯 ASCII，
写了中文会直接验证失败。

## 为什么不用"借浏览器的 app 模式"

曾经试过调用系统 Edge/Chrome 的 `--app=URL` 来开窗口（零依赖、看起来很划算）。
实测在受管机器上浏览器会**静默退出**（exit 0、无任何输出），既开不出窗口也无法可靠诊断；
把"能不能开窗口"押在用户浏览器的行为/策略上，正是多人分发最不该做的事。

现在的唯一窗口后端是**系统 WebView（pywebview）**；它和 Python 运行时一起被打进
`YARO.exe`，同事机器上不需要装任何东西。

## 两条分发路线

| 路线 | 适用 | 同事需要准备 | 产物 |
| --- | --- | --- | --- |
| **A. 打包 exe**（推荐） | 绝大多数情况 | 什么都不用装 | `YARO-win64.zip` |
| **B. 离线 wheel** | 同事有 Python 3.10+、但没网 | Python | `wheelhouse/` 目录 |

两条路线都由"构建机（能上网）"产出，目标机器全程离线。

---

## 路线 A：打包 `YARO.exe`（推荐）

### 构建机（有网络，Windows x64，Python 3.10+）

```powershell
cd C:\11\YARO
powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1
```

脚本做六件事：建 `.venv-build` → 装 `pywebview` + `openai` + `pyinstaller`（**不装 `-e .`**，PyInstaller 直接从源码树读 yaro）→ 打包 →
**校验产物里带了 `yaro/web` 与 `yaro/data`**（漏了就会白页/丢契约）→
冒烟跑一次 `YARO.exe doctor` → 打 zip。

产物：

```text
dist\YARO\YARO.exe        双击启动（控制台 + 独立窗口）
dist\YARO-win64.zip       直接发这个
```

依赖装不动时（代理/内网）：

```powershell
powershell -ExecutionPolicy Bypass -File packaging\build_wheelhouse.ps1
# 然后 -SkipDeps 复用已装好的 .venv-build，或用 wheelhouse 离线装
```

### 同事机器（无需 Python、无需网络）

1. 解压 `YARO-win64.zip` 到任意目录（建议 `D:\YARO`）；
2. 双击 `YARO.exe`；
3. 出现控制台窗口 + YARO 窗口。关窗口即退出。

首次启动会自动生成演示数据（`<工作区>\demo\dataset`，含故意埋的脏数据）。
左上「+ 新会话」→ 勾**演示模式** → `Ctrl+Enter`，可零 token 跑完整链路。

> 控制台窗口是**故意保留**的：窗口后端出问题时，用户至少能看到一行中文原因，
> 而不是"双击没反应"。想要纯窗口形态，把 `packaging/yaro.spec` 里 `console=False`
> 重新打包即可。

### 前置条件（Win10/11 基本都满足）

- Windows 10/11 x64；
- Microsoft Edge WebView2 Runtime（Win11 自带；Win10 若缺，安装包只有几 MB，
  从 https://developer.microsoft.com/microsoft-edge/webview2/ 装即可）。

---

## 路线 B：离线 wheel（同事有 Python、没网）

### 构建机

```powershell
cd C:\11\YARO
powershell -ExecutionPolicy Bypass -File packaging\build_wheelhouse.ps1
# 默认按 cp311 / win_amd64 下载；同事是 3.12 就加 -PythonVersion 312
```

产物：`dist\wheelhouse\*.whl`（yaro 自己的 wheel + pywebview/openai/numpy/opencv 的 wheel）。

把**整个仓库目录 + `dist\wheelhouse`** 拷给同事（U 盘、内网共享都行）。

### 同事机器

```powershell
cd D:\YARO
powershell -ExecutionPolicy Bypass -File packaging\install_offline.ps1 -Wheelhouse D:\YARO\dist\wheelhouse
```

脚本：检查 Python 版本 → 在 `%LOCALAPPDATA%\YARO\.venv` 建独立环境 →
`pip install --no-index --find-links`（**全程不联网**）→ 生成
`start-yaro.cmd` 与桌面快捷方式 → `yaro init` 初始化工作区。

之后双击桌面「YARO 视觉开发助手」即可。

---

## 网络被拦（`SSLError ... UNEXPECTED_EOF_WHILE_READING`）怎么过

构建机访问 PyPI 失败是这类环境最常见的一关，典型输出：

```text
WARNING: Retrying ... SSLError(SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING] ...'))
ERROR: Could not find a version that satisfies the requirement setuptools>=68 (from versions: none)
```

注意最后一行：**连 `setuptools` 都下不到**。editable 安装默认在隔离环境里构建，
会去下 `setuptools`，所以换了 index 之后仍可能卡在这里——脚本已内置
"失败后自动改用 `--no-build-isolation` 重试"。

按顺序试：

```powershell
# 1) 先探测哪个源能连通（只下载一个很小的 wheel，不安装任何东西）
powershell -ExecutionPolicy Bypass -File packaging\probe_index.ps1

# 2) 用探测到的源构建（以清华源为例）
powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1 `
    -IndexUrl https://pypi.tuna.tsinghua.edu.cn/simple `
    -TrustedHost pypi.tuna.tsinghua.edu.cn

# 3) 如果源连通但构建隔离仍失败，强制复用本地工具链
powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1 `
    -IndexUrl https://pypi.tuna.tsinghua.edu.cn/simple -NoIsolation

# 4) 走 HTTP 代理
$env:HTTPS_PROXY = "http://proxy.corp:8080"
$env:HTTP_PROXY  = "http://proxy.corp:8080"
powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1
```

**一条也走不通**（完全断网/全域名拦截）时，只剩两条路：

- 在一台能上网的机器上跑 `build_windows.ps1`，把 `dist\YARO-win64.zip` 拷回来；
- 或在能上网的机器上跑 `build_wheelhouse.ps1`，把 `dist\wheelhouse` 拷过来，用 `-FindLinks` 构建。

## 依赖安装一行输出都没有 / 看起来卡死

先分清是"在下载"还是"真卡住"：

- pip 26 的进度条用回车原地重绘，在某些终端里回车不换行，**整屏看起来像空白**。
  构建脚本现在给 pip 加了 `--progress-bar off`，每个包下载完成打一行；
- 判断是否真卡住：另开一个 PowerShell，跑
  `Get-Process python | Select-Object Id,CPU,StartTime,WS` —— CPU 在涨就是在解压/编译，CPU 不动多半卡在网络。

绕过办法（按顺序）：

```powershell
# 1) 用已有 .venv（里面通常已经有 numpy / opencv / openai）只补装 pyinstaller + pywebview
.\.venv\Scripts\python.exe -m pip install --progress-bar off pyinstaller pywebview `
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2) 直接复用这个 venv 打包，不要再建 .venv-build
powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1 `
    -BuildVenv .venv -SkipDeps `
    -IndexUrl https://pypi.tuna.tsinghua.edu.cn/simple
```

`-BuildVenv` 就是为"这台机器 pip 装不进新 venv"准备的：换任何已存在的 venv 都行，
但里面必须有 `pyinstaller`（`-SkipDeps` 时脚本不碰依赖；PyInstaller 从源码树读 yaro，
不需要 `-e .` 安装）。

## 验收清单（发给同事前自己先过一遍）

- [ ] 在**一台没装过 Python/没登录过本仓库**的机器上解压并双击 `YARO.exe`；
- [ ] 窗口出现，左栏底部 `build` 行是绿字（说明样式生效、没被外部环境改写）；
- [ ] 「+ 新会话」→ 勾演示模式 → `Ctrl+Enter`，8 轮工具卡跑完，右栏产物库有缩略图；
- [ ] 关窗口后进程退出（任务管理器里没有残留 `YARO.exe`）；
- [ ] `YARO.exe doctor` 能打印 JSON；
- [ ] 断网状态再跑一次（egress=off，本来就不该联网）。

## 常见问题

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 双击 `YARO.exe` 一闪而过 | 控制台被组策略隐藏 / 缺 WebView2 | 用 cmd 运行 `YARO.exe app` 看输出；装 WebView2 Runtime |
| 窗口打开但白页 | 产物里漏了 `yaro/web` | 重新打包，确认 build 脚本第 4 步校验通过 |
| 杀软报毒 | PyInstaller 常见误报 | exe 加白名单；或用路线 B（纯 Python 运行） |
| 同事 Python 版本不匹配 | wheelhouse 是按 3.11 下的 | 构建机加 `-PythonVersion 312` 重新下载 |
| 想连真实模型 | `egress=off` 默认不出网 | `python -m yaro key set` 配 Key，再显式开 `YARO_EGRESS`（见 docs/INSTALL.md） |
