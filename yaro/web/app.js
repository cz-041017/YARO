/* YARO GUI 前端 —— 无框架、无构建步骤。
 *
 * 数据流只有一条：轮询 /api/sessions/{id}?since=<seq> 拿增量事件 → 重绘。
 * 所有状态都在 DOM 之外的这个 S 对象里，重绘是纯函数式的，刷新页面不会丢东西
 * （因为真正的状态在服务端的事件链里）。
 */
"use strict";

const S = {
  config: null,
  sessions: [],
  current: null,      // session_id
  events: [],         // 累积事件（按 seq）
  seq: 0,
  lineage: [],        // 当前分支的事件 id 顺序
  head: "",
  verified: true,
  running: false,
  stats: null,
  pollTimer: null,
  welcome: true,
};

/* ------------------------------------------------------------------ 工具 */

function $(id) { return document.getElementById(id); }

function esc(text) {
  return String(text == null ? "" : text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

/* 属性值专用：会话 id 来自目录名，不转义就是持久化 XSS（code review C-4）。 */
function attr(value) { return esc(value); }

async function api(path, options) {
  const opts = Object.assign({ headers: {} }, options || {});
  if (opts.body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(opts.body);
  }
  const response = await fetch(path, opts);
  let data = null;
  try { data = await response.json(); } catch (err) { data = null; }
  if (!response.ok) {
    const message = data && data.error ? data.error : ("HTTP " + response.status);
    throw new Error(message);
  }
  return data;
}

function toast(message, isError) {
  const node = $("toast");
  node.textContent = message;
  node.className = "toast" + (isError ? " err" : "");
  node.hidden = false;
  clearTimeout(toast._timer);
  toast._timer = setTimeout(function () { node.hidden = true; }, isError ? 4200 : 2400);
}

function clock(ts) {
  if (!ts) return "--:--:--";
  const d = new Date(ts * 1000);
  return [d.getHours(), d.getMinutes(), d.getSeconds()]
    .map(function (n) { return String(n).padStart(2, "0"); })
    .join(":");
}

function money(usd) {
  const v = Number(usd || 0);
  if (v >= 1) return "$" + v.toFixed(2);
  if (v >= 0.01) return "$" + v.toFixed(3);
  return "$" + v.toFixed(4);
}

function num(n) {
  const v = Number(n || 0);
  return Math.abs(v) >= 100 ? String(Math.round(v)) : String(Math.round(v * 1000) / 1000);
}

function fileSize(bytes) {
  const v = Number(bytes || 0);
  if (v > 1024 * 1024) return (v / 1024 / 1024).toFixed(1) + "MB";
  if (v > 1024) return Math.round(v / 1024) + "KB";
  return v + "B";
}

function artifactUrl(sha) { return "/api/artifact/" + encodeURIComponent(sha); }

/* 极简 markdown：只支持 **粗体**、`代码`、- 列表、空行分段。
   够用就行——渲染超过这个复杂度的内容说明模型在写文档，而不是在干活。 */
function mdLite(raw) {
  const text = String(raw == null ? "" : raw);
  const blocks = text.split(/\n{2,}/);
  return blocks.map(function (block) {
    const lines = block.split("\n");
    const isList = lines.every(function (line) { return /^\s*[-*]\s+/.test(line); });
    if (isList) {
      const items = lines.map(function (line) {
        return "<li>" + inline(line.replace(/^\s*[-*]\s+/, "")) + "</li>";
      }).join("");
      return "<ul>" + items + "</ul>";
    }
    return "<p>" + lines.map(inline).join("<br/>") + "</p>";
  }).join("");
}

function inline(raw) {
  let html = esc(raw);
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  return html;
}

/* ------------------------------------------------------------------ 渲染 */

/* CSS 自检：把"浏览器实际渲染出来的背景色"报出来，用来区分
 * ①样式没生效（拿到旧文件/被缓存）②样式生效了但浏览器/扩展把页面反转成深色。
 * 期望值：light → rgb(242, 244, 247)；dark → rgb(27, 30, 35)。 */
const CSS_EXPECT = { light: "rgb(242, 244, 247)", dark: "rgb(27, 30, 35)" };

function cssDiag(mode) {
  try {
    const bg = getComputedStyle(document.body).backgroundColor;
    return { bg: bg, ok: bg === CSS_EXPECT[mode] };
  } catch (err) {
    return { bg: "?", ok: false };
  }
}

function renderBuildStamp() {
  const node = $("foot-build");
  if (!node || !S.config) return;
  const mode = currentTheme();
  const diag = cssDiag(mode);
  const text = "v" + S.config.version + " · " + mode + " · " + diag.bg;
  node.textContent = text;
  node.className = diag.ok ? "ok" : "bad";
  node.title = diag.ok
    ? "样式已生效（背景色与主题一致）"
    : "样式没生效：期望 " + CSS_EXPECT[mode] + "，实际 " + diag.bg +
      "。若浏览器里页面看着是暗的，说明是浏览器/扩展在做深色反转（用无痕窗口打开对比即可）。";
}

function renderFoot() {
  const cfg = S.config;
  if (!cfg) return;
  $("brand-version").textContent = "v" + cfg.version;
  $("foot-model").textContent = cfg.model;
  // Key 的入口：点这一行打开本机对话框（Key 只发到 127.0.0.1，不写页面/日志）。
  $("foot-key").textContent = cfg.has_key ? "configured" : "missing";
  $("foot-key").className = cfg.has_key ? "ok" : "bad";
  $("foot-key").title = cfg.has_key
    ? "已读到 API Key（来源：" + ((cfg.key && cfg.key.source) || "unknown") + "，掩码 " +
      ((cfg.key && cfg.key.masked) || "***") + "）。点击可更换 / 清除。"
    : "没读到 API Key。点这里填写；也可以终端执行 yaro key set。";
  $("foot-key").style.cursor = "pointer";
  $("foot-docker").textContent = cfg.docker ? "docker" : "none";
  $("foot-docker").className = cfg.docker ? "ok" : "bad";
  // 出网策略常驻可见：用户有权一眼看到现在会不会把数据发出去。
  const egress = (cfg.egress && cfg.egress.mode) || "off";
  $("foot-egress").textContent = egress;
  $("foot-egress").className = egress === "off" ? "ok" : "warn";
  $("foot-egress").title = (cfg.egress && cfg.egress.summary) || "";
  const hintNode = $("composer-hint");
  if (hintNode) {
    hintNode.textContent = egress === "off"
      ? "egress=off：真实模型调用会被拒绝；勾选「演示模式」可零 token 跑通全流程"
      : "Ctrl+Enter 运行 · Esc 停止";
  }
  $("foot-artifacts").textContent = String(cfg.artifact_count);
  $("foot-workspace").textContent = cfg.workspace;
  $("foot-workspace").title = cfg.workspace;
  renderBuildStamp();
  // egress=off 时真实调用必然被拒，所以默认就替用户勾上演示模式（第一次体验必须能跑通）。
  $("opt-demo").checked = !cfg.has_key || egress === "off";
}

function renderSessions() {
  const box = $("session-list");
  if (!S.sessions.length) {
    box.innerHTML = '<div class="empty">还没有会话</div>';
    return;
  }
  box.innerHTML = S.sessions.map(function (item) {
    const active = item.session_id === S.current ? " active" : "";
    const live = item.running ? '<span class="live">running</span>' : "";
    const bad = item.verified ? "" : '<span class="bad">tampered</span>';
    return (
      '<button class="session-item' + active + '" data-session="' + attr(item.session_id) + '">' +
        '<div class="s-title">' + esc(item.title || "未命名会话") + "</div>" +
        '<div class="s-meta"><span>' + item.events + " ev</span>" + live + bad + "</div>" +
      "</button>"
    );
  }).join("");
}

function renderHeader() {
  const head = $("head-title");
  const dot = $("head-dot");
  if (!S.current) {
    head.textContent = "未选择会话";
    dot.className = "dot";
    $("head-verify").hidden = true;
    $("head-demo").hidden = true;
    $("btn-run").disabled = true;
    $("btn-verify").disabled = true;
    return;
  }
  const session = S.sessions.find(function (item) { return item.session_id === S.current; });
  head.textContent = session ? (session.title || "未命名会话") : S.current.slice(0, 12);
  dot.className = "dot" + (S.running ? " live" : "");
  const verify = $("head-verify");
  verify.hidden = false;
  // verified === null 表示这一轮没做全链校验（轮询里只做增量，见 server.session_state）
  if (S.verified === null || S.verified === undefined) {
    verify.textContent = "unverified";
    verify.className = "badge";
  } else {
    verify.textContent = S.verified ? "verified" : "tampered";
    verify.className = "badge" + (S.verified ? "" : " bad");
  }
  $("head-demo").hidden = !(S.stats && S.stats.stop_reason === "demo" ) && !S.isDemoRun;
  $("btn-run").disabled = S.running;
  $("btn-stop").disabled = !S.running;
  $("btn-verify").disabled = false;

  const stats = S.stats || { cost_usd: 0, hit_rate: 0, turns: 0 };
  $("stat-cost").textContent = money(stats.cost_usd);
  $("stat-cache").textContent = stats.hit_rate ? Math.round(stats.hit_rate * 100) + "%" : "—";
  $("stat-turns").textContent = String(stats.turns || 0);
}

/* 空态用字符串而不是复用 DOM 节点：transcript 是整块 innerHTML 重绘的，
   复用节点会在第一次重绘后失效（选中一个空会话就没有空态了）。 */
const WELCOME_HTML = [
  '<section class="welcome">',
  "<h1>YARO — You Only Run Once</h1>",
  "<p>面向机器视觉全链路的开发 harness：数据集体检、标注抽查、切分、训练计划、调优、部署验收，全部可回滚可追溯。</p>",
  "<ul>",
  "<li>在下方输入任务，或勾选 <code>演示模式</code> 看一条不花 token 的全链路运行（<code>demo/dataset</code>）。</li>",
  "<li>每次工具调用的证据（图像 / 表格 / 指标）都会落在产物库里，<code>artifact://</code> 地址可追溯。</li>",
  "<li>右栏 <code>时间线</code> 支持 rewind / fork，回滚只移动指针，不删除历史。</li>",
  "<li>模型自报的分数一律不被采信：指标由 harness 的评估函数计算。</li>",
  "<li><b>默认不出网</b>（<code>egress=off</code>）：原始图与密钥不出本机；真实模型调用需显式授权。</li>",
  "</ul>",
  '<p class="hint">Ctrl+Enter 运行 · Alt+N 新会话 · Esc 停止</p>',
  "</section>",
].join("");

function renderTranscript(forceScroll) {
  const box = $("transcript");
  const nearBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 120;
  if (!S.current || S.events.length === 0) {
    box.innerHTML = WELCOME_HTML;
    return;
  }

  const inLineage = {};
  (S.lineage || []).forEach(function (id) { inLineage[id] = true; });
  const visible = S.events.filter(function (event) { return inLineage[event.event_id]; });
  const parts = [];
  visible.forEach(function (event) { parts.push(renderEvent(event)); });
  box.innerHTML = parts.join("");
  if (forceScroll || nearBottom) box.scrollTop = box.scrollHeight;
}

function renderEvent(event) {
  const p = event.payload || {};
  if (event.type === "user") {
    return (
      '<div class="entry user">' +
        '<div class="msg-head"><span class="who user">YOU</span>' +
          '<span class="time">' + clock(event.ts) + "</span></div>" +
        '<div class="prose">' + mdLite(p.content || "") + "</div>" +
      "</div>"
    );
  }
  if (event.type === "assistant") {
    const calls = (p.tool_calls || []).map(function (call) {
      return '<span class="tag">' + esc(call.function ? call.function.name : "tool") + "</span>";
    }).join(" ");
    const thinking = p.reasoning_content
      ? '<details class="thinking"><summary>thinking</summary><div class="think-body">' +
          esc(p.reasoning_content) + "</div></details>"
      : "";
    const content = p.content
      ? '<div class="prose">' + mdLite(p.content) + "</div>"
      : '<div class="prose"><span class="muted">（无自然语言输出，直接调用工具）</span></div>';
    return (
      '<div class="entry">' +
        '<div class="msg-head"><span class="who assistant">YARO</span>' + calls +
          '<span class="time">' + clock(event.ts) + "</span></div>" +
        content + thinking +
      "</div>"
    );
  }
  if (event.type === "tool") return renderTool(event);
  if (event.type === "run_end") return renderRunEnd(event);
  if (event.type === "compaction") {
    return (
      '<div class="entry"><div class="msg-head"><span class="who">COMPACT</span>' +
        '<span class="time">' + clock(event.ts) + "</span></div>" +
        '<div class="prose muted">上下文已折叠为摘要（' +
          esc(String((p.summary || "").length)) + " 字），逐字历史仍可用 rewind 回看。</div></div>"
    );
  }
  return "";
}

function renderTool(event) {
  const p = event.payload || {};
  const ok = p.ok !== false;
  const metrics = p.metrics || {};
  const chips = Object.keys(metrics).slice(0, 4).map(function (key) {
    return '<span class="chip">' + esc(key) + " " + num(metrics[key]) + "</span>";
  }).join("");
  const images = (p.evidence || []).filter(function (item) {
    return item.kind === "image" && item.payload && item.payload.sha256;
  });
  const tables = (p.evidence || []).filter(function (item) {
    return item.kind === "table" && Array.isArray(item.payload) && item.payload.length;
  });
  const texts = (p.evidence || []).filter(function (item) {
    return item.kind === "text" || item.kind === "scalar" || item.kind === "json";
  });
  const artifacts = p.artifacts || [];
  const degraded = (p.degraded || []).length
    ? '<div class="degraded">degraded: ' + esc((p.degraded || []).join(", ")) + "</div>"
    : "";
  const hint = p.hint ? '<div class="hintline">' + esc(p.hint) + "</div>" : "";

  let body = "";
  if (images.length) {
    body += '<div class="ev-label">evidence · image</div><div class="ev-images">' +
      images.map(function (item) {
        const ref = item.payload;
        return '<figure class="thumb" data-sha="' + esc(ref.sha256) + '">' +
          '<img src="' + artifactUrl(ref.sha256) + '" alt="' + esc(item.caption || "") + '" />' +
          "<figcaption>" + esc(item.caption || ref.uri || "") + "</figcaption></figure>";
      }).join("") + "</div>";
  }
  tables.forEach(function (item) {
    const rows = item.payload;
    const cols = Object.keys(rows[0] || {});
    body += '<div class="ev-label">evidence · table · ' + esc(item.caption || "") + "</div>" +
      '<table class="kv"><thead><tr>' +
        cols.map(function (col) { return "<th>" + esc(col) + "</th>"; }).join("") +
      "</tr></thead><tbody>" +
        rows.slice(0, 30).map(function (row) {
          return "<tr>" + cols.map(function (col) {
            return "<td>" + esc(row[col]) + "</td>";
          }).join("") + "</tr>";
        }).join("") +
      "</tbody></table>";
  });
  texts.forEach(function (item) {
    const value = typeof item.payload === "string" ? item.payload : JSON.stringify(item.payload);
    body += '<div class="ev-label">evidence · ' + esc(item.kind) + " · " + esc(item.caption || "") + "</div>" +
      '<div class="ev-text">' + esc(value) + "</div>";
  });
  if (artifacts.length) {
    body += '<div class="arow">' + artifacts.map(function (ref) {
      return '<span class="achip" data-sha="' + esc(ref.sha256) + '">' +
        esc((ref.media_type || "").split("/").pop() || "bin") + " · " + esc(ref.sha256.slice(0, 10)) + "</span>";
    }).join("") + "</div>";
  }
  if (!body) {
    body = '<div class="ev-text">' + esc(p.content || "(无内容)") + "</div>";
  }
  body += degraded + hint;

  return (
    '<div class="entry"><div class="tool' + (ok ? "" : " bad") + '">' +
      "<details><summary>" +
        '<span class="state">' + (ok ? "✓" : "✗") + "</span>" +
        '<span class="tname">' + esc(p.name || "tool") + "</span>" +
        '<span class="tsum">' + esc(p.summary || p.content || "") + "</span>" +
        '<span class="tmetrics">' + chips + "</span>" +
      "</summary>" +
      '<div class="tool-body">' + body + "</div>" +
      "</details>" +
    "</div></div>"
  );
}

function renderRunEnd(event) {
  const p = event.payload || {};
  const reason = p.stop_reason || "done";
  const label = {
    done: "完成", stopped: "已停止", stuck: "卡死（重复调用）",
    max_turns: "达到轮数上限", max_iterations: "达到轮数上限",
    max_cost: "达到成本上限", max_wall: "达到时长上限",
    tool_errors: "连续工具失败", error: "运行出错",
  }[reason] || reason;
  const usage = p.usage || {};
  const hit = usage.hit || 0;
  const miss = usage.miss || 0;
  const rate = hit + miss ? Math.round((hit / (hit + miss)) * 100) : 0;
  const cls = reason === "done" ? "done" : (reason === "stopped" ? "stopped" : "");
  return (
    '<div class="endbar"><div class="endbar-inner">' +
      '<span class="sr ' + cls + '">■ ' + esc(label) + "</span>" +
      (p.turn ? "<span>turn " + esc(p.turn) + "</span>" : "") +
      (usage.out ? "<span>out " + esc(usage.out) + " tok</span>" : "") +
      "<span>cache " + rate + "%</span>" +
      '<div class="hitbar" title="输入缓存命中率"><i style="width:' + rate + '%"></i></div>' +
      '<span class="spacer"></span>' +
      (p.cost_usd != null ? "<span>" + money(p.cost_usd) + "</span>" : "") +
      '<span class="time">' + clock(event.ts) + "</span>" +
    "</div></div>"
  );
}

function renderTimeline() {
  const box = $("timeline");
  const ids = S.lineage || [];
  const byId = {};
  S.events.forEach(function (event) { byId[event.event_id] = event; });
  const lineage = ids.map(function (id) { return byId[id]; }).filter(Boolean);
  $("timeline-count").textContent = String(lineage.length);
  if (!lineage.length) {
    box.innerHTML = '<div class="empty">—</div>';
    return;
  }
  box.innerHTML = lineage.map(function (event) {
    const p = event.payload || {};
    let desc = "";
    if (event.type === "user") desc = String(p.content || "").slice(0, 42);
    else if (event.type === "assistant") desc = (p.tool_calls || []).length
      ? "→ " + (p.tool_calls || []).map(function (c) { return c.function.name; }).join(", ")
      : String(p.content || "").slice(0, 42);
    else if (event.type === "tool") desc = (p.name || "") + " · " + (p.summary || "");
    else if (event.type === "run_end") desc = String(p.stop_reason || "");
    else if (event.type === "compaction") desc = "compaction";
    const isHead = event.event_id === S.head ? " head" : "";
    const acts = event.type === "compaction"
      ? ""
      : '<span class="tl-acts">' +
          '<button data-rewind="' + attr(event.event_id) + '" title="把分支指针移到这里">rewind</button>' +
          '<button data-fork="' + attr(event.event_id) + '" title="从这里分出新分支">fork</button>' +
        "</span>";
    return '<div class="tl-item' + isHead + '">' +
      '<span class="seq">' + String(event.seq).padStart(3, "0") + "</span>" +
      '<span class="ttype ' + event.type + '">' + esc(event.type) + "</span>" +
      '<span class="tdesc">' + esc(desc) + "</span>" + acts + "</div>";
  }).join("");
}

async function renderArtifacts() {
  const box = $("artifact-grid");
  try {
    const data = await api("/api/artifacts");
    const rows = data.artifacts || [];
    $("artifact-count").textContent = String(rows.length);
    if (!rows.length) {
      box.innerHTML = '<div class="empty">还没有产物。工具产生的图像与文件会出现在这里。</div>';
      return;
    }
    box.innerHTML = rows.map(function (row) {
      const isImage = (row.media_type || "").indexOf("image/") === 0;
      const media = isImage
        ? '<img src="' + artifactUrl(row.sha256) + '" alt="" />'
        : '<div class="aplaceholder">' + esc(row.ext || "bin") + "</div>";
      return '<div class="artifact" data-sha="' + attr(row.sha256) + '">' + media +
        '<div class="ameta"><b>' + esc((row.media_type || "").split("/").pop()) + "</b> " +
        fileSize(row.size) + " · " + esc(row.sha256.slice(0, 8)) + "</div></div>";
    }).join("");
  } catch (err) {
    box.innerHTML = '<div class="empty">' + esc(err.message) + "</div>";
  }
}

async function loadGates() {
  const box = $("gate-result");
  const path = $("gate-path").value || ".";
  box.innerHTML = '<div class="empty">扫描中…</div>';
  try {
    const data = await api("/api/gates?path=" + encodeURIComponent(path));
    $("tab-gates").querySelector(".pane-head span").textContent = "静态闸门 · " + (data.ok ? "pass" : "block");
    const findings = data.findings || [];
    if (!findings.length) {
      box.innerHTML = '<div class="empty">没有发现问题。</div>';
      return;
    }
    box.innerHTML = findings.map(function (item) {
      return '<div class="finding ' + esc(item.severity) + '">' +
        '<div class="frule">' + esc(item.severity) + " · " + esc(item.rule) + "</div>" +
        '<div class="fmsg">' + esc(item.message) + "</div>" +
        (item.fix ? '<div class="ffix">fix: ' + esc(item.fix) + "</div>" : "") +
        (item.file ? '<div class="floc">' + esc(item.file) + (item.line ? ":" + item.line : "") + "</div>" : "") +
        "</div>";
    }).join("");
  } catch (err) {
    box.innerHTML = '<div class="empty">' + esc(err.message) + "</div>";
  }
}

async function loadDoctor() {
  const box = $("doctor");
  box.innerHTML = '<div class="empty">检查中…</div>';
  try {
    const data = await api("/api/doctor");
    const checks = data.checks || [];
    box.innerHTML = checks.map(function (item) {
      return '<div class="dcheck' + (item.ok ? "" : " bad") + '">' +
        '<span class="dstate">' + (item.ok ? "✓" : "✗") + "</span>" +
        '<span class="dname">' + esc(item.check) + "</span>" +
        '<span class="ddetail">' + esc(item.detail || "") + "</span></div>";
    }).join("") +
      '<div class="empty" style="margin-top:10px">workspace: ' + esc(data.workspace || "") + "</div>";
  } catch (err) {
    box.innerHTML = '<div class="empty">' + esc(err.message) + "</div>";
  }
}

/* ------------------------------------------------------------------ 数据 */

async function loadConfig() {
  S.config = await api("/api/config");
  renderFoot();
}

async function loadSessions() {
  const data = await api("/api/sessions");
  S.sessions = data.sessions || [];
  renderSessions();
  renderHeader();
}

async function selectSession(sessionId, options) {
  S.current = sessionId;
  S.events = [];
  S.seq = 0;
  S.lineage = [];
  S.head = "";
  S.stats = null;
  try { localStorage.setItem("yaro.session", sessionId); } catch (err) { /* 隐私模式忽略 */ }
  await loadSessions();
  const data = await api("/api/sessions/" + sessionId + "?since=0");
  ingest(data, true);
  renderTranscript(true);
  restartPolling();
  if (!(options && options.quiet)) $("task").focus();
}

function ingest(data, replace) {
  if (replace) S.events = [];
  const incoming = data.events || [];
  const known = {};
  S.events.forEach(function (event) { known[event.event_id] = true; });
  incoming.forEach(function (event) { if (!known[event.event_id]) S.events.push(event); });
  S.events.sort(function (a, b) { return a.seq - b.seq; });
  const last = S.events.length ? S.events[S.events.length - 1].seq : 0;
  S.seq = Math.max(S.seq, data.seq || 0, last);
  S.lineage = data.lineage || S.lineage;
  S.head = data.head || S.head;
  // verified 为 null = 本次轮询没做全链校验（服务端只给首屏/显式校验做），保留上次结论
  if (data.verified === false || data.verified === true) S.verified = data.verified;
  S.running = !!(data.running && data.running.active);
  S.isDemoRun = !!(data.running && data.running.demo);
  S.stats = data.stats || S.stats;
}

async function poll() {
  if (!S.current) return;
  try {
    const data = await api("/api/sessions/" + S.current + "?since=" + S.seq);
    const before = S.events.length;
    ingest(data, false);
    if (S.events.length !== before) {
      renderTranscript(false);
      renderTimeline();
      if (S.running) { renderArtifacts(); }
    }
    renderHeader();
    if (!S.running && S.pollTimer) { restartPolling(); }
    if (S.stats && S.stats.stop_reason && !S._completionShown) {
      S._completionShown = true;
      loadSessions();
      renderArtifacts();
    }
  } catch (err) {
    /* 单次轮询失败不打断界面；下一轮会补上 */
  }
}

function restartPolling() {
  if (S.pollTimer) { clearTimeout(S.pollTimer); S.pollTimer = null; }
  const interval = S.running ? 800 : 2400;
  S.pollTimer = setTimeout(async function () {
    await poll();
    restartPolling();
  }, interval);
}

/* ------------------------------------------------------------------ 操作 */

async function newSession() {
  try {
    const data = await api("/api/sessions", { method: "POST", body: {} });
    await loadSessions();
    await selectSession(data.session_id);
    $("task").focus();
  } catch (err) { toast(err.message, true); }
}

async function runTask() {
  if (!S.current) {
    await newSession();
    if (!S.current) return;
  }
  const task = $("task").value.trim();
  const demo = $("opt-demo").checked;
  if (!task && !demo) { toast("先写一句任务描述", true); $("task").focus(); return; }
  const effort = $("opt-effort").value || null;
  try {
    await api("/api/sessions/" + S.current + "/run", { method: "POST", body: { task: task, demo: demo, effort: effort } });
    S._completionShown = false;
    $("task").value = "";
    autoGrow();
    S.running = true;
    renderHeader();
    restartPolling();
  } catch (err) { toast(err.message, true); }
}

async function stopRun() {
  if (!S.current) return;
  try {
    await api("/api/sessions/" + S.current + "/stop", { method: "POST", body: {} });
    toast("已请求停止：在当前一轮结束后生效");
  } catch (err) { toast(err.message, true); }
}

async function rewindTo(eventId) {
  if (!S.current) return;
  if (!window.confirm("把分支指针移回这个事件？\n\n事件不会被删除，只是 head 移回去；之后可以再 rewind 回来。")) return;
  try {
    await api("/api/sessions/" + S.current + "/rewind", { method: "POST", body: { event_id: eventId } });
    S._completionShown = false;
    await selectSession(S.current, { quiet: true });
    toast("已回滚。历史事件仍在，可用 fork 从旧点继续。");
  } catch (err) { toast(err.message, true); }
}

async function forkAt(eventId) {
  if (!S.current) return;
  const name = window.prompt("新分支名（字母数字下划线）", "alt-" + Date.now().toString(36).slice(-4));
  if (!name) return;
  try {
    await api("/api/sessions/" + S.current + "/fork", { method: "POST", body: { event_id: eventId, branch: name } });
    await loadSessions();
    toast("已创建分支 " + name + "（P0 的 GUI 只展示 main 分支，可用 CLI/Python API 切换）");
  } catch (err) { toast(err.message, true); }
}

async function verifyChain() {
  if (!S.current) return;
  try {
    const report = await api("/api/sessions/" + S.current + "/verify", { method: "POST", body: {} });
    if (report.ok) toast("事件链校验通过：" + report.events + " 条事件，digest 全部匹配");
    else toast("校验失败：" + (report.problems || []).slice(0, 2).join("；"), true);
    S.verified = report.ok;
    renderHeader();
  } catch (err) { toast(err.message, true); }
}

function openLightbox(sha) {
  $("lightbox-img").src = artifactUrl(sha);
  $("lightbox").hidden = false;
}

/* ------------------------------------------------------------------ 绑定 */

function autoGrow() {
  const node = $("task");
  node.style.height = "auto";
  node.style.height = Math.min(node.scrollHeight, 190) + "px";
}

function openKeyDialog() {
  const cfg = S.config || {};
  const key = cfg.key || {};
  $("key-status").textContent = cfg.has_key
    ? "已配置（" + (key.source || "unknown") + "，掩码 " + (key.masked || "***") + "）"
    : "未配置";
  $("key-egress").textContent = (cfg.model || "—") + " · egress=" +
    ((cfg.egress && cfg.egress.mode) || "off") + " · " +
    ((cfg.egress && cfg.egress.summary) || "");
  const eg = cfg.egress || {};
  $("egress-mode").value = eg.mode || "off";
  $("egress-public").checked = !!eg.allow_public;
  $("egress-summary").textContent = eg.summary || "—";
  $("egress-warn").textContent = "";
  $("key-input").value = "";
  $("key-input2").value = "";
  $("key-warn").textContent = "";
  $("key-dialog").hidden = false;
  setTimeout(function () { $("key-input").focus(); }, 0);
}

function closeKeyDialog() {
  $("key-dialog").hidden = true;
  // 不要留在 DOM 里：关掉就清空，避免有人从开发者工具里读到刚输入的 Key。
  $("key-input").value = "";
  $("key-input2").value = "";
  $("key-warn").textContent = "";
}

async function saveKey() {
  const first = $("key-input").value.trim();
  const second = $("key-input2").value.trim();
  const warn = $("key-warn");
  if (!first) { warn.textContent = "先粘贴 API Key"; return; }
  if (first !== second) { warn.textContent = "两次输入不一致"; return; }
  if (/\s/.test(first)) { warn.textContent = "Key 里不能有空格或换行"; return; }
  warn.textContent = "保存中…";
  try {
    await api("/api/key", { method: "POST", body: { action: "set", key: first } });
    closeKeyDialog();
    toast("API Key 已保存到本机");
    await loadConfig();
    renderFoot();
  } catch (error) {
    // 错误信息由服务端给（不含 Key 原文），这里只显示原因。
    warn.textContent = error.message || "保存失败";
  }
}

async function clearKey() {
  const warn = $("key-warn");
  if (!window.confirm("清除本机保存的 API Key？")) return;
  warn.textContent = "清除中…";
  try {
    await api("/api/key", { method: "POST", body: { action: "clear" } });
    closeKeyDialog();
    toast("已清除本机保存的 API Key");
    await loadConfig();
    renderFoot();
  } catch (error) {
    warn.textContent = error.message || "清除失败";
  }
}

async function saveEgress() {
  const warn = $("egress-warn");
  const mode = $("egress-mode").value;
  const allowPublic = $("egress-public").checked;
  if (allowPublic && mode === "full") {
    if (!window.confirm("full + 公网 = 原始证据图会发给云端端点。确定要这样放宽吗？")) return;
  }
  warn.textContent = "保存中…";
  try {
    await api("/api/egress", { method: "POST", body: { mode: mode, allow_public: allowPublic } });
    await loadConfig();
    const cfg = S.config || {};
    const eg = cfg.egress || {};
    $("egress-summary").textContent = eg.summary || "—";
    $("key-egress").textContent = (cfg.model || "—") + " · egress=" + (eg.mode || "off") +
      " · " + (eg.summary || "");
    renderFoot();
    warn.textContent = "";
    toast("出网策略已保存并生效");
  } catch (error) {
    warn.textContent = error.message || "保存失败";
  }
}

function bind() {
  $("btn-new").addEventListener("click", newSession);
  $("btn-run").addEventListener("click", runTask);
  $("btn-stop").addEventListener("click", stopRun);
  $("btn-verify").addEventListener("click", verifyChain);
  $("btn-theme").addEventListener("click", toggleTheme);
  $("btn-gate").addEventListener("click", loadGates);
  $("btn-doctor").addEventListener("click", loadDoctor);
  // 左栏 api key 一行可点：打开本机 Key 对话框（Key 只发到 127.0.0.1）。
  $("foot-key").addEventListener("click", openKeyDialog);
  $("btn-key-close").addEventListener("click", closeKeyDialog);
  $("btn-key-save").addEventListener("click", saveKey);
  $("btn-key-clear").addEventListener("click", clearKey);
  $("btn-egress-save").addEventListener("click", saveEgress);
  $("key-dialog").addEventListener("click", function (event) {
    if (event.target === $("key-dialog")) closeKeyDialog();
  });
  ["key-input", "key-input2"].forEach(function (id) {
    $(id).addEventListener("keydown", function (event) {
      if (event.key === "Enter") { event.preventDefault(); saveKey(); }
    });
  });
  $("task").addEventListener("input", autoGrow);

  $("task").addEventListener("keydown", function (event) {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      runTask();
    }
  });

  document.addEventListener("keydown", function (event) {
    if (event.altKey && (event.key === "n" || event.key === "N")) {
      event.preventDefault();
      newSession();
    }
    if (event.key === "Escape") {
      if (!$("key-dialog").hidden) { closeKeyDialog(); return; }
      if (!$("lightbox").hidden) { $("lightbox").hidden = true; return; }
      if (S.running) stopRun();
    }
  });

  $("session-list").addEventListener("click", function (event) {
    const button = event.target.closest("[data-session]");
    if (button) selectSession(button.getAttribute("data-session"));
  });

  $("transcript").addEventListener("click", function (event) {
    const node = event.target.closest("[data-sha]");
    if (node) openLightbox(node.getAttribute("data-sha"));
  });

  $("artifact-grid").addEventListener("click", function (event) {
    const node = event.target.closest("[data-sha]");
    if (node) openLightbox(node.getAttribute("data-sha"));
  });

  $("timeline").addEventListener("click", function (event) {
    const rewind = event.target.closest("[data-rewind]");
    if (rewind) { rewindTo(rewind.getAttribute("data-rewind")); return; }
    const fork = event.target.closest("[data-fork]");
    if (fork) forkAt(fork.getAttribute("data-fork"));
  });

  $("tabs").addEventListener("click", function (event) {
    const tab = event.target.closest(".tab");
    if (!tab) return;
    const name = tab.getAttribute("data-tab");
    Array.prototype.forEach.call(document.querySelectorAll(".tab"), function (node) {
      node.classList.toggle("active", node === tab);
    });
    ["timeline", "artifacts", "gates", "doctor"].forEach(function (key) {
      $("tab-" + key).hidden = key !== name;
    });
    if (name === "artifacts") renderArtifacts();
    if (name === "doctor") loadDoctor();
    if (name === "gates" && !$("gate-result").dataset.loaded) loadGates();
  });

  $("lightbox").addEventListener("click", function () { $("lightbox").hidden = true; });
}

/* ------------------------------------------------------------------ 主题 */

const THEME_KEY = "yaro.theme";

function currentTheme() {
  try {
    const saved = localStorage.getItem(THEME_KEY);
    if (saved === "dark" || saved === "light") return saved;
  } catch (err) { /* 隐私模式忽略 */ }
  return "light"; /* 默认浅色高对比：白底近黑字最不费眼 */
}

function applyTheme(mode) {
  document.documentElement.dataset.theme = mode;
  const btn = $("btn-theme");
  if (btn) {
    btn.textContent = mode === "dark" ? "浅色" : "深色";
    btn.title = mode === "dark" ? "切到浅色主题（高对比）" : "切到深色主题";
  }
  renderBuildStamp();
}

function toggleTheme() {
  const next = currentTheme() === "dark" ? "light" : "dark";
  try { localStorage.setItem(THEME_KEY, next); } catch (err) { /* 隐私模式忽略 */ }
  applyTheme(next);
  toast(next === "dark" ? "已切到深色主题" : "已切到浅色主题（高对比）");
}

/* ------------------------------------------------------------------ 启动 */

async function boot() {
  applyTheme(currentTheme());
  bind();
  autoGrow();
  try { await loadConfig(); } catch (err) { toast("无法读取 /api/config：" + err.message, true); return; }
  try { await loadSessions(); } catch (err) { toast(err.message, true); }
  let saved = null;
  try { saved = localStorage.getItem("yaro.session"); } catch (err) { saved = null; }
  if (saved && S.sessions.some(function (item) { return item.session_id === saved; })) {
    await selectSession(saved, { quiet: true });
  } else {
    renderHeader();
    restartPolling();
  }
}

document.addEventListener("DOMContentLoaded", boot);
