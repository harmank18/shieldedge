"use strict";
// ShieldEdge dashboard front-end. No external libraries: everything is served by the device.

const $ = (id) => document.getElementById(id);
const events = new Map();          // id -> event (most recent ~600)
const MAX_ROWS = 160;
let filter = "smart";
let selected = null;
let explainES = null;
let llmBench = null;

// ------------------------------------------------------------------ helpers
const KIND_LABEL = { URL: "LINK", SMS: "SMS", NETWORK: "NET" };
const pct = (x, d = 1) => (x * 100).toFixed(d) + "%";
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}
function passes(ev) {
  if (filter === "all") return true;
  if (filter === "threats") return ev.verdict !== "SAFE";
  return ev.kind !== "NETWORK" || ev.verdict !== "SAFE";
}
function truthText(ev) {
  if (ev.label === null || ev.label === undefined) return null;
  const lab = String(ev.label);
  const isThreat = !["benign", "genuine"].includes(lab.toLowerCase());
  const flagged = ev.verdict !== "SAFE";
  return { lab, ok: isThreat === flagged };
}

// ------------------------------------------------------------------ live feed
function row(ev) {
  const r = el("div", "item");
  r.dataset.id = ev.id;
  if (selected === ev.id) r.classList.add("sel");
  r.append(el("span", "t", ev.time), el("span", "k", KIND_LABEL[ev.kind] || ev.kind));
  const x = el("span", "x");
  const src = el("span", "src" + (ev.source === "you" ? " you" : ""), ev.source === "you" ? "you" : "replay");
  x.append(src, document.createTextNode(ev.display));
  x.title = ev.display;
  r.append(x);
  const b = el("span", "badge v-" + ev.verdict, ev.verdict);
  r.append(b);
  r.addEventListener("click", () => select(ev.id));
  return r;
}
function addToFeed(ev) {
  if (!passes(ev)) return;
  const feed = $("feed");
  feed.prepend(row(ev));
  while (feed.children.length > MAX_ROWS) feed.lastChild.remove();
}
function rebuildFeed() {
  const feed = $("feed");
  feed.replaceChildren();
  const list = [...events.values()].filter(passes).slice(-MAX_ROWS).reverse();
  for (const ev of list) feed.append(row(ev));
}
function remember(ev) {
  events.set(ev.id, ev);
  if (events.size > 600) events.delete(events.keys().next().value);
}

function connect() {
  const es = new EventSource("/api/events");
  es.onmessage = (m) => {
    const ev = JSON.parse(m.data);
    if (events.has(ev.id)) return;
    remember(ev);
    addToFeed(ev);
    if (selected === null && ev.threat) select(ev.id);  // show something useful on first load
  };
  es.onerror = () => { /* EventSource reconnects by itself (retry: 2000) */ };
}

document.querySelectorAll(".chips [data-filter]").forEach((c) =>
  c.addEventListener("click", () => {
    document.querySelectorAll(".chips [data-filter]").forEach((x) => x.classList.toggle("on", x === c));
    filter = c.dataset.filter;
    rebuildFeed();
  }));

// ------------------------------------------------------------------ detail + Llama
function select(id, autoExplain = false) {
  const ev = events.get(id);
  if (!ev) return;
  selected = id;
  document.querySelectorAll(".item.sel").forEach((x) => x.classList.remove("sel"));
  const r = document.querySelector(`.item[data-id="${id}"]`);
  if (r) r.classList.add("sel");
  if (explainES) { explainES.close(); explainES = null; }

  $("detailEmpty").hidden = true;
  $("detailBody").hidden = false;
  $("detailMeta").textContent = `#${ev.id} · ${ev.time} · ${ev.source === "you" ? "analysed by you" : "replay"}`;
  const v = $("dVerdict");
  v.className = "badge v-" + ev.verdict;
  v.textContent = ev.verdict;
  $("dConf").textContent = `${pct(ev.confidence)} confidence`;
  $("dKind").textContent = { URL: "Link", SMS: "SMS message", NETWORK: "Network flow" }[ev.kind] || ev.kind;
  $("dInput").textContent = ev.display;

  const truth = $("dTruth");
  truth.replaceChildren();
  const t = truthText(ev);
  if (t) {
    truth.append(document.createTextNode(`Dataset label: ${t.lab}${ev.dataset ? " (" + ev.dataset + ")" : ""} · `));
    truth.append(el("span", t.ok ? "ok" : "no", t.ok ? "verdict agrees with the label" : "verdict differs from the label"));
  }

  const sig = $("dSignals");
  sig.replaceChildren();
  (ev.signals.length ? ev.signals : ["No warning signs found."]).forEach((s) => sig.append(el("li", "", s)));

  const steps = $("dSteps");
  steps.replaceChildren();
  (ev.steps.length ? ev.steps : ["Nothing to do: this looks safe."]).forEach((s) => steps.append(el("li", "", s)));
  $("dCaveat").textContent = ev.caveat || "";

  const btn = $("explainBtn");
  const txt = $("dLlm");
  txt.classList.remove("streaming");
  $("dLlmMeta").textContent = "";
  if (ev.advisory) {
    btn.disabled = false;
    btn.hidden = false;
    txt.textContent = "Llama 3.2 1B can explain this verdict in plain language, on this device.";
    txt.classList.add("muted");
    if (autoExplain) explain(id);
  } else {
    btn.hidden = true;
    txt.textContent = "No explanation needed: nothing suspicious was found.";
    txt.classList.add("muted");
  }
}

function explain(id) {
  const btn = $("explainBtn");
  const txt = $("dLlm");
  const meta = $("dLlmMeta");
  btn.disabled = true;
  txt.classList.remove("muted");
  txt.textContent = "";
  meta.textContent = "starting llama_main…";
  const t0 = performance.now();
  let first = null;
  explainES = new EventSource(`/api/explain/${id}`);
  explainES.onmessage = (m) => {
    const d = JSON.parse(m.data);
    if (selected !== id) return;
    if (d.type === "start") {
      txt.classList.add("streaming");
      meta.textContent = d.llm ? `${d.model} · running on this device…` : "Llama runner not installed: template explanation";
    } else if (d.type === "token") {
      if (first === null) first = performance.now() - t0;
      txt.textContent += d.text;
    } else if (d.type === "done") {
      txt.classList.remove("streaming");
      txt.textContent = d.explanation;
      const m2 = d.metrics;
      const parts = [d.source];
      if (m2) parts.push(`TTFT ${(m2.ttft_ms / 1000).toFixed(2)} s`, `prefill ${m2.prefill_tok_s.toFixed(0)} tok/s`,
        `decode ${m2.decode_tok_s.toFixed(1)} tok/s`, `${m2.generated_tokens} tokens`);
      meta.textContent = parts.join(" · ");
      if (d.remediation && d.remediation.length) {
        const steps = $("dSteps");
        steps.replaceChildren();
        d.remediation.forEach((s) => steps.append(el("li", "", s)));
      }
      btn.disabled = false;
      explainES.close();
      explainES = null;
    } else if (d.type === "busy") {
      txt.classList.remove("streaming");
      txt.textContent = `Llama is already explaining item #${d.id}. One explanation runs at a time to keep memory low; try again in a few seconds.`;
      meta.textContent = "";
      btn.disabled = false;
      explainES.close();
      explainES = null;
    } else if (d.type === "error") {
      txt.classList.remove("streaming");
      txt.textContent = d.message;
      btn.disabled = false;
      explainES.close();
      explainES = null;
    }
  };
  explainES.onerror = () => {
    if (explainES) { explainES.close(); explainES = null; }
    btn.disabled = false;
    txt.classList.remove("streaming");
  };
}
$("explainBtn").addEventListener("click", () => selected && explain(selected));

// ------------------------------------------------------------------ try it
document.querySelectorAll(".chip.ex").forEach((c) =>
  c.addEventListener("click", () => { $("tryText").value = c.dataset.ex; $("tryText").focus(); }));
$("tryForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  $("tryErr").textContent = "";
  const text = $("tryText").value.trim();
  if (!text) { $("tryErr").textContent = "Type a link or an SMS first."; return; }
  try {
    const res = await fetch("/api/analyze", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }),
    });
    const data = await res.json();
    if (!res.ok) { $("tryErr").textContent = data.error || "Could not analyse that."; return; }
    if (!events.has(data.id)) { remember(data); addToFeed(data); }
    select(data.id, true);
    loadAlerts();
  } catch (err) {
    $("tryErr").textContent = "Dashboard server not reachable.";
  }
});

// ------------------------------------------------------------------ stats, chart
function drawChart(timeline) {
  const c = $("chart");
  const dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth, h = c.clientHeight;
  if (c.width !== w * dpr || c.height !== h * dpr) { c.width = w * dpr; c.height = h * dpr; }
  const g = c.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  const padL = 28, padB = 18, padT = 6;
  const max = Math.max(6, ...timeline.map((t) => t[1]));
  const bw = (w - padL) / timeline.length;
  g.strokeStyle = "#233140"; g.fillStyle = "#8b9bab"; g.font = "11px system-ui"; g.lineWidth = 1;
  for (const f of [0, 0.5, 1]) {
    const y = padT + (h - padB - padT) * (1 - f);
    g.beginPath(); g.moveTo(padL, y); g.lineTo(w, y); g.stroke();
    g.fillText(String(Math.round(max * f)), 2, y + 4);
  }
  timeline.forEach(([, flows, bad], i) => {
    const x = padL + i * bw;
    const H = h - padB - padT;
    const hb = (bad / max) * H, hn = ((flows - bad) / max) * H;
    g.fillStyle = "#2c4a5e";
    g.fillRect(x + 0.5, padT + H - hn - hb, Math.max(1, bw - 1), hn);
    g.fillStyle = "#ef5b5b";
    g.fillRect(x + 0.5, padT + H - hb, Math.max(1, bw - 1), hb);
  });
  g.fillStyle = "#8b9bab";
  g.fillText("−2 min", padL, h - 4);
  g.fillText("−1 min", padL + (w - padL) / 2 - 16, h - 4);
  g.fillText("now", w - 24, h - 4);
}

async function loadStats() {
  try {
    const s = await (await fetch("/api/stats")).json();
    document.querySelectorAll(".kpi[data-kind]").forEach((k) => {
      const c = s.counts[k.dataset.kind] || {};
      const total = Object.values(c).reduce((a, b) => a + b, 0);
      k.querySelector(".k-total").textContent = total.toLocaleString();
      k.querySelector(".k-threat").textContent = (total - (c.SAFE || 0)).toLocaleString();
      const a = s.agreement[k.dataset.kind] || {};
      const n = (a.correct || 0) + (a.wrong || 0);
      k.querySelector(".k-agree").textContent = n ? `replay: ${pct((a.correct || 0) / n)} agree with dataset labels (n=${n})` : "";
    });
    $("phase").textContent = s.replay_on ? (s.phase || "–") : "paused";
    $("replayBtn").textContent = s.replay_on ? "Replay: on" : "Replay: paused";
    $("replayBanner").hidden = false;
    const st = $("llmState");
    st.textContent = s.llm_busy_id ? `explaining item #${s.llm_busy_id}…` : "idle · on demand";
    st.classList.toggle("busy", !!s.llm_busy_id);
    drawChart(s.timeline);
  } catch (e) { /* server restarting */ }
}
$("replayBtn").addEventListener("click", async () => {
  await fetch("/api/replay", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
  loadStats();
});

// ------------------------------------------------------------------ device
async function loadDevice() {
  try {
    const d = await (await fetch("/api/device")).json();
    $("deviceName").textContent = `${d.device} · ${d.arch} · ${d.ram_gb} GB RAM`;
    $("devArch").textContent = d.llama_model ? `Llama program: ${d.llama_model}` : "Llama runner not installed";
    const cores = $("cores");
    if (cores.children.length !== d.cpu.length) {
      cores.replaceChildren();
      d.cpu.forEach((_, i) => {
        const c = el("div", "core");
        c.append(el("div", "", `core ${i}`), el("div", "cbar"), el("b", "", "0%"));
        c.querySelector(".cbar").append(el("i"));
        cores.append(c);
      });
    }
    d.cpu.forEach((v, i) => {
      const c = cores.children[i];
      c.querySelector(".cbar i").style.height = `${v}%`;
      c.querySelector("b").textContent = `${Math.round(v)}%`;
    });
    $("ramBar").style.width = `${(d.ram_used_gb / d.ram_total_gb) * 100}%`;
    $("ramTxt").textContent = `${d.ram_used_gb.toFixed(1)} / ${d.ram_total_gb} GB`;
    $("pillRam").textContent = `RAM ${d.ram_used_gb.toFixed(1)}/${d.ram_total_gb} GB`;
    $("pillTemp").textContent = d.temp_c ? `CPU ${d.temp_c.toFixed(1)} °C` : "CPU temp n/a";
    $("llamaTxt").textContent = d.llama_running ? `running${d.llama_rss_mb ? " · " + d.llama_rss_mb + " MB" : ""}` : "idle";
    $("appRss").textContent = `${d.app_rss_mb} MB`;
    const ext = $("extConn");
    ext.textContent = String(d.external_connections);
    ext.classList.toggle("zero", d.external_connections === 0);
    $("ifaces").textContent = d.interfaces_up.length ? d.interfaces_up.join(", ") : "none (fully offline)";
    const pill = $("pillOffline");
    pill.classList.toggle("good", d.external_connections === 0);
    pill.classList.toggle("badp", d.external_connections > 0);
    pill.querySelector("span").textContent = `External connections: ${d.external_connections}`;
  } catch (e) { console.error("device panel:", e); }
}

// ------------------------------------------------------------------ quality + audit
async function loadQuality() {
  try {
    const q = await (await fetch("/api/quality")).json();
    const tb = document.querySelector("#qTable tbody");
    tb.replaceChildren();
    q.detection.filter((r) => !/baseline|ablation/i.test(r.dataset)).forEach((r) => {
      const tr = el("tr");
      tr.append(el("td", "", r.task.replace(" phishing", "").replace(" scam", "").replace(" anomaly", "")),
        el("td", "", r.dataset), el("td", "num", r.n.toLocaleString()), el("td", "num", pct(r.precision)),
        el("td", "num", pct(r.recall)), el("td", "num", r.f1.toFixed(3)), el("td", "num", pct(r.fpr)));
      tb.append(tr);
    });
    if (q.llm) {
      llmBench = q.llm;
      $("llmBench").textContent = `${q.llm.device.replace(" Model B Rev 1.0", "")}: TTFT ${(q.llm.ttft_ms / 1000).toFixed(1)} s · ` +
        `${q.llm.decode_tok_s.toFixed(1)} tok/s · ${(q.llm.peak_rss_mb / 1024).toFixed(2)} GB`;
    }
  } catch (e) { /* ignore */ }
}
async function loadAlerts() {
  try {
    const a = await (await fetch("/api/alerts")).json();
    const list = $("audit");
    list.replaceChildren();
    if (!a.alerts.length) list.append(el("div", "muted small", "No stored alerts."));
    a.alerts.forEach((r) => {
      const d = el("div", "arow");
      d.append(el("span", "id", `#${r.id}`), el("span", "", `${r.time} UTC · ${r.type}`));
      d.append(el("span", "badge v-" + r.verdict, r.verdict));
      if (r.marked_safe) d.append(el("span", "muted small", "marked safe"));
      else {
        const b = el("button", "btn ghost", "Mark safe");
        b.addEventListener("click", async () => { await fetch(`/api/alerts/${r.id}/safe`, { method: "POST" }); loadAlerts(); });
        d.append(b);
      }
      list.append(d);
    });
    $("auditNote").textContent = `Flagged: ${a.feedback.flagged} · marked safe by you: ${a.feedback.marked_safe_by_user}. ` +
      "Only items you analyse are stored, as SHA-256 hashes. Replay items are never logged.";
  } catch (e) { /* ignore */ }
}
let clearArmed = null;
$("clearBtn").addEventListener("click", async () => {
  const b = $("clearBtn");
  if (!clearArmed) {
    b.textContent = "Click again to delete all";
    clearArmed = setTimeout(() => { clearArmed = null; b.textContent = "Clear all logs"; }, 4000);
    return;
  }
  clearTimeout(clearArmed);
  clearArmed = null;
  await fetch("/api/alerts/clear", { method: "POST" });
  b.textContent = "Clear all logs";
  loadAlerts();
});

// ------------------------------------------------------------------ start
// ?nostream skips the live feed (used for automated screenshots, which wait for the network to go idle)
if (!new URLSearchParams(location.search).has("nostream")) connect();
loadQuality();
loadAlerts();
loadStats();
loadDevice();
setInterval(loadStats, 1000);
setInterval(loadDevice, 2000);
setInterval(loadAlerts, 8000);
window.addEventListener("resize", loadStats);
