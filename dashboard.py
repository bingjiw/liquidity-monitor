"""
生成流动性仪表盘(单个自包含 HTML 文件,数据内嵌,图表用 Plotly)。
由 liquidity_monitor.py 在每次运行结束时调用。
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# 仪表盘里可查看的原始指标: 列名 -> (中文名, 单位, 分组, 预警线或 None)
RAW_SERIES = {
    "reserves":        ("银行准备金", "十亿美元", "A 钱的数量", 2900),
    "tga":             ("财政部 TGA", "十亿美元", "A 钱的数量", None),
    "rrp":             ("隔夜逆回购", "十亿美元", "A 钱的数量", None),
    "net_liq":         ("净流动性(总资产 − TGA − 逆回购)", "十亿美元", "A 钱的数量", None),
    "fed_bills_chg":   ("联储持有短债四周变化", "十亿美元", "A 钱的数量", None),
    "sofr_iorb":       ("SOFR − IORB", "BP", "B 回购管道价格", 10),
    "sofr99_iorb":     ("SOFR99 − IORB", "BP", "B 回购管道价格", 25),
    "tgcr_rrp":        ("TGCR − 逆回购利率", "BP", "B 回购管道价格", None),
    "srf":             ("SRF 等回购操作用量", "十亿美元", "B 回购管道价格", 10),
    "ccc_oas":         ("CCC 及以下 OAS", "%", "C 信用价格", 12.15),
    "ccc_cost_approx": ("CCC 借钱成本(5 年美债 + CCC OAS)", "%", "C 信用价格", None),
    "hy_oas":          ("高收益债 OAS", "%", "C 信用价格", None),
    "ig_oas":          ("投资级 OAS", "%", "C 信用价格", None),
    "real_policy_rate":("实际政策利率(EFFR − 核心 PCE)", "%", "D 利率与政策", None),
    "tips5":           ("5 年 TIPS 实际收益率", "%", "D 利率与政策", None),
    "term_premium":    ("10 年期限溢价", "%", "D 利率与政策", None),
    "core_pce_yoy":    ("核心 PCE 同比", "%", "D 利率与政策", None),
    "vix":             ("VIX", "点", "E 市场压力与美元", None),
    "usdjpy":          ("美元兑日元", "日元", "E 市场压力与美元", None),
    "nfci":            ("芝加哥联储 NFCI(正 = 偏紧)", "", "E 市场压力与美元", 0),
    "stlfsi":          ("圣路易斯联储金融压力指数(正 = 压力)", "", "E 市场压力与美元", 0),
}


def _clean(v):
    if v is None:
        return None
    try:
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
    except TypeError:
        pass
    return v


def _col(df: pd.DataFrame, c: str, nd: int):
    if c not in df:
        return [None] * len(df)
    return [_clean(round(float(x), nd)) if pd.notna(x) else None for x in df[c]]


def write_dashboard(f: pd.DataFrame, sc: pd.DataFrame, alerts: list[str], cal: pd.DataFrame,
                    path: Path, log_path: Path, bt: dict | None = None) -> None:
    from liquidity_monitor import PILLAR_NAMES, PILLAR_WEIGHTS, SCORED

    valid = sc.dropna(subset=["liquidity_index"])
    idx = valid.index
    ff = f.ffill().reindex(idx)

    data = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "dates": [d.strftime("%Y-%m-%d") for d in idx],
        "index": _col(valid, "liquidity_index", 1),
        "pillars_used": _col(valid, "pillars_used", 0),
        "pillars": {p: {"name": PILLAR_NAMES[p], "weight": w, "z": _col(valid, p, 2)}
                    for p, w in PILLAR_WEIGHTS.items()},
        "raw": {k: {"name": n, "unit": u, "group": g, "threshold": t,
                    "v": _col(ff, k, 3)}
                for k, (n, u, g, t) in RAW_SERIES.items() if k in ff},
        "contrib": [{"pillar": PILLAR_NAMES[p], "name": name,
                     "z": _clean(round(float(valid[f"z_{c}"].ffill().iloc[-1]), 2))
                     if f"z_{c}" in valid and pd.notna(valid[f"z_{c}"].ffill().iloc[-1]) else None}
                    for c, p, _, name in SCORED],
        "alerts": alerts,
        "calendar": [{"date": r.issue_date.strftime("%Y-%m-%d"), "gross_bn": _clean(round(float(r.gross_bn), 1)) if pd.notna(r.gross_bn) else None,
                      "detail": r.detail} for r in cal.itertuples()] if not cal.empty else [],
        "bt": bt,
        "log": json.loads(pd.read_csv(log_path).to_json(orient="records")) if log_path.exists() else [],
    }
    html = TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>美元流动性仪表盘</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.2/plotly.min.js"></script>
<style>
:root{
  --bg:#f6f7f9; --card:#ffffff; --ink:#1b1f24; --muted:#5f6b7a; --line:#e3e7ec; --grid:#eef1f4;
  --accent:#2f6fde; --tight:#d64545; --semitight:#e8913a; --neutral:#8a94a3; --semiloose:#57b26a; --loose:#2e8b57;
  --band-tight:rgba(214,69,69,.10); --band-semitight:rgba(232,145,58,.10); --band-neutral:rgba(138,148,163,.08);
  --band-semiloose:rgba(87,178,106,.10); --band-loose:rgba(46,139,87,.13);
  --p1:#2f6fde; --p2:#e8913a; --p3:#d64545; --p4:#7b5cd6; --p5:#2aa198;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#111418; --card:#1a1f25; --ink:#e7ebf0; --muted:#98a3b1; --line:#2a313a; --grid:#232a32;
    --accent:#6b9cff; --tight:#ff6b6b; --semitight:#f0a35a; --neutral:#8d97a6; --semiloose:#6cc87e; --loose:#43b27a;
    --band-tight:rgba(255,107,107,.12); --band-semitight:rgba(240,163,90,.12); --band-neutral:rgba(141,151,166,.08);
    --band-semiloose:rgba(108,200,126,.12); --band-loose:rgba(67,178,122,.15);
    --p1:#6b9cff; --p2:#f0a35a; --p3:#ff6b6b; --p4:#a58bff; --p5:#3cc2b5;
  }
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif}
.wrap{max-width:1120px;margin:0 auto;padding:20px 16px 48px}
h1{font-size:22px;margin:0 0 4px}
h2{font-size:17px;margin:0 0 12px}
.sub{color:var(--muted);font-size:13px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin-top:16px}
.hero{display:flex;flex-wrap:wrap;gap:20px;align-items:center}
.big{font-size:56px;font-weight:700;line-height:1;font-variant-numeric:tabular-nums}
.badge{display:inline-block;padding:3px 10px;border-radius:999px;color:#fff;font-weight:600;font-size:14px}
.deltas{display:grid;grid-template-columns:repeat(4,minmax(72px,1fr));gap:8px;flex:1;min-width:280px}
.delta{background:var(--bg);border-radius:8px;padding:8px 10px}
.delta b{display:block;font-size:18px;font-variant-numeric:tabular-nums}
.delta span{font-size:12px;color:var(--muted)}
.alerts{margin:12px 0 0;padding:0;list-style:none}
.alerts li{padding:8px 10px;border-left:3px solid var(--tight);background:var(--band-tight);border-radius:4px;margin-top:6px;font-size:14px}
.ok{color:var(--muted);font-size:14px;margin-top:10px}
.chart{width:100%;height:360px}
.row{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-bottom:10px}
select,input[type=date],button{font:inherit;color:var(--ink);background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:6px 10px}
button{cursor:pointer}
button.on{background:var(--accent);color:#fff;border-color:var(--accent)}
table{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}
th,td{padding:7px 8px;border-bottom:1px solid var(--line);text-align:right}
th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-weight:600}
.pos{color:var(--loose)} .neg{color:var(--tight)}
.bar{display:inline-block;height:8px;border-radius:4px;vertical-align:middle}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media (max-width:760px){.grid2{grid-template-columns:1fr}.big{font-size:44px}.chart{height:300px}}
.note{color:var(--muted);font-size:13px;margin-top:8px}
.scroll{overflow-x:auto}
</style>
</head>
<body>
<div class="wrap">
  <h1>美元流动性仪表盘</h1>
  <div class="sub" id="asof"></div>

  <div class="card">
    <div class="hero">
      <div>
        <div class="sub">综合流动性指数(0–100,50 = 三年中性)</div>
        <div class="big" id="now"></div>
        <div style="margin-top:8px"><span class="badge" id="regime"></span></div>
      </div>
      <div class="deltas" id="deltas"></div>
    </div>
    <ul class="alerts" id="alerts"></ul>
  </div>

  <div class="card" id="btCard">
    <h2>历史校准信号(净流动性规则)</h2>
    <div class="hero">
      <div>
        <div class="sub">建议股票仓位</div>
        <div class="big" id="btPos"></div>
      </div>
      <div style="flex:1;min-width:260px" id="btNow"></div>
    </div>
    <div class="row" style="margin-top:12px" id="btRangeBtns"></div>
    <div id="cBt" class="chart"></div>
    <div id="cNl" class="chart" style="height:220px"></div>
    <div class="note">上图:SPY(对数坐标),红色底色 = 规则建议减到半仓的时段。下图:净流动性(联储总资产 − TGA − 逆回购)13 周变化,十亿美元。规则:13 周变化连续 20 个交易日为负 → 半仓;连续 20 个交易日为正 → 满仓。数据按实际公布时间滞后,次日执行。</div>
    <h2 style="margin-top:18px">回测成绩(SPY,2019-03 至今)</h2>
    <div class="scroll"><table id="btTable"></table></div>
    <div class="note">“同仓位对照”= 一直持有和该规则平均仓位相同比例的 SPY。样本外 = 2023 年以后,规则参数没有用这段数据调过。综合指数本身的水平和趋势规则没有跑赢同仓位对照,所以综合指数只作环境参考,不直接用于加减仓。完整报告见仓库 output/backtest_summary.md。历史表现不代表未来。</div>
  </div>

  <div class="card">
    <h2>综合指数走势</h2>
    <div class="row" id="rangeBtns"></div>
    <div id="cIndex" class="chart"></div>
    <div class="note">背景色:红 = 紧张(&lt;30),橙 = 偏紧(30–45),灰 = 中性(45–55),浅绿 = 偏松(55–70),绿 = 宽松(≥70)。</div>
  </div>

  <div class="card">
    <h2>五大支柱(z 分数,正 = 宽松)</h2>
    <div id="cPillars" class="chart"></div>
    <div class="note">点击图例可隐藏或显示某条线。</div>
  </div>

  <div class="card">
    <h2>单项指标</h2>
    <div class="row"><select id="rawSel"></select><span class="sub" id="rawNote"></span></div>
    <div id="cRaw" class="chart"></div>
  </div>

  <div class="card">
    <h2>任意两天对比</h2>
    <div class="row">
      <label>日期 A <input type="date" id="dA"></label>
      <label>日期 B <input type="date" id="dB"></label>
      <button data-back="5">B = A 前一周</button>
      <button data-back="21">前一月</button>
      <button data-back="252">前一年</button>
    </div>
    <div class="scroll"><table id="cmp"></table></div>
  </div>

  <div class="grid2">
    <div class="card">
      <h2>今日各序列贡献</h2>
      <div class="scroll"><table id="contrib"></table></div>
    </div>
    <div class="card">
      <h2>每日运行记录</h2>
      <div class="scroll"><table id="log"></table></div>
      <h2 style="margin-top:18px">未来 30 天国债结算</h2>
      <div class="scroll"><table id="cal"></table></div>
    </div>
  </div>

  <div class="note" style="margin-top:16px">数据来源:FRED(圣路易斯联储)、纽约联储、美国财政部。ICE 信用利差在 FRED 上只有约 3 年历史,所以较早年份的综合指数没有信用支柱,用其余支柱按权重归一计算。仅用于研究,不构成投资建议。</div>
</div>

<script>
const D = __DATA__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const N = D.dates.length;
const last = i => { for (let k = i; k >= 0; k--) if (D.index[k] != null) return k; return -1; };
const li = last(N - 1);
const fmt = (v, nd = 1) => v == null ? "—" : Number(v).toLocaleString("zh-CN", {minimumFractionDigits: nd, maximumFractionDigits: nd});
function regime(v){
  if (v == null) return ["数据不足", css("--neutral")];
  if (v >= 70) return ["宽松", css("--loose")];
  if (v >= 55) return ["偏松", css("--semiloose")];
  if (v >= 45) return ["中性", css("--neutral")];
  if (v >= 30) return ["偏紧", css("--semitight")];
  return ["紧张", css("--tight")];
}
function idxOnOrBefore(dstr){
  let lo = 0, hi = N - 1, ans = -1;
  while (lo <= hi){ const m = (lo + hi) >> 1; if (D.dates[m] <= dstr){ ans = m; lo = m + 1; } else hi = m - 1; }
  return ans;
}

// 头部
document.getElementById("asof").textContent = `数据截至 ${D.dates[li]} · 页面生成于 ${D.generated}`;
const nowV = D.index[li];
document.getElementById("now").textContent = fmt(nowV);
const [rn, rc] = regime(nowV);
const badge = document.getElementById("regime"); badge.textContent = rn; badge.style.background = rc;
document.getElementById("deltas").innerHTML = [["一周前",5],["四周前",20],["三个月前",63],["一年前",252]].map(([t,n]) => {
  const k = li - n, v = k >= 0 ? D.index[k] : null, d = v == null ? null : nowV - v;
  const cls = d == null ? "" : d >= 0 ? "pos" : "neg";
  return `<div class="delta"><span>${t}</span><b>${fmt(v)}</b><span class="${cls}">${d == null ? "" : (d >= 0 ? "+" : "") + fmt(d)}</span></div>`;
}).join("");
document.getElementById("alerts").outerHTML = D.alerts.length
  ? `<ul class="alerts">${D.alerts.map(a => `<li>${a}</li>`).join("")}</ul>`
  : `<div class="ok">当前没有触发硬预警。</div>`;

// Plotly 通用布局
function baseLayout(extra = {}){
  return Object.assign({
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: {color: css("--ink"), size: 12},
    margin: {l: 48, r: 16, t: 10, b: 36},
    xaxis: {gridcolor: css("--grid"), linecolor: css("--line"), type: "date"},
    yaxis: {gridcolor: css("--grid"), zerolinecolor: css("--line")},
    hovermode: "x unified", showlegend: false,
    legend: {orientation: "h", y: -0.15},
  }, extra);
}
const cfg = {displaylogo: false, responsive: true, modeBarButtonsToRemove: ["lasso2d","select2d"]};

// 综合指数
const bands = [[0,30,"--band-tight"],[30,45,"--band-semitight"],[45,55,"--band-neutral"],[55,70,"--band-semiloose"],[70,100,"--band-loose"]]
  .map(([a,b,c]) => ({type:"rect", xref:"paper", x0:0, x1:1, y0:a, y1:b, fillcolor: css(c), line:{width:0}, layer:"below"}));
Plotly.newPlot("cIndex", [{x: D.dates, y: D.index, type: "scatter", mode: "lines", line: {color: css("--accent"), width: 2},
  name: "综合指数", hovertemplate: "%{y:.1f}<extra></extra>"}],
  baseLayout({shapes: bands, yaxis: {range: [0,100], gridcolor: css("--grid"), dtick: 10}}), cfg);

// 支柱
const pk = Object.keys(D.pillars), pc = ["--p1","--p2","--p3","--p4","--p5"];
Plotly.newPlot("cPillars", pk.map((p,i) => ({x: D.dates, y: D.pillars[p].z, type:"scatter", mode:"lines",
  name: `${D.pillars[p].name}(${Math.round(D.pillars[p].weight*100)}%)`, line:{color: css(pc[i]), width:1.6},
  hovertemplate: "%{y:.2f}"})),
  baseLayout({showlegend: true, yaxis: {range: [-3.2,3.2], gridcolor: css("--grid"), zerolinecolor: css("--muted")}}), cfg);

// 时间范围按钮,同时作用于三张图
const ranges = [["1年",252],["3年",756],["5年",1260],["全部",0]];
const rb = document.getElementById("rangeBtns");
function setRange(n){
  const x0 = n ? D.dates[Math.max(0, N - n)] : D.dates[0], x1 = D.dates[N-1];
  ["cIndex","cPillars","cRaw"].forEach(id => Plotly.relayout(id, {"xaxis.range": [x0, x1]}));
  [...rb.children].forEach(b => b.classList.toggle("on", +b.dataset.n === n));
}
ranges.forEach(([t,n]) => { const b = document.createElement("button"); b.textContent = t; b.dataset.n = n; b.onclick = () => setRange(n); rb.appendChild(b); });

// 单项指标
const sel = document.getElementById("rawSel");
let lastGroup = null, og = null;
Object.entries(D.raw).forEach(([k, s]) => {
  if (s.group !== lastGroup){ og = document.createElement("optgroup"); og.label = s.group; sel.appendChild(og); lastGroup = s.group; }
  const o = document.createElement("option"); o.value = k; o.textContent = s.name; og.appendChild(o);
});
function drawRaw(){
  const s = D.raw[sel.value];
  const shapes = s.threshold == null ? [] : [{type:"line", xref:"paper", x0:0, x1:1, y0:s.threshold, y1:s.threshold, line:{color: css("--tight"), dash:"dash", width:1}}];
  const xr = document.getElementById("cIndex").layout.xaxis.range;
  Plotly.react("cRaw", [{x: D.dates, y: s.v, type:"scatter", mode:"lines", line:{color: css("--accent"), width:1.8},
    hovertemplate: `%{y:,.2f} ${s.unit}<extra></extra>`}],
    baseLayout({shapes, xaxis: {gridcolor: css("--grid"), type:"date", range: xr}, yaxis: {gridcolor: css("--grid"), title: {text: s.unit}}}), cfg);
  const lv = s.v[li];
  document.getElementById("rawNote").textContent = `最新 ${fmt(lv, 2)} ${s.unit}` + (s.threshold == null ? "" : ` · 红虚线 = 预警线 ${s.threshold}`);
}
sel.onchange = drawRaw;
drawRaw();
setRange(756);

// 历史校准信号
if (!D.bt){ document.getElementById("btCard").style.display = "none"; }
else {
  const B = D.bt, now = B.now;
  document.getElementById("btPos").textContent = Math.round(now.position * 100) + "%";
  const low = now.position < 1, need = Math.max(0, now.confirm - now.streak);
  const turn = low
    ? (now.net_liq_chg > 0 ? `净流动性 13 周变化已连续 ${now.streak} 个交易日为正,再 ${need} 天仍为正即恢复满仓。` : `净流动性仍在收缩,维持半仓。`)
    : (now.net_liq_chg < 0 ? `净流动性 13 周变化已连续 ${now.streak} 个交易日为负,再 ${need} 天仍为负即减到半仓。` : `净流动性在扩张,维持满仓。`);
  document.getElementById("btNow").innerHTML =
    `<div>截至 ${now.date}:净流动性 13 周变化 <b class="${now.net_liq_chg >= 0 ? "pos" : "neg"}">${now.net_liq_chg >= 0 ? "+" : ""}${fmt(now.net_liq_chg, 0)}</b> 十亿美元。</div>` +
    `<div style="margin-top:6px">${turn}</div>` +
    `<div class="sub" style="margin-top:6px">辅助观察:5 年美债 13 周变化 ${now.ust5_chg >= 0 ? "+" : ""}${fmt(now.ust5_chg, 2)} 个百分点${now.rate_ok ? "" : "(利率冲击中,注意风险)"}。</div>`;
  // 半仓时段画成红色底色
  const segs = []; let s0 = null;
  B.position.forEach((p, i) => {
    if (p < 1 && s0 == null) s0 = i;
    if ((p >= 1 || i === B.position.length - 1) && s0 != null){ segs.push([B.dates[s0], B.dates[i]]); s0 = null; }
  });
  const shp = segs.map(([a, b]) => ({type:"rect", xref:"x", yref:"paper", x0:a, x1:b, y0:0, y1:1, fillcolor: css("--band-tight"), line:{width:0}, layer:"below"}));
  Plotly.newPlot("cBt", [{x: B.dates, y: B.spy, type:"scatter", mode:"lines", line:{color: css("--accent"), width:1.8},
    hovertemplate: "SPY %{y:,.0f}<extra></extra>"}],
    baseLayout({shapes: shp, yaxis: {type:"log", gridcolor: css("--grid")}}), cfg);
  Plotly.newPlot("cNl", [{x: B.dates, y: B.net_liq_chg, type:"bar",
    marker:{color: B.net_liq_chg.map(v => v == null ? css("--neutral") : v >= 0 ? css("--loose") : css("--tight"))},
    hovertemplate: "%{y:+,.0f} 十亿<extra></extra>"}],
    baseLayout({shapes: shp, bargap: 0, yaxis: {gridcolor: css("--grid"), zerolinecolor: css("--muted")}}), cfg);
  const bn = B.dates.length, brb = document.getElementById("btRangeBtns");
  const setBt = n => {
    const x0 = n ? B.dates[Math.max(0, bn - n)] : B.dates[0], x1 = B.dates[bn - 1];
    ["cBt","cNl"].forEach(id => Plotly.relayout(id, {"xaxis.range": [x0, x1]}));
    [...brb.children].forEach(b => b.classList.toggle("on", +b.dataset.n === n));
  };
  [["1年",252],["3年",756],["全部",0]].forEach(([t,n]) => { const b = document.createElement("button"); b.textContent = t; b.dataset.n = n; b.onclick = () => setBt(n); brb.appendChild(b); });
  setBt(0);
  const pct = v => v == null ? "—" : (v * 100).toFixed(1) + "%";
  const oos = Object.fromEntries(B.table_oos.map(r => [r.rule, r]));
  document.getElementById("btTable").innerHTML =
    `<tr><th>规则</th><th>年化</th><th>同仓位对照</th><th>最大回撤</th><th>平均仓位</th><th>样本外年化</th><th>样本外对照</th></tr>` +
    B.table.map(r => `<tr><td>${r.rule}</td><td>${pct(r.cagr)}</td><td>${pct(r.const)}</td><td>${pct(r.maxdd)}</td><td>${Math.round(r.exposure*100)}%</td>` +
      `<td>${pct(oos[r.rule]?.cagr)}</td><td>${pct(oos[r.rule]?.const)}</td></tr>`).join("");
}

// 两天对比
const dA = document.getElementById("dA"), dB = document.getElementById("dB");
dA.min = dB.min = D.dates[0]; dA.max = dB.max = D.dates[N-1];
dA.value = D.dates[li]; dB.value = D.dates[Math.max(0, li - 252)];
function cmp(){
  const a = idxOnOrBefore(dA.value), b = idxOnOrBefore(dB.value);
  if (a < 0 || b < 0){ document.getElementById("cmp").innerHTML = "<tr><td>所选日期早于数据起点</td></tr>"; return; }
  const row = (name, va, vb, nd, unit = "") => {
    const d = va == null || vb == null ? null : va - vb;
    const cls = d == null ? "" : "";
    return `<tr><td>${name}</td><td>${fmt(va, nd)}${unit}</td><td>${fmt(vb, nd)}${unit}</td><td>${d == null ? "—" : (d >= 0 ? "+" : "") + fmt(d, nd)}</td></tr>`;
  };
  let h = `<tr><th>项目</th><th>A:${D.dates[a]}</th><th>B:${D.dates[b]}</th><th>A − B</th></tr>`;
  h += row("<b>综合流动性指数</b>", D.index[a], D.index[b], 1);
  pk.forEach(p => h += row(D.pillars[p].name + "(z)", D.pillars[p].z[a], D.pillars[p].z[b], 2));
  Object.values(D.raw).forEach(s => h += row(s.name, s.v[a], s.v[b], 2, s.unit && s.unit !== "点" ? " " + s.unit : ""));
  document.getElementById("cmp").innerHTML = h;
}
dA.onchange = dB.onchange = cmp;
document.querySelectorAll("button[data-back]").forEach(b => b.onclick = () => {
  const a = idxOnOrBefore(dA.value); dB.value = D.dates[Math.max(0, a - +b.dataset.back)]; cmp();
});
cmp();

// 贡献
document.getElementById("contrib").innerHTML = `<tr><th>支柱 / 序列</th><th>z</th><th style="width:40%"></th></tr>` +
  D.contrib.map(c => {
    const z = c.z, w = z == null ? 0 : Math.min(Math.abs(z), 3) / 3 * 50;
    const bar = z == null ? "" : `<span style="display:inline-block;width:50%;text-align:right">${z < 0 ? `<span class="bar" style="width:${w*2}%;background:${css("--tight")}"></span>` : ""}</span><span style="display:inline-block;width:50%;text-align:left">${z > 0 ? `<span class="bar" style="width:${w*2}%;background:${css("--loose")}"></span>` : ""}</span>`;
    return `<tr><td><span class="sub">${c.pillar}</span><br>${c.name}</td><td class="${z == null ? "" : z >= 0 ? "pos" : "neg"}">${fmt(z, 2)}</td><td>${bar}</td></tr>`;
  }).join("");

// 运行记录
document.getElementById("log").innerHTML = `<tr><th>运行日</th><th>数据日</th><th>指数</th><th>状态</th></tr>` +
  [...D.log].reverse().slice(0, 30).map(r => `<tr><td>${r.run_date}</td><td>${r.data_date}</td><td>${fmt(r.liquidity_index)}</td><td>${r.regime}</td></tr>`).join("");

// 结算日历
document.getElementById("cal").innerHTML = D.calendar.length
  ? `<tr><th>结算日</th><th>毛额(十亿美元)</th><th>品种</th></tr>` + D.calendar.map(c => `<tr><td>${c.date}</td><td>${c.gross_bn == null ? "待公布" : fmt(c.gross_bn)}</td><td>${c.detail}</td></tr>`).join("")
  : `<tr><td class="sub">暂无数据</td></tr>`;
</script>
</body>
</html>
"""
