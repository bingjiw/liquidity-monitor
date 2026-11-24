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
    <h2>当前操作信号(净流动性规则)</h2>
    <div class="hero">
      <div>
        <div class="sub">建议股票仓位</div>
        <div class="big" id="btPos"></div>
      </div>
      <div style="flex:1;min-width:260px" id="btNow"></div>
    </div>
    <div id="cNl" class="chart" style="height:220px;margin-top:8px"></div>
    <div class="note">净流动性(联储总资产 − TGA − 逆回购)13 周变化,十亿美元。红色底色 = 规则处于半仓的时段。规则:13 周变化连续 20 个交易日为负 → 半仓;连续 20 个交易日为正 → 满仓。为什么选它,见下面的历史模拟。</div>
  </div>

  <div class="card" id="simCard">
    <h2>历史模拟:用不同数据指导操作,结果如何</h2>
    <div class="sub">2019-03 起,假如每天按某条规则调整美股仓位。信号按数据实际公布时间滞后计算,第二天执行,每次调仓扣 0.05% 成本,没买股票的钱拿联邦基金利率。</div>
    <div class="row" style="margin-top:12px" id="simAsset"></div>
    <div class="row" id="simPeriod"></div>
    <div class="row" id="simRules"></div>
    <div id="simDesc" class="note" style="margin-top:0"></div>
    <h2 style="margin-top:14px;font-size:15px" id="simPxTitle"></h2>
    <div id="cSimPx" class="chart"></div>
    <div class="note">红色底色 = 所选规则减仓的时段(颜色越深仓位越低)。虚线 = 样本内 / 样本外分界(2023-01)。</div>
    <h2 style="margin-top:14px;font-size:15px">资金曲线(起点 = 1)</h2>
    <div id="cSimEq" class="chart"></div>
    <div class="note">粗线 = 所选规则。点击图例可隐藏或显示某条线。</div>
    <h2 style="margin-top:18px;font-size:15px">各规则成绩对比</h2>
    <div class="scroll"><table id="simTable"></table></div>
    <div class="note">“同仓位对照”= 一直持有和该规则平均仓位相同比例的股票。“择时增益”= 年化 − 同仓位对照,只有它为正才说明这套数据真的帮上了忙,而不只是少买了股票。“判断正确率”= 每次调仓之后到下一次调仓之间:减仓段股票跑输现金、加仓段股票跑赢现金,算判断正确。</div>
    <div class="grid2" style="margin-top:14px">
      <div>
        <h2 style="font-size:15px">所选规则:之后三个月的概率</h2>
        <div class="scroll"><table id="simStates"></table></div>
        <div class="note">按规则处于满仓或减仓状态的每一天,统计之后 63 个交易日股票的涨跌。减仓时的上涨概率和平均涨幅如果明显低于满仓时,说明这套数据能区分好坏时期;反过来就说明它常在不该减仓的时候减仓。</div>
      </div>
      <div>
        <h2 style="font-size:15px">所选规则:逐年收益</h2>
        <div class="scroll"><table id="simYears"></table></div>
      </div>
    </div>
    <h2 style="margin-top:18px;font-size:15px">所选规则:每一次调仓</h2>
    <div class="scroll"><table id="simSegs"></table></div>
    <div class="note">历史表现不代表未来。2020 年疫情、2025 年 4 月关税这类突发暴跌,流动性数据事先看不出来。完整报告见仓库 output/backtest_summary.md。</div>
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
if (!D.bt){ document.getElementById("btCard").style.display = "none"; document.getElementById("simCard").style.display = "none"; }
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
  const bandsOf = pos => {
    const out = []; let s0 = null;
    for (let i = 0; i <= pos.length; i++){
      const p = i < pos.length ? pos[i] : 1;
      if (s0 != null && (p !== pos[s0] || i === pos.length)){ out.push({x0: B.dates[s0], x1: B.dates[Math.min(i, pos.length - 1)], p: pos[s0]}); s0 = null; }
      if (s0 == null && p < 1 && i < pos.length) s0 = i;
    }
    return out.map(g => ({type:"rect", xref:"x", yref:"paper", x0:g.x0, x1:g.x1, y0:0, y1:1, line:{width:0}, layer:"below",
      fillcolor: css("--tight"), opacity: g.p <= 0.5 ? 0.16 : 0.08}));
  };
  const nlRule = B.rules.find(r => r.key === "nl");
  Plotly.newPlot("cNl", [{x: B.dates, y: B.net_liq_chg, type:"bar",
    marker:{color: B.net_liq_chg.map(v => v == null ? css("--neutral") : v >= 0 ? css("--loose") : css("--tight"))},
    hovertemplate: "%{y:+,.0f} 十亿<extra></extra>"}],
    baseLayout({shapes: bandsOf(nlRule.position), bargap: 0, yaxis: {gridcolor: css("--grid"), zerolinecolor: css("--muted")}}), cfg);

  // ---- 历史模拟
  const RC = {bh: "--muted", lvl: "--p2", trd: "--p4", nl: "--p1", nlr: "--p5", ma: "--p3"};
  const st = {asset: "SPY", period: B.periods[2], rule: "nl"};
  const pct = (v, nd = 1, sign = false) => v == null ? "—" : (sign && v >= 0 ? "+" : "") + (v * 100).toFixed(nd) + "%";
  const periodRange = lab => lab.startsWith("样本内") ? [B.dates[0], "2022-12-31"] : lab.startsWith("样本外") ? [B.split, B.dates[B.dates.length - 1]] : [B.dates[0], B.dates[B.dates.length - 1]];
  const chips = (id, items, key) => {
    const el = document.getElementById(id); el.innerHTML = "";
    items.forEach(([v, t, col]) => { const b = document.createElement("button"); b.textContent = t; b.dataset.v = v;
      if (col) b.style.borderLeft = `4px solid ${css(col)}`;
      b.onclick = () => { st[key] = v; draw(); }; el.appendChild(b); });
  };
  chips("simAsset", [["SPY","标普 500(SPY)"],["QQQ","纳斯达克 100(QQQ)"]], "asset");
  chips("simPeriod", B.periods.map(p => [p, p]), "period");
  chips("simRules", B.rules.map(r => [r.key, r.name, RC[r.key]]), "rule");
  const mark = () => [["simAsset","asset"],["simPeriod","period"],["simRules","rule"]].forEach(([id,k]) =>
    [...document.getElementById(id).children].forEach(b => b.classList.toggle("on", b.dataset.v === st[k])));
  function draw(){
    mark();
    const R = B.rules.find(r => r.key === st.rule), A = R.assets[st.asset];
    const [x0, x1] = periodRange(st.period);
    const i0 = Math.max(0, B.dates.findIndex(d => d >= x0)), i1 = B.dates.length - 1 - [...B.dates].reverse().findIndex(d => d <= x1);
    document.getElementById("simDesc").textContent = R.desc;
    document.getElementById("simPxTitle").textContent = `${st.asset} 走势与「${R.name}」的仓位`;
    const px = st.asset === "SPY" ? B.spy : B.qqq;
    const split = {type:"line", xref:"x", yref:"paper", x0:B.split, x1:B.split, y0:0, y1:1, line:{color: css("--muted"), dash:"dot", width:1}};
    const posTxt = R.position.map(p => Math.round(p * 100) + "%");
    Plotly.react("cSimPx", [{x: B.dates, y: px, type:"scatter", mode:"lines", line:{color: css("--ink"), width:1.5},
      customdata: posTxt, hovertemplate: `${st.asset} %{y:,.0f} · 仓位 %{customdata}<extra></extra>`}],
      baseLayout({shapes: [...bandsOf(R.position), split], xaxis: {type:"date", range:[x0, x1], gridcolor: css("--grid")},
        yaxis: {type:"log", gridcolor: css("--grid")}}), cfg);
    const traces = B.rules.map(r => {
      const e = r.assets[st.asset].equity, base = i0 > 0 ? e[i0 - 1] : 1;
      const sel = r.key === st.rule;
      return {x: B.dates.slice(i0, i1 + 1), y: e.slice(i0, i1 + 1).map(v => v / base), type:"scatter", mode:"lines", name: r.name,
        line:{color: css(RC[r.key]), width: sel ? 3 : 1.2, dash: r.key === "bh" ? "dot" : "solid"}, opacity: sel || r.key === "bh" ? 1 : 0.55,
        hovertemplate: "%{y:.2f}"};
    });
    Plotly.react("cSimEq", traces, baseLayout({showlegend: true, shapes: [split], xaxis: {type:"date", range:[x0, x1], gridcolor: css("--grid")},
      yaxis: {type:"log", gridcolor: css("--grid")}}), cfg);
    // 对比表
    document.getElementById("simTable").innerHTML =
      `<tr><th>规则</th><th>年化</th><th>同仓位对照</th><th>择时增益</th><th>最大回撤</th><th>夏普</th><th>平均仓位</th><th>调仓次数</th><th>判断正确率</th></tr>` +
      B.rules.map(r => {
        const s = r.assets[st.asset].stats[st.period], g = s.cagr - s.const, sel = r.key === st.rule;
        return `<tr style="${sel ? "background:var(--bg);font-weight:600" : ""};cursor:pointer" data-k="${r.key}"><td><span class="bar" style="width:10px;background:${css(RC[r.key])}"></span> ${r.name}</td>` +
          `<td>${pct(s.cagr)}</td><td>${pct(s.const)}</td><td class="${r.key === "bh" ? "" : g >= 0 ? "pos" : "neg"}">${r.key === "bh" ? "—" : pct(g, 1, true)}</td>` +
          `<td class="neg">${pct(s.maxdd)}</td><td>${fmt(s.sharpe, 2)}</td><td>${pct(s.exposure, 0)}</td><td>${s.switches}</td>` +
          `<td>${r.assets[st.asset].hit == null ? "—" : pct(r.assets[st.asset].hit, 0) + `(${r.assets[st.asset].n_closed} 次)`}</td></tr>`;
      }).join("");
    document.querySelectorAll("#simTable tr[data-k]").forEach(tr => tr.onclick = () => { st.rule = tr.dataset.k; draw(); });
    // 概率
    const S = A.states, row = (t, x) => x.days ? `<tr><td>${t}</td><td>${x.days}</td><td class="${x.avg >= 0 ? "pos" : "neg"}">${pct(x.avg, 1, true)}</td><td>${pct(x.up, 0)}</td><td>${pct(x.bad, 0)}</td></tr>` : `<tr><td>${t}</td><td>0</td><td colspan="3" class="sub">从未出现</td></tr>`;
    document.getElementById("simStates").innerHTML = `<tr><th>规则状态</th><th>天数</th><th>之后 3 个月平均</th><th>上涨概率</th><th>跌超 10% 概率</th></tr>` +
      row("满仓时", S.full) + row("减仓时", S.reduced);
    // 逐年
    const bhY = B.rules[0].assets[st.asset].yearly;
    document.getElementById("simYears").innerHTML = `<tr><th>年份</th><th>${R.name}</th><th>买入持有</th><th>差</th></tr>` +
      Object.keys(A.yearly).map(y => { const d = A.yearly[y] - bhY[y];
        return `<tr><td>${y}${y === B.dates[0].slice(0,4) ? "(3 月起)" : y === B.dates[B.dates.length-1].slice(0,4) ? "(至今)" : ""}</td><td>${pct(A.yearly[y])}</td><td>${pct(bhY[y])}</td><td class="${d >= 0 ? "pos" : "neg"}">${pct(d, 1, true)}</td></tr>`; }).join("");
    // 每段
    const segs = A.segments.slice(1);
    document.getElementById("simSegs").innerHTML = segs.length
      ? `<tr><th>开始</th><th>结束</th><th>仓位</th><th>期间 ${st.asset}</th><th>期间现金</th><th>判断</th></tr>` +
        [...segs].reverse().map(g => `<tr><td>${g.start}</td><td>${g.open ? "进行中" : g.end}</td><td>${Math.round(g.pos * 100)}%</td>` +
          `<td class="${g.asset >= 0 ? "pos" : "neg"}">${pct(g.asset, 1, true)}</td><td>${pct(g.cash, 1, true)}</td>` +
          `<td>${g.open ? "—" : g.right ? "✓ 正确" : "✗ 错误"}</td></tr>`).join("")
      : `<tr><td class="sub">这条规则从不调仓。</td></tr>`;
  }
  draw();
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
