"""市场传动图:把金融市场的因果关系画成一台机器,每个部件按当天的真实数据转动。

按“下跌四层”框架组织(层数越深越严重,解药也不同):
  第一层 估值   —— 利率与长久期估值(杠杆)、收益率与债券价格(齿轮)、
                   股债相关性和黄金看实际利率(离合器,会换挡)
  第二层 杠杆   —— 强平 / 轧空(飞轮,正反馈)
  第三层 流动性 —— 缺现金时连好资产一起卖(抽水泵)
  第四层 信用   —— 借不到钱、违约(信用压力表)

部件的判定线是经验值,没有回测验证,只做展示,不进仓位规则。
页面部分也放在这里:HTML(卡片和样式)插在模板的 __GEARS__ 处,SCRIPT 插在主脚本之后。
"""
from __future__ import annotations

import math

import pandas as pd

DAYS = 756            # 页面上可回放约三年
WIN = 20              # 四周变化
CORR_WIN = 60         # 离合器用 60 日相关
CLUTCH_BAND = 0.2     # 相关性在 ±0.2 之间算“空挡”
MARGIN_LINE = 12.15   # CCC OAS 预警线,与 liquidity_monitor.ALERTS 一致

LAYERS = [
    {"n": 1, "name": "估值", "fix": "利率回落,或盈利预期回升", "desc": "利率涨、盈利预期降,长久期资产先跌"},
    {"n": 2, "name": "杠杆", "fix": "时间:强平盘卖完,飞轮自己停下", "desc": "借钱买的仓位被迫卖出,越跌越卖"},
    {"n": 3, "name": "流动性", "fix": "央行放水:SRF、购买短债、降息", "desc": "缺现金,好资产也被抽走换钱"},
    {"n": 4, "name": "信用", "fix": "救助和财政兜底", "desc": "借不到钱,违约开始传染"},
]


def _r(s: pd.Series, nd: int):
    return [None if pd.isna(v) or (isinstance(v, float) and math.isinf(v)) else round(float(v), nd) for v in s]


def compute(f: pd.DataFrame, raw: pd.DataFrame, m: pd.DataFrame, vix: pd.Series | None = None) -> dict | None:
    need = {"QQQ", "SPY", "IEF", "GLD"}
    if m is None or m.empty or not need <= set(m.columns):
        return None
    px = m[sorted(need)].dropna()
    if len(px) < CORR_WIN + WIN:
        return None
    idx = px.index
    ff = f.ffill().reindex(idx, method="ffill")
    ret = px.pct_change()

    # ---- 第一层:估值 ----
    tips10 = ff["tips10"]
    tips_chg = (tips10 - tips10.shift(WIN)) * 100              # BP
    ust5 = ff["ust5"]
    ust5_chg = (ust5 - ust5.shift(WIN)) * 100                  # BP
    ief20 = px["IEF"] / px["IEF"].shift(WIN) - 1
    qqq20 = px["QQQ"] / px["QQQ"].shift(WIN) - 1
    qqq_dd = px["QQQ"] / px["QQQ"].rolling(60, min_periods=1).max() - 1
    sb_corr = ret["SPY"].rolling(CORR_WIN).corr(ret["IEF"])     # 股债相关
    dtips = tips10.diff()
    gr_corr = ret["GLD"].rolling(CORR_WIN).corr(dtips)          # 黄金 vs 实际利率变化
    gld5 = px["GLD"] / px["GLD"].shift(5) - 1
    qqq5 = px["QQQ"] / px["QQQ"].shift(5) - 1
    tips5 = (tips10 - tips10.shift(5)) * 100

    # ---- 第二层:杠杆(飞轮) ----
    if vix is None or vix.empty:
        vix = ff["vix"] if "vix" in ff else pd.Series(index=idx, dtype=float)
    vx = vix.reindex(idx.union(vix.index)).ffill().reindex(idx)
    vix_max10 = vx.rolling(10, min_periods=1).max()
    both_down = (ret["GLD"] < -0.01) & (ret["QQQ"] < -0.01)
    dd20 = both_down.rolling(20, min_periods=1).sum()
    # 正数 = 强平(向下转),负数 = 轧空(向上转),0–1
    sell = ((-qqq5 - 0.02) / 0.06).clip(0, 1) * ((vx - 18) / 14).clip(0, 1)
    squeeze = ((qqq5 - 0.04) / 0.06).clip(0, 1) * ((vix_max10 - 22) / 10).clip(0, 1) * (qqq_dd.shift(5) < -0.08)
    fly = (sell - squeeze).where(sell >= squeeze, -squeeze)

    # ---- 第三层:流动性(抽水泵) ----
    sofr = ff["sofr_iorb"] if "sofr_iorb" in ff else pd.Series(index=idx, dtype=float)
    usd = ff["usd_chg"] if "usd_chg" in ff else pd.Series(index=idx, dtype=float)
    jpy = ff["usdjpy_chg"] if "usdjpy_chg" in ff else pd.Series(index=idx, dtype=float)
    c_pipe = sofr >= 3
    c_dump = (dd20 >= 2) & (qqq5 < 0) & (gld5 < 0)
    c_usd = usd > 2
    c_jpy = jpy < -3
    pump = c_pipe.astype(int) + c_dump.astype(int) + c_usd.astype(int) + c_jpy.astype(int)

    # ---- 第四层:信用 ----
    hy = ff["hy_oas"] if "hy_oas" in ff else pd.Series(index=idx, dtype=float)
    ccc = ff["ccc_oas"] if "ccc_oas" in ff else pd.Series(index=idx, dtype=float)
    hy_chg = (hy - hy.shift(WIN)) * 100

    # ---- 各层状态:0 平静 / 1 警戒 / 2 运转 ----
    def lv(hot, warm):
        return (hot.fillna(False).astype(int) * 2).where(hot.fillna(False), warm.fillna(False).astype(int))
    st1 = lv((tips_chg > 25) | (qqq_dd < -0.10), (tips_chg > 10) | (qqq_dd < -0.05))
    st2 = lv(fly.abs() > 0.15, (qqq5 < -0.02) | (vx > 22))
    st3 = lv(pump >= 2, pump == 1)
    st4 = lv((hy > 5) | (hy_chg > 100), (ccc > MARGIN_LINE) | (hy_chg > 50))

    sl = slice(-DAYS, None)
    out = {
        "dates": [d.strftime("%Y-%m-%d") for d in idx[sl]],
        "layers": LAYERS,
        "st": [[int(v) for v in s.iloc[sl]] for s in (st1, st2, st3, st4)],
        "tips10": _r(tips10.iloc[sl], 2), "tips_chg": _r(tips_chg.iloc[sl], 0),
        "ust5": _r(ust5.iloc[sl], 2), "ust5_chg": _r(ust5_chg.iloc[sl], 0),
        "ief20": _r(ief20.iloc[sl], 4), "qqq20": _r(qqq20.iloc[sl], 4), "qqq_dd": _r(qqq_dd.iloc[sl], 4),
        "sb_corr": _r(sb_corr.iloc[sl], 2), "gr_corr": _r(gr_corr.iloc[sl], 2),
        "gld5": _r(gld5.iloc[sl], 4), "qqq5": _r(qqq5.iloc[sl], 4), "tips5": _r(tips5.iloc[sl], 0),
        "vix": _r(vx.iloc[sl], 1), "dd20": _r(dd20.iloc[sl], 0), "fly": _r(fly.iloc[sl], 2),
        "sofr": _r(sofr.iloc[sl], 0), "usd": _r(usd.iloc[sl], 2), "jpy": _r(jpy.iloc[sl], 2),
        "pump": _r(pump.iloc[sl], 0),
        "hy": _r(hy.iloc[sl], 2), "ccc": _r(ccc.iloc[sl], 2), "hy_chg": _r(hy_chg.iloc[sl], 0),
        "ccc_line": MARGIN_LINE, "band": CLUTCH_BAND,
    }
    return out


def clutch_sb(c):
    if c is None:
        return "—"
    return "通胀挡(股债同向)" if c > CLUTCH_BAND else "衰退挡(股债反向)" if c < -CLUTCH_BAND else "空挡"


def report_line(g: dict | None) -> str | None:
    """日报里的一行摘要。"""
    if not g:
        return None
    st = [int(s[-1] or 0) for s in g["st"]]
    hot = [i for i, s in enumerate(st) if s == 2]
    warm = [i for i, s in enumerate(st) if s == 1]
    names = [f"第{L['n']}层{L['name']}" for L in LAYERS]
    if hot:
        head = f"卡在{names[max(hot)]}"
    elif warm:
        head = f"无层在运转,{names[max(warm)]}警戒"
    else:
        head = "四层都平静"
    parts = "、".join(f"{names[i]}{['平静', '警戒', '运转'][s]}" for i, s in enumerate(st))
    return (f"- **市场传动图**({g['dates'][-1]}):{head}。{parts}。"
            f"股债离合器:{clutch_sb(g['sb_corr'][-1])},60 日相关 {g['sb_corr'][-1]:+.2f}。")


# ---------------------------------------------------------------------------
# 页面部分
# ---------------------------------------------------------------------------
HTML = r"""
  <div class="card" id="gearCard">
    <h2>市场传动图:把四层下跌画成一台机器</h2>
    <div class="sub">每个部件都按当天的真实数据转动。齿轮和杠杆是确定的反向关系;离合器会换挡,同一组齿轮在不同时期咬合方向相反;飞轮是正反馈,一推就越转越快;抽水泵代表缺现金时什么都卖。</div>
    <div class="gHead">
      <div><div class="sub" id="gDate"></div><div class="gNow" id="gNow"></div></div>
      <div class="gLadder" id="gLadder"></div>
    </div>
    <div class="row" style="margin-top:6px">
      <button id="gPlay">▶ 回放近三年</button>
      <input type="range" id="gSlider" min="0" max="0" value="0" style="flex:1;min-width:180px">
      <button data-gj="2025-04-08">2025-04 关税</button><button data-gj="2024-08-05">2024-08 套息平仓</button><button id="gToday">今天</button>
    </div>
    <div class="gLayer"><div class="gTag" data-l="0"></div>
      <div class="gTiles">
        <div class="gTile"><svg id="gsLever" viewBox="0 0 320 190"></svg><div class="gCap" id="gcLever"></div></div>
        <div class="gTile"><svg id="gsGear" viewBox="0 0 320 190"></svg><div class="gCap" id="gcGear"></div></div>
        <div class="gTile"><svg id="gsSB" viewBox="0 0 320 190"></svg><div class="gCap" id="gcSB"></div></div>
        <div class="gTile"><svg id="gsGold" viewBox="0 0 320 190"></svg><div class="gCap" id="gcGold"></div></div>
      </div></div>
    <div class="gLayer"><div class="gTag" data-l="1"></div>
      <div class="gTiles">
        <div class="gTile"><svg id="gsFly" viewBox="0 0 320 190"></svg><div class="gCap" id="gcFly"></div></div>
        <div class="gTile gText" id="gtFly"></div>
      </div></div>
    <div class="gLayer"><div class="gTag" data-l="2"></div>
      <div class="gTiles">
        <div class="gTile"><svg id="gsPump" viewBox="0 0 320 190"></svg><div class="gCap" id="gcPump"></div></div>
        <div class="gTile gText" id="gtPump"></div>
      </div></div>
    <div class="gLayer"><div class="gTag" data-l="3"></div>
      <div class="gTiles">
        <div class="gTile"><svg id="gsCredit" viewBox="0 0 320 190"></svg><div class="gCap" id="gcCredit"></div></div>
        <div class="gTile gText" id="gtCredit"></div>
      </div></div>
    <div class="note">判定线是经验值,没有回测验证,只帮助看懂市场卡在哪一层,不进仓位规则。层的状态:灰 = 平静,橙 = 警戒,红 = 运转。第一层看 10 年 TIPS 四周变化和纳指(QQQ)离 60 日高点的回撤;第二层看纳指 5 日涨跌和 VIX;第三层看下面列出的 4 个抽水条件,满足 2 个算运转;第四层看高收益债利差。离合器用 60 日日收益相关性,在 ±0.2 之间算空挡。</div>
  </div>
<style>
.gHead{display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between;margin-top:10px}
.gNow{font-size:20px;font-weight:700;margin-top:2px}
.gLadder{display:flex;gap:6px;flex-wrap:wrap}
.gLadder span{padding:4px 10px;border-radius:999px;font-size:13px;font-weight:600;background:var(--bg);border:1px solid var(--line);color:var(--muted)}
.gLadder span.s1{background:var(--band-semitight);border-color:var(--semitight);color:var(--ink)}
.gLadder span.s2{background:var(--tight);border-color:var(--tight);color:#fff}
.gLayer{display:grid;grid-template-columns:120px 1fr;gap:10px;margin-top:12px;padding-top:12px;border-top:1px solid var(--line)}
@media (max-width:760px){.gLayer{grid-template-columns:1fr}}
.gTag{font-size:13px;color:var(--muted)}
.gTag b{display:block;font-size:16px;color:var(--ink)}
.gTag .dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px;vertical-align:middle}
.gTiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:10px}
.gTile{background:var(--bg);border-radius:10px;padding:8px 10px;font-size:13px}
.gTile svg{width:100%;height:auto;display:block}
.gCap{color:var(--muted);margin-top:2px}
.gCap b{color:var(--ink)}
.gText{line-height:1.7}
.gText .ck{margin:2px 0}
#gearCard svg text{font-size:12px;fill:var(--ink)}
#gearCard svg .mut{fill:var(--muted)}
#gearCard svg .big{font-size:15px;font-weight:700}
</style>
"""

SCRIPT = r"""
<script>
(function(){
const G = typeof D !== "undefined" && D.watch && D.watch.gears;
const card = document.getElementById("gearCard");
if (!G){ card.style.display = "none"; return; }
const $ = id => document.getElementById(id);
const C = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const n = G.dates.length;
let I = n - 1;
const f1 = (v, nd = 1) => v == null ? "—" : Number(v).toFixed(nd);
const sg = (v, nd = 0) => v == null ? "—" : (v >= 0 ? "+" : "") + Number(v).toFixed(nd);
const pc = (v, nd = 1) => v == null ? "—" : (v >= 0 ? "+" : "") + (v * 100).toFixed(nd) + "%";
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const stCol = s => s === 2 ? C("--tight") : s === 1 ? C("--semitight") : C("--neutral");
const stName = s => ["平静", "警戒", "运转"][s || 0];
const NS = "http://www.w3.org/2000/svg";
function el(tag, attrs, parent){ const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; }
function txt(parent, x, y, s, cls, anchor){ const t = el("text", {x, y, "text-anchor": anchor || "middle"}, parent); if (cls) t.setAttribute("class", cls); t.textContent = s; return t; }
function gearPath(r, teeth, depth){
  let d = ""; const step = Math.PI * 2 / teeth;
  for (let k = 0; k < teeth; k++){
    const a = k * step, pts = [[a, r - depth], [a + step * .15, r], [a + step * .45, r], [a + step * .6, r - depth]];
    pts.forEach(([ang, rr], j) => { d += (k === 0 && j === 0 ? "M" : "L") + (rr * Math.cos(ang)).toFixed(2) + " " + (rr * Math.sin(ang)).toFixed(2); });
  }
  return d + "Z";
}
function gear(parent, cx, cy, r, teeth, fill){
  const g = el("g", {transform: `translate(${cx} ${cy})`}, parent);
  const rot = el("g", {}, g);
  el("path", {d: gearPath(r, teeth, 6), fill, stroke: C("--line")}, rot);
  el("circle", {r: r * .35, fill: C("--card"), stroke: C("--line")}, rot);
  el("line", {x1: 0, y1: -r * .35, x2: 0, y2: -r + 7, stroke: C("--card"), "stroke-width": 3}, rot);
  el("circle", {r: 4, fill: C("--muted")}, g);
  return {rot, angle: 0};
}

// ---------- 静态骨架 ----------
// 第一层 A:杠杆
const sL = $("gsLever");
const PIV = 70, PY = 105;
el("path", {d: `M${PIV} ${PY} L${PIV - 16} ${PY + 34} L${PIV + 16} ${PY + 34} Z`, fill: C("--muted")}, sL);
el("line", {x1: 20, y1: PY + 34, x2: 310, y2: PY + 34, stroke: C("--line"), "stroke-width": 2}, sL);
const beam = el("g", {}, sL);
el("rect", {x: 30, y: -4, width: 280, height: 8, rx: 4, fill: C("--accent")}, beam);
el("circle", {cx: 40, cy: 0, r: 11, fill: C("--p2")}, beam);
const tR = txt(beam, 40, 4, "利率", "", "middle"); tR.setAttribute("style", "font-size:9px;fill:#fff");
const ARMS = [[1, 95], [10, 185], [20, 290]];
const armLab = ARMS.map(([y, x]) => { el("circle", {cx: x, cy: 0, r: 6, fill: C("--p4")}, beam); return txt(beam, x, -12, "", "", "middle"); });
ARMS.map(([y, x]) => txt(beam, x, 20, `${y}年`, "mut", "middle"));
txt(sL, 12, 20, "支点越远,翘得越厉害", "mut", "start");
const levTitle = txt(sL, 12, 182, "", "", "start");

// 第一层 B:齿轮
const sG = $("gsGear");
const gA = gear(sG, 100, 92, 52, 14, C("--p2")), gB = gear(sG, 202, 92, 52, 14, C("--p1"));
gB.angle = 180 / 14;
txt(sG, 100, 170, "5 年美债收益率", "", "middle"); txt(sG, 202, 170, "债券价格(IEF)", "", "middle");
const gAv = txt(sG, 100, 22, "", "big", "middle"), gBv = txt(sG, 202, 22, "", "big", "middle");

// 离合器(换挡杆):三个挡位
function gate(svg, labels){
  el("rect", {x: 30, y: 30, width: 260, height: 10, rx: 5, fill: C("--line")}, svg);
  const xs = [55, 160, 265];
  xs.forEach((x, k) => { el("rect", {x: x - 5, y: 30, width: 10, height: 70, rx: 5, fill: C("--line")}, svg); txt(svg, x, 120, labels[k], "", "middle"); });
  el("line", {x1: 30, y1: 155, x2: 290, y2: 155, stroke: C("--line"), "stroke-width": 2}, svg);
  [-1, 0, 1].forEach(v => { const x = 160 + v * 130; el("line", {x1: x, y1: 150, x2: x, y2: 160, stroke: C("--muted")}, svg); txt(svg, x, 176, v === 0 ? "0" : (v > 0 ? "+1" : "−1"), "mut", "middle"); });
  const zone = el("rect", {x: 160 - .2 * 130, y: 150, width: .4 * 130, height: 10, fill: C("--band-neutral")}, svg);
  const mark = el("path", {d: "M0 0 L-6 -10 L6 -10 Z", fill: C("--accent")}, svg);
  const knob = el("g", {}, svg);
  el("line", {x1: 0, y1: 0, x2: 0, y2: -28, stroke: C("--muted"), "stroke-width": 4}, knob);
  const ball = el("circle", {cx: 0, cy: -32, r: 12, fill: C("--accent")}, knob);
  const val = txt(svg, 300, 22, "", "big", "end");
  txt(svg, 20, 22, "60 日相关", "mut", "start");
  return {xs, knob, ball, mark, val, kx: 160, ky: 95};
}
const gSB = gate($("gsSB"), ["衰退挡:反向", "空挡", "通胀挡:同向"]);
const gGo = gate($("gsGold"), ["利率挡:反向", "脱钩", "同向:砍仓/避险"]);

// 第二层:飞轮
const sF = $("gsFly");
const fly = el("g", {transform: "translate(95 95)"}, sF);
const flyRot = el("g", {}, fly);
el("circle", {r: 70, fill: "none", stroke: C("--muted"), "stroke-width": 12}, flyRot);
for (let k = 0; k < 6; k++){ const a = k * Math.PI / 3; el("line", {x1: 0, y1: 0, x2: 64 * Math.cos(a), y2: 64 * Math.sin(a), stroke: C("--muted"), "stroke-width": 4}, flyRot); }
const flyMark = el("circle", {cx: 0, cy: -70, r: 7, fill: C("--tight")}, flyRot);
el("circle", {r: 10, fill: C("--card"), stroke: C("--muted"), "stroke-width": 3}, fly);
const flyDir = txt(sF, 245, 70, "", "big", "middle");
const flySpd = txt(sF, 245, 95, "", "", "middle");
const flyBarBg = el("rect", {x: 185, y: 110, width: 120, height: 10, rx: 5, fill: C("--line")}, sF);
const flyBar = el("rect", {x: 185, y: 110, width: 0, height: 10, rx: 5, fill: C("--tight")}, sF);
txt(sF, 245, 140, "跌 → 强平 → 再跌", "mut", "middle");

// 第三层:抽水泵
const sP = $("gsPump");
el("rect", {x: 20, y: 40, width: 110, height: 120, rx: 8, fill: "none", stroke: C("--muted"), "stroke-width": 2}, sP);
const water = el("rect", {x: 22, y: 100, width: 106, height: 58, rx: 6, fill: C("--accent"), opacity: .55}, sP);
txt(sP, 75, 30, "市场里的现金", "mut", "middle");
el("path", {d: "M130 140 L175 140 L175 95", fill: "none", stroke: C("--muted"), "stroke-width": 10}, sP);
const pump = gear(sP, 200, 80, 34, 8, C("--p3"));
el("path", {d: "M234 80 L300 80", fill: "none", stroke: C("--muted"), "stroke-width": 10}, sP);
const pumpTxt = txt(sP, 200, 150, "", "big", "middle");
txt(sP, 268, 70, "卖资产换钱", "mut", "middle");

// 第四层:信用压力表
const sC = $("gsCredit");
const CX = 160, CY = 135, R = 105, lo = 2, hi = 10;
const ang = v => Math.PI * (1 - (clamp(v, lo, hi) - lo) / (hi - lo));
const arc = (a, b, col) => { const p = (v) => [CX + R * Math.cos(ang(v)), CY - R * Math.sin(ang(v))]; const [x1, y1] = p(a), [x2, y2] = p(b);
  el("path", {d: `M${x1} ${y1} A${R} ${R} 0 0 1 ${x2} ${y2}`, fill: "none", stroke: col, "stroke-width": 14}, sC); };
arc(2, 4, C("--semiloose")); arc(4, 5, C("--semitight")); arc(5, 10, C("--tight"));
[2, 4, 5, 7, 10].forEach(v => txt(sC, CX + (R + 16) * Math.cos(ang(v)), CY - (R + 16) * Math.sin(ang(v)) + 4, v + "%", "mut", "middle"));
const needle = el("line", {x1: CX, y1: CY, x2: CX, y2: CY - R + 18, stroke: C("--ink"), "stroke-width": 3, "stroke-linecap": "round"}, sC);
el("circle", {cx: CX, cy: CY, r: 7, fill: C("--ink")}, sC);
const credV = txt(sC, CX, CY + 30, "", "big", "middle");

// 层标签
document.querySelectorAll("#gearCard .gTag").forEach(t => { const L = G.layers[+t.dataset.l];
  t.innerHTML = `<span class="dot"></span>第 ${L.n} 层<b>${L.name}</b>${L.desc}<br><span style="font-size:12px">解药:${L.fix}</span>`; });

// ---------- 动画状态 ----------
const S = {lever: 0, leverT: 0, gear: 0, fly: 0, pump: 0, sb: 0, sbT: 0, go: 0, goT: 0, need: 0, needT: 0};
const reduce = window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;

function knobX(gt, c){ return c == null ? gt.xs[1] : c > G.band ? gt.xs[2] : c < -G.band ? gt.xs[0] : gt.xs[1]; }
function pv(r, chg, yrs){ if (r == null || chg == null) return null; return Math.pow((1 + r / 100) / (1 + r / 100 + chg / 10000), yrs) - 1; }

function render(i){
  I = i; $("gSlider").value = i;
  const d = G.dates[i], st = G.st.map(s => s[i] || 0);
  $("gDate").textContent = `读数日期 ${d}` + (i === n - 1 ? "(最新)" : "(回放中)");
  const hot = st.map((s, k) => s === 2 ? k : -1).filter(k => k >= 0), warm = st.map((s, k) => s === 1 ? k : -1).filter(k => k >= 0);
  $("gNow").innerHTML = hot.length ? `市场卡在<span style="color:${C("--tight")}">第 ${Math.max(...hot) + 1} 层:${G.layers[Math.max(...hot)].name}</span>`
    : warm.length ? `没有层在运转,第 ${Math.max(...warm) + 1} 层${G.layers[Math.max(...warm)].name}警戒` : "四层都平静";
  $("gLadder").innerHTML = G.layers.map((L, k) => `<span class="s${st[k]}">${L.n} ${L.name}·${stName(st[k])}</span>`).join("");
  document.querySelectorAll("#gearCard .gTag .dot").forEach((e, k) => e.style.background = stCol(st[k]));

  // 杠杆
  const tc = G.tips_chg[i], r = G.tips10[i];
  S.leverT = clamp((tc || 0) * 0.4, -16, 16);
  ARMS.forEach(([y], k) => armLab[k].textContent = pc(pv(r, tc, y), 1));
  levTitle.textContent = `10 年实际利率 ${f1(r, 2)}%,四周 ${sg(tc)}BP`;
  $("gcLever").innerHTML = `<b>杠杆:实际利率 → 长久期估值</b>。利率端抬高,估值端下沉;数字 = 同样的利率变化对 1/10/20 年后一笔钱今天价值的影响。纳指四周 ${pc(G.qqq20[i])},离 60 日高点 ${pc(G.qqq_dd[i])}。`;

  // 齿轮
  const uc = G.ust5_chg[i];
  S.gearT = clamp((uc || 0) / 40, -1.5, 1.5);
  gAv.textContent = `${sg(uc)}BP`; gBv.textContent = pc(G.ief20[i]);
  gAv.style.fill = (uc || 0) > 0 ? C("--tight") : C("--loose"); gBv.style.fill = (G.ief20[i] || 0) < 0 ? C("--tight") : C("--loose");
  $("gcGear").innerHTML = `<b>齿轮:收益率 ↔ 债券价格</b>,永远反向转。四周变化:收益率 ${sg(uc)}BP,中期国债 ETF ${pc(G.ief20[i])}。转速 = 收益率变化快慢。`;

  // 离合器
  const sb = G.sb_corr[i], go = G.gr_corr[i];
  S.sbT = knobX(gSB, sb); S.goT = knobX(gGo, go);
  [[gSB, sb], [gGo, go]].forEach(([gt, c]) => { gt.val.textContent = c == null ? "—" : sg(c, 2);
    gt.mark.setAttribute("transform", `translate(${160 + clamp(c || 0, -1, 1) * 130} 150)`); });
  const sbName = sb == null ? "—" : sb > G.band ? "通胀挡:股债同涨同跌,债券不再对冲" : sb < -G.band ? "衰退挡:股跌债涨,债券是减震器" : "空挡:股债各走各的";
  $("gcSB").innerHTML = `<b>离合器 1:股债相关</b>(标普 SPY vs 中期国债 IEF)。现在挂在 <b>${sbName}</b>。市场怕通胀时,利率一涨股债一起跌;怕衰退时,资金从股票逃进债券。`;
  const dump = (G.gld5[i] || 0) < -0.02 && (G.tips5[i] || 0) < 0;
  gGo.ball.setAttribute("fill", dump ? C("--tight") : C("--accent"));
  if (dump) S.goT = gGo.xs[2];
  const goName = go == null ? "—" : dump ? "砍仓挡:实际利率在降,黄金却跌,被当提款机卖了" : go < -G.band ? "利率挡:实际利率涨金跌,正常咬合" : go > G.band ? "同向:黄金与实际利率一起涨,央行买盘或避险主导" : "脱钩:黄金不看实际利率";
  $("gcGold").innerHTML = `<b>离合器 2:黄金 vs 实际利率</b>(GLD 日收益 vs 10 年 TIPS 日变化)。现在 <b>${goName}</b>。黄金 5 日 ${pc(G.gld5[i])},实际利率 5 日 ${sg(G.tips5[i])}BP。`;

  // 飞轮
  const fv = G.fly[i] || 0;
  S.flyT = fv;
  flyDir.textContent = fv > 0.15 ? "强平中" : fv < -0.15 ? "轧空中" : Math.abs(fv) > 0 ? "开始转动" : "静止";
  flyDir.style.fill = fv > 0.15 ? C("--tight") : fv < -0.15 ? C("--loose") : C("--muted");
  flySpd.textContent = `转速 ${Math.round(Math.abs(fv) * 100)}%`;
  flyBar.setAttribute("width", 120 * Math.abs(fv)); flyBar.setAttribute("fill", fv < 0 ? C("--loose") : C("--tight"));
  flyMark.setAttribute("fill", fv < 0 ? C("--loose") : C("--tight"));
  $("gcFly").innerHTML = `<b>飞轮:强平 / 轧空</b>。没有限速器的正反馈:逆时针 = 越跌越卖,顺时针 = 空头被迫回补,越涨越买。`;
  const mg = D.watch.margin;
  $("gtFly").innerHTML = `<div class="ck">纳指 5 日 <b>${pc(G.qqq5[i])}</b>(跌超 2% 才开始推飞轮,跌 8% 满速)</div>
    <div class="ck">VIX <b>${f1(G.vix[i])}</b>(高于 18 才算有恐慌,32 时满速)</div>
    <div class="ck">黄金与纳指近 20 日同日双跌 <b>${G.dd20[i] ?? "—"}</b> 天</div>
    <div class="ck">轧空:纳指 5 日涨超 4%,此前回撤超 8%,且近 10 日 VIX 高于 22</div>
    ${mg && i === n - 1 ? `<div class="ck">飞轮有多重:券商融资余额 ${mg.value_bn.toLocaleString()} 十亿美元(${mg.month}),同比 ${pc(mg.yoy)}</div>` : ""}`;

  // 抽水泵
  const pmp = G.pump[i] || 0;
  S.pumpT = pmp / 2;
  water.setAttribute("y", 42 + pmp * 22); water.setAttribute("height", 116 - pmp * 22);
  pumpTxt.textContent = `${pmp}/4`;
  $("gcPump").innerHTML = `<b>抽水泵:现金挤兑</b>。缺现金时不管资产好坏,先卖最好卖的;转速 = 满足的抽水条件数,水位 = 剩下的现金。`;
  const ck = (ok, s) => `<div class="ck">${ok ? "✓" : "✗"} ${s}</div>`;
  $("gtPump").innerHTML = ck(G.sofr[i] >= 3, `回购管道偏紧:SOFR − IORB <b>${f1(G.sofr[i], 0)}BP</b>(≥ 3)`) +
    ck(G.dd20[i] >= 2 && G.qqq5[i] < 0 && G.gld5[i] < 0, `黄金和纳指一起被卖:近 5 日都跌,20 日内同日双跌 <b>${G.dd20[i] ?? "—"}</b> 天(≥ 2)`) +
    ck(G.usd[i] > 2, `美元走强抽走全球美元:广义美元四周 <b>${sg(G.usd[i], 1)}%</b>(> 2%)`) +
    ck(G.jpy[i] < -3, `日元套息平仓:美元兑日元四周 <b>${sg(G.jpy[i], 1)}%</b>(< −3%)`);

  // 信用
  const hy = G.hy[i];
  S.needT = hy == null ? 0 : (Math.PI / 2 - ang(hy)) * 180 / Math.PI;
  credV.textContent = "高收益债利差 " + (hy == null ? "无数据" : f1(hy, 2) + "%");
  $("gcCredit").innerHTML = `<b>信用压力表</b>。高收益债利差 = 市场借钱给差公司要多收的利息;超过 5% 或四周急升 1 个百分点,说明信用层在运转。`;
  $("gtCredit").innerHTML = `<div class="ck">高收益债利差 <b>${f1(hy, 2)}%</b>,四周 ${sg(G.hy_chg[i])}BP</div>
    <div class="ck">CCC 级利差 <b>${f1(G.ccc[i], 2)}%</b>(预警线 ${G.ccc_line}%,${G.ccc[i] > G.ccc_line ? "已超过:最弱的公司先借不到钱" : "未超过"})</div>
    ${G.ccc[i] > G.ccc_line && hy != null && hy < 5 ? `<div class="ck sub">只有 CCC 超线、整体利差还低,说明压力停在最弱的一档,还没传到整个信用市场。</div>` : ""}`;
}

// ---------- 动画循环 ----------
let t0 = null;
function frame(t){
  const dt = t0 == null ? 0 : Math.min(0.05, (t - t0) / 1000); t0 = t;
  const k = reduce ? 1 : 1 - Math.pow(0.002, dt);
  S.lever += (S.leverT - S.lever) * k;
  beam.setAttribute("transform", `translate(${PIV} ${PY}) rotate(${S.lever}) translate(${-PIV} 0)`);
  const sp = reduce ? 0 : 1;
  gA.angle += (S.gearT || 0) * 60 * dt * sp; gB.angle -= (S.gearT || 0) * 60 * dt * sp;
  gA.rot.setAttribute("transform", `rotate(${gA.angle})`); gB.rot.setAttribute("transform", `rotate(${gB.angle})`);
  S.fly += (S.flyT || 0) * -240 * dt * sp; flyRot.setAttribute("transform", `rotate(${S.fly})`);
  pump.angle += (S.pumpT || 0) * 200 * dt * sp; pump.rot.setAttribute("transform", `rotate(${pump.angle})`);
  S.sb += (S.sbT - S.sb) * k; S.go += (S.goT - S.go) * k;
  if (!S.init){ S.sb = S.sbT; S.go = S.goT; S.init = 1; }
  gSB.knob.setAttribute("transform", `translate(${S.sb} ${gSB.ky})`); gGo.knob.setAttribute("transform", `translate(${S.go} ${gGo.ky})`);
  S.need += (S.needT - S.need) * k; needle.setAttribute("transform", `rotate(${S.need} ${CX} ${CY})`);
  requestAnimationFrame(frame);
}

// ---------- 控件 ----------
const sl = $("gSlider"); sl.max = n - 1;
sl.addEventListener("input", () => { stop(); render(+sl.value); });
let timer = null;
function stop(){ if (timer){ clearInterval(timer); timer = null; $("gPlay").textContent = "▶ 回放近三年"; } }
$("gPlay").addEventListener("click", () => {
  if (timer) return stop();
  let i = I >= n - 1 ? 0 : I; $("gPlay").textContent = "⏸ 暂停";
  timer = setInterval(() => { render(i); i += 2; if (i >= n){ render(n - 1); stop(); } }, 60);
});
$("gToday").addEventListener("click", () => { stop(); render(n - 1); });
document.querySelectorAll("#gearCard [data-gj]").forEach(b => b.addEventListener("click", () => {
  stop(); let k = G.dates.findIndex(x => x >= b.dataset.gj); render(k < 0 ? n - 1 : k); }));
render(n - 1);
requestAnimationFrame(frame);
})();
</script>
"""
