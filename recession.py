"""衰退概率:给仪表盘补上“实体经济”这一层。前面的指标几乎都是金融面的。

三个互相独立的读数,都是公开发表、免费的方法:
  1 收益率曲线模型(纽约联储,Estrella & Mishkin)
      未来 12 个月内处在衰退中的概率 = Φ(−0.5333 − 0.6330 × 利差)
      利差 = 10 年美债月均收益率 − 3 个月国库券月均收益率(贴现率换算成债券等价收益率)。
      这是“领先”读数:看的是 12 个月以后。纽约联储官网的 Prob_Rec.xlsx 能下载到时用来核对。
  2 Sahm 规则(FRED SAHMREALTIME,实时版本)
      失业率 3 个月均值比过去 12 个月最低点高出 0.5 个百分点 → 衰退大概率已经开始。是“同步”读数。
  3 Chauvet–Piger 平滑衰退概率(FRED RECPROUSM156N)
      用就业、工业产出、收入、销售四个同步指标估计“现在是否在衰退中”,公布有约 2 个月滞后。

只做展示,不进仓位规则。输出 output/recession.json,页面 HTML 和脚本也在本文件里,由 dashboard.py 插入。
"""
from __future__ import annotations

import io
import json
import math
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
ALPHA, BETA = -0.5333, -0.6330
FRED_IDS = {"GS10": "10 年美债月均收益率", "TB3MS": "3 个月国库券月均收益率(贴现率)",
            "SAHMREALTIME": "Sahm 规则实时值", "RECPROUSM156N": "Chauvet–Piger 平滑衰退概率",
            "USREC": "NBER 衰退月份"}
NYFED_XLS = "https://www.newyorkfed.org/medialibrary/media/research/capital_markets/Prob_Rec.xlsx"
# 档位:(下限, 名称, 颜色变量)
LEVELS = [(0, "低", "--loose"), (20, "需要留意", "--semitight"), (40, "偏高", "--tight")]
SAHM_LINE = 0.5


def _get(url: str, params: dict | None = None, retries: int = 3) -> requests.Response:
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            return r
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"{url}: {last}")


def fred(sid: str) -> pd.Series:
    """fredgraph.csv 不需要 API key;不带自定义 User-Agent(FRED 会卡住)。"""
    r = _get("https://fred.stlouisfed.org/graph/fredgraph.csv", {"id": sid, "cosd": "1950-01-01"})
    df = pd.read_csv(io.StringIO(r.text))
    s = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    s.index = pd.to_datetime(df.iloc[:, 0]).dt.to_period("M").dt.to_timestamp()
    s = s.dropna()
    if s.empty:
        raise RuntimeError("空序列")
    return s.groupby(level=0).last()


def load(out: Path, offline: bool = False) -> tuple[pd.DataFrame, dict]:
    """抓 FRED;失败的序列用上次保存的 output/recession_raw.csv 顶上。"""
    backup = out / "recession_raw.csv"
    old = pd.read_csv(backup, index_col=0, parse_dates=True) if backup.exists() else pd.DataFrame()
    cols, status = {}, {}
    for sid, name in FRED_IDS.items():
        s, note = None, ""
        if not offline:
            try:
                s = fred(sid)
            except Exception as e:  # noqa: BLE001
                note = f"抓取失败,用上次的数据:{str(e)[:80]}"
                print(f"[警告] 衰退概率 {sid}: {e}", file=sys.stderr)
        if s is None and sid in old:
            s = old[sid].dropna()
        if s is not None and len(s):
            cols[sid] = s
        status[sid] = {"src": f"FRED {sid}({name})", "ok": s is not None and len(s) > 0 and not note,
                       "note": note or (f"到 {s.index[-1]:%Y-%m}" if s is not None and len(s) else "无数据")}
    raw = pd.DataFrame(cols).sort_index()
    if not raw.empty:
        raw.to_csv(backup, float_format="%.4f")
    return raw, status


def bey(discount_pct: pd.Series) -> pd.Series:
    """3 个月国库券贴现率 → 债券等价收益率(纽约联储模型用的口径)。"""
    d = discount_pct / 100
    return 100 * 365 * d / (360 - 91 * d)


def norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def probit(spread: pd.Series) -> pd.Series:
    return (ALPHA + BETA * spread).map(norm_cdf) * 100


def level(p) -> tuple[str, str]:
    r = LEVELS[0]
    for L in LEVELS:
        if p is not None and p >= L[0]:
            r = L
    return r[1], r[2]


def _r(v, nd=1):
    return None if v is None or pd.isna(v) else round(float(v), nd)


def episodes(prob: pd.Series, rec: pd.Series) -> list[dict]:
    """每次衰退开始前 6–18 个月里,模型给出的最高概率。用来看它以前灵不灵。"""
    rec = rec.dropna()
    starts = rec[(rec == 1) & (rec.shift(1) == 0)].index
    out = []
    for t in starts:
        w = prob[(prob.index >= t - pd.DateOffset(months=18)) & (prob.index <= t - pd.DateOffset(months=6))].dropna()
        if len(w):
            out.append({"start": t.strftime("%Y-%m"), "max": _r(w.max()), "at": w.idxmax().strftime("%Y-%m")})
    return out


def false_alarms(prob: pd.Series, rec: pd.Series, line: float = 30) -> list[str]:
    """概率越过 line 之后 24 个月内没有出现衰退的月份(按每段第一次越线记)。"""
    rec = rec.dropna()
    above = prob >= line
    first = above & ~above.shift(1, fill_value=False)
    out = []
    for t in prob[first].index:
        nxt = rec[(rec.index > t) & (rec.index <= t + pd.DateOffset(months=24))]
        if len(nxt) and nxt.max() == 0 and nxt.index[-1] >= t + pd.DateOffset(months=24):
            out.append(t.strftime("%Y-%m"))
    return out


def run(out: Path, offline: bool = False) -> dict | None:
    raw, status = load(out, offline)
    if raw.empty or not {"GS10", "TB3MS"} <= set(raw):
        return None
    spread = (raw["GS10"] - bey(raw["TB3MS"])).dropna()
    prob = probit(spread)
    rec = raw["USREC"] if "USREC" in raw else pd.Series(dtype=float)
    sahm = raw["SAHMREALTIME"].dropna() if "SAHMREALTIME" in raw else pd.Series(dtype=float)
    cp = raw["RECPROUSM156N"].dropna() if "RECPROUSM156N" in raw else pd.Series(dtype=float)

    # 纽约联储官方数值核对(只记录差异,不影响结果)
    check = None
    if not offline:
        try:
            x = pd.read_excel(io.BytesIO(_get(NYFED_XLS).content))
            print(f"  纽约联储 Prob_Rec.xlsx 列:{list(x.columns)[:6]},最后一行:{x.dropna(how='all').tail(1).values.tolist()}")
            check = "已下载纽约联储 Prob_Rec.xlsx(见运行日志)"
        except Exception as e:  # noqa: BLE001
            print(f"[提示] 纽约联储 Prob_Rec.xlsx 下载失败,只用公式计算:{e}", file=sys.stderr)

    t = prob.index[-1]
    tgt = t + pd.DateOffset(months=12)
    lv, color = level(prob.iloc[-1])
    sahm_v = _r(sahm.iloc[-1], 2) if len(sahm) else None
    cp_v = _r(cp.iloc[-1], 1) if len(cp) else None
    signals = sum([prob.iloc[-1] >= 30, sahm_v is not None and sahm_v >= SAHM_LINE, cp_v is not None and cp_v >= 50])
    start = pd.Timestamp("1960-01-01")
    months = pd.date_range(start, max(prob.index[-1], sahm.index[-1] if len(sahm) else start), freq="MS")
    rr = rec.reindex(months)
    data = {
        "asof": t.strftime("%Y-%m"),
        "target": tgt.strftime("%Y-%m"),
        "prob": _r(prob.iloc[-1]),
        "prob_chg12": _r(prob.iloc[-1] - prob.get(t - pd.DateOffset(months=12), float("nan"))),
        "spread": _r(spread.iloc[-1], 2),
        "gs10": _r(raw["GS10"].dropna().iloc[-1], 2),
        "tb3": _r(bey(raw["TB3MS"]).dropna().iloc[-1], 2),
        "level": lv, "color": color,
        "levels": [[a, b, c] for a, b, c in LEVELS],
        "sahm": {"value": sahm_v, "asof": sahm.index[-1].strftime("%Y-%m") if len(sahm) else None,
                 "line": SAHM_LINE, "max12": _r(sahm.iloc[-12:].max(), 2) if len(sahm) else None},
        "cp": {"value": cp_v, "asof": cp.index[-1].strftime("%Y-%m") if len(cp) else None},
        "signals": int(signals),
        "episodes": episodes(prob, rec) if len(rec) else [],
        "false_alarms": false_alarms(prob, rec) if len(rec) else [],
        "inverted_months": int((spread.iloc[-24:] < 0).sum()),
        "check": check,
        "sources": status,
        "months": [d.strftime("%Y-%m") for d in months],
        "series": {
            "prob": [_r(v) for v in prob.reindex(months)],
            # 模型给出的是 12 个月后的概率,画在目标月份上方便和实际衰退对照
            "prob_target": [_r(v) for v in prob.shift(12, freq="MS").reindex(months)],
            "sahm": [_r(v, 2) for v in sahm.reindex(months)],
            "cp": [_r(v) for v in cp.reindex(months)],
            "rec": [None if pd.isna(v) else int(v) for v in rr],
        },
    }
    (out / "recession.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return data


def report_lines(d: dict | None) -> list[str]:
    if not d:
        return []
    s, c = d["sahm"], d["cp"]
    L = ["## 衰退概率(实体经济,只做观察)", "",
         f"- **收益率曲线模型(纽约联储)**:按 {d['asof']} 的利差 {d['spread']:+.2f} 个百分点"
         f"(10 年 {d['gs10']}% − 3 个月 {d['tb3']}%),到 {d['target']} 处在衰退中的概率 **{d['prob']}%**({d['level']}),"
         f"比一年前 {d['prob_chg12']:+.1f} 个百分点;近 24 个月有 {d['inverted_months']} 个月倒挂。",
         f"- **Sahm 规则**:{s['asof']} 为 {s['value']}(≥ {s['line']} 表示衰退大概率已开始)。",
         f"- **Chauvet–Piger 平滑衰退概率**:{c['asof']} 为 {c['value']}%(≥ 50% 视为身处衰退)。",
         f"- 三个读数里有 {d['signals']} 个亮灯。"]
    if d["episodes"]:
        L.append("- 历史对照(衰退开始前 6–18 个月模型最高概率):"
                 + ";".join(f"{e['start']} 前 {e['max']}%" for e in d["episodes"]) + "。")
    L.append("")
    return L


# ---------------------------------------------------------------------------
# 页面部分
# ---------------------------------------------------------------------------
HTML = r"""
  <div class="card" id="recCard">
    <h2>衰退概率:实体经济会不会掉下去</h2>
    <div class="sub">前面的指标大多看金融市场。这里补上实体经济:一个领先读数(收益率曲线,看 12 个月后)和两个同步读数(失业率、四个同步指标),都是公开发表的方法。只做观察,不进仓位规则。</div>
    <div class="hero" style="margin-top:12px">
      <div>
        <div class="sub" id="rAsof"></div>
        <div class="big" id="rNow"></div>
        <div style="margin-top:8px"><span class="badge" id="rLevel"></span></div>
      </div>
      <div class="bPil" id="rPil"></div>
    </div>
    <div class="sub" id="rLine" style="margin-top:10px"></div>
    <div class="row" style="margin-top:12px" id="rRange"></div>
    <div id="cRec" class="chart" style="height:380px"></div>
    <div class="note">灰色底色是 NBER 认定的衰退月份。蓝线是收益率曲线模型的概率,画在它预测的那个月(12 个月后)上,所以蓝线的尖峰应该落在灰色区域里才算“预测对了”;点图例可以切换成画在计算当月。橙线是 Sahm 规则(右轴,0.5 以上算触发),绿线是 Chauvet–Piger 概率。</div>
    <h3 class="bH3">历史上灵不灵</h3>
    <div class="scroll"><table id="rEp"></table></div>
    <div class="note" id="rFa"></div>
    <details><summary>方法和数据来源</summary><div class="note">
      收益率曲线模型:Estrella & Mishkin(1996),纽约联储每月公布。概率 = Φ(−0.5333 − 0.6330 × 利差),利差 = 10 年美债月均收益率 − 3 个月国库券月均收益率(贴现率换算成债券等价收益率),Φ 是标准正态分布函数。它在 1960 年代以来每次衰退前都给过信号,但 2022–2024 年倒挂很深却没有出现衰退,是它最大的一次误报;疫情那次是外部冲击,它也没法提前知道。<br>
      Sahm 规则:Claudia Sahm(2019),用 FRED 的实时版本 SAHMREALTIME,即当时公布的失业率,不是后来修订过的。<br>
      Chauvet–Piger:用非农就业、工业产出、扣除转移支付的实际个人收入、实际制造业和贸易销售估计,FRED RECPROUSM156N,有约 2 个月滞后。<br>
      <span id="rSrc"></span></div></details>
  </div>
"""

SCRIPT = r"""
<script>
(function(){
const card = document.getElementById("recCard");
const R = window.__RECESSION__;
if (!R || !window.Plotly){ if (!R) card.style.display = "none"; return; }
const $ = id => document.getElementById(id);
const C = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const f1 = v => v == null ? "—" : Number(v).toFixed(1);
const sg = v => v == null ? "—" : (v >= 0 ? "+" : "") + Number(v).toFixed(2);

$("rAsof").textContent = `到 ${R.target} 处在衰退中的概率(按 ${R.asof} 的收益率曲线)`;
$("rNow").textContent = f1(R.prob) + "%";
$("rLevel").textContent = R.level; $("rLevel").style.background = C(R.color);
const tiles = [
  ["10 年 − 3 个月利差", sg(R.spread), `${R.asof}`, R.spread != null && R.spread < 0],
  ["Sahm 规则", R.sahm.value == null ? "—" : R.sahm.value.toFixed(2), `${R.sahm.asof || "—"} · ≥ ${R.sahm.line} 触发`, R.sahm.value != null && R.sahm.value >= R.sahm.line],
  ["Chauvet–Piger", R.cp.value == null ? "—" : f1(R.cp.value) + "%", `${R.cp.asof || "—"} · ≥ 50% 身处衰退`, R.cp.value != null && R.cp.value >= 50],
  ["亮灯", `${R.signals} / 3`, "模型 ≥30% · Sahm · C–P", R.signals >= 2]];
$("rPil").innerHTML = tiles.map(([a, b, c, bad]) => `<div>${a}<b style="color:${bad ? C("--tight") : C("--ink")}">${b}</b><span>${c}</span></div>`).join("");
$("rLine").textContent = `模型概率比一年前 ${R.prob_chg12 == null ? "—" : (R.prob_chg12 >= 0 ? "+" : "") + f1(R.prob_chg12)} 个百分点;近 24 个月里有 ${R.inverted_months} 个月利差为负(倒挂)。档位:<20% 低,20–40% 需要留意,≥40% 偏高。`;

const x = R.months.map(m => m + "-01");
// 衰退底色:把连续的衰退月份合成一块
const shapes = []; let s0 = null;
R.series.rec.forEach((v, i) => {
  if (v === 1 && s0 === null) s0 = x[i];
  if ((v !== 1 || i === x.length - 1) && s0 !== null){ shapes.push({type: "rect", yref: "paper", y0: 0, y1: 1, x0: s0, x1: x[i], fillcolor: C("--neutral"), opacity: 0.22, line: {width: 0}, layer: "below"}); s0 = null; }
});
shapes.push({type: "line", xref: "paper", x0: 0, x1: 1, y0: 0.5, y1: 0.5, yref: "y2", line: {color: C("--p2"), width: 1, dash: "dash"}});
const tr = [
  {x, y: R.series.prob_target, name: "曲线模型(画在预测的月份)", type: "scatter", mode: "lines", line: {color: C("--p1"), width: 2}, hovertemplate: "曲线模型 %{y:.0f}%<extra></extra>"},
  {x, y: R.series.prob, name: "曲线模型(画在计算当月)", type: "scatter", mode: "lines", line: {color: C("--p1"), width: 1.2, dash: "dot"}, visible: "legendonly", hovertemplate: "当月算出 %{y:.0f}%<extra></extra>"},
  {x, y: R.series.cp, name: "Chauvet–Piger", type: "scatter", mode: "lines", line: {color: C("--p5"), width: 1.4}, hovertemplate: "C–P %{y:.0f}%<extra></extra>"},
  {x, y: R.series.sahm, name: "Sahm(右轴)", type: "scatter", mode: "lines", yaxis: "y2", line: {color: C("--p2"), width: 1.4}, hovertemplate: "Sahm %{y:.2f}<extra></extra>"}];
const lay = {paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: {color: C("--ink"), size: 12},
  margin: {l: 40, r: 40, t: 12, b: 30}, hovermode: "x unified", legend: {orientation: "h", y: -0.12},
  xaxis: {type: "date", gridcolor: C("--grid"), linecolor: C("--line"), hoverformat: "%Y-%m"},
  yaxis: {range: [0, 100], gridcolor: C("--grid"), ticksuffix: "%"},
  yaxis2: {overlaying: "y", side: "right", range: [-0.5, 4], showgrid: false}, shapes};
const cfg = {displaylogo: false, responsive: true, modeBarButtonsToRemove: ["lasso2d", "select2d"]};
Plotly.newPlot("cRec", tr, lay, cfg);
const rb = $("rRange"), last = x[x.length - 1];
[["全部", null], ["1990 年起", "1990-01-01"], ["近 10 年", (Number(last.slice(0, 4)) - 10) + last.slice(4)]].forEach(([t, s], i) => {
  const b = document.createElement("button"); b.textContent = t; if (i === 0) b.classList.add("on");
  b.onclick = () => { Plotly.relayout("cRec", {"xaxis.range": [s || x[0], last]}); [...rb.children].forEach(c => c.classList.toggle("on", c === b)); };
  rb.appendChild(b);
});
$("rEp").innerHTML = `<tr><th>衰退开始</th><th>开始前 6–18 个月模型最高概率</th><th>出现在</th></tr>` +
  R.episodes.map(e => `<tr><td style="white-space:nowrap">${e.start}</td><td style="color:${e.max >= 30 ? C("--tight") : C("--muted")}">${f1(e.max)}%</td><td style="white-space:nowrap">${e.at}</td></tr>`).join("");
$("rFa").textContent = R.false_alarms.length ? `误报:概率第一次升到 30% 以上、之后 24 个月却没有衰退的有 ${R.false_alarms.join("、")}。` : "";
$("rSrc").innerHTML = "本次抓取:" + Object.values(R.sources).map(s => `${s.ok ? "✓" : "✗"} ${s.src}(${s.note})`).join(";");
})();
</script>
"""


def page(out: Path) -> tuple[str, str]:
    p = out / "recession.json"
    data = p.read_text(encoding="utf-8") if p.exists() else "null"
    return HTML, f"<script>window.__RECESSION__ = {data};</script>" + SCRIPT


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "output"))
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    print("\n".join(report_lines(run(Path(a.out), a.offline)) or ["没有数据"]))
